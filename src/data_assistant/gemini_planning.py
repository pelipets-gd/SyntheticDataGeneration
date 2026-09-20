"""Gemini-backed generation planning and validated feedback contracts."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, ContextManager, Iterator, Literal, Protocol

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, model_validator

from data_assistant.querying import QueryError, compile_readonly_select
from data_assistant.schema import DatabaseSchema, TableSchema
from data_assistant.settings import Settings

_QUERY_VALIDATION_SCHEMA = "dataset_" + ("0" * 32)


class ColumnGenerationPlan(BaseModel):
    """Semantic generation guidance for one schema column."""

    model_config = ConfigDict(extra="forbid")

    column_name: str
    generator: str = Field(min_length=1)
    semantic_type: str | None = None
    locale: str | None = None
    minimum: float | str | None = None
    maximum: float | str | None = None
    distribution: Literal["uniform", "normal", "weighted", "sequential"] = "uniform"
    categories: list[str] = Field(default_factory=list)
    nullable_probability: float = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_range(self) -> ColumnGenerationPlan:
        if (
            isinstance(self.minimum, (int, float))
            and isinstance(self.maximum, (int, float))
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum must not exceed maximum")
        return self


class TableGenerationPlan(BaseModel):
    """Generation guidance for one database table."""

    model_config = ConfigDict(extra="forbid")

    table_name: str
    row_count: int = Field(ge=1, le=1_000_000)
    columns: list[ColumnGenerationPlan]


class DatabaseGenerationPlan(BaseModel):
    """Structured generation guidance for a parsed database schema."""

    model_config = ConfigDict(extra="forbid")

    database_description: str = Field(min_length=1)
    tables: list[TableGenerationPlan]

    def validate_against(self, schema: DatabaseSchema) -> DatabaseGenerationPlan:
        """Validate every planned table and column against the parsed schema."""
        seen_tables: set[str] = set()
        for table_plan in self.tables:
            table_key = table_plan.table_name.casefold()
            if table_key in seen_tables:
                raise ValueError(f"Duplicate table plan {table_plan.table_name!r}")
            seen_tables.add(table_key)
            try:
                table = schema.table(table_plan.table_name)
            except KeyError as error:
                raise ValueError(str(error)) from error
            table_plan.table_name = table.name

            seen_columns: set[str] = set()
            for column_plan in table_plan.columns:
                column_key = column_plan.column_name.casefold()
                if column_key in seen_columns:
                    raise ValueError(
                        f"Duplicate column plan {table.name}.{column_plan.column_name}"
                    )
                seen_columns.add(column_key)
                try:
                    column = table.column(column_plan.column_name)
                except KeyError as error:
                    raise ValueError(str(error)) from error
                column_plan.column_name = column.name
                if not column.nullable and column_plan.nullable_probability:
                    raise ValueError(
                        f"Non-nullable column {table.name}.{column.name} cannot generate nulls"
                    )
            required_columns = {
                column.name.casefold(): column.name
                for column in table.columns
                if not _is_key_column(table, column.name)
            }
            missing_columns = [
                name
                for key, name in required_columns.items()
                if key not in seen_columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Table {table.name!r} is missing non-key columns: "
                    f"{', '.join(missing_columns)}"
                )

        missing_tables = [
            table.name
            for table in schema.tables
            if table.name.casefold() not in seen_tables
        ]
        if missing_tables:
            raise ValueError(
                f"Generation plan is missing table plans: {', '.join(missing_tables)}"
            )
        return self


class NaturalLanguageQuery(BaseModel):
    """A SELECT that answers a question about one stored dataset."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1, max_length=8_000)
    explanation: str = Field(min_length=1, max_length=2_000)


class RegenerateColumnsOperation(BaseModel):
    """Request regeneration of selected non-key columns."""

    model_config = ConfigDict(extra="forbid")

    operation_type: Literal["regenerate_columns"] = "regenerate_columns"
    table_name: str
    columns: list[str] = Field(min_length=1)
    instruction: str | None = Field(default=None, max_length=1_000)


class TransformValuesOperation(BaseModel):
    """Request a bounded numeric or date transformation."""

    model_config = ConfigDict(extra="forbid")

    operation_type: Literal["transform_values"] = "transform_values"
    table_name: str
    column: str
    transformation: Literal["add", "multiply", "clamp", "shift_days"]
    value: float | None = Field(default=None, ge=-1_000_000, le=1_000_000)
    minimum: float | None = Field(default=None, ge=-1_000_000, le=1_000_000)
    maximum: float | None = Field(default=None, ge=-1_000_000, le=1_000_000)

    @model_validator(mode="after")
    def validate_parameters(self) -> TransformValuesOperation:
        if self.transformation in {"add", "multiply", "shift_days"} and self.value is None:
            raise ValueError(f"{self.transformation} requires value")
        if self.transformation == "shift_days" and self.value is not None:
            if not self.value.is_integer():
                raise ValueError("shift_days value must be a whole number")
            if not -3650 <= self.value <= 3650:
                raise ValueError("shift_days value must be less than or equal to 3650")
        if self.transformation == "clamp":
            if self.minimum is None or self.maximum is None:
                raise ValueError("clamp requires minimum and maximum")
            if self.minimum > self.maximum:
                raise ValueError("minimum must not exceed maximum")
        return self


class ReplaceCategoricalValuesOperation(BaseModel):
    """Request categorical substitutions in one non-key column."""

    model_config = ConfigDict(extra="forbid")

    operation_type: Literal["replace_categorical_values"] = "replace_categorical_values"
    table_name: str
    column: str
    replacements: dict[str, str] = Field(min_length=1)


FeedbackOperation = (
    RegenerateColumnsOperation
    | TransformValuesOperation
    | ReplaceCategoricalValuesOperation
)


class GeminiClient(Protocol):
    """Minimal fake-friendly Gemini client interface used by the service."""

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any: ...

    def generate_content_stream(
        self, *, model: str, contents: str, config: Any
    ) -> Iterator[Any]: ...


class Tracer(Protocol):
    """Minimal tracing interface used around model calls."""

    def trace(self, name: str, metadata: dict[str, Any]) -> ContextManager[None]: ...


class NoOpTracer:
    """Tracer used when Langfuse is unavailable or not configured."""

    @contextmanager
    def trace(self, name: str, metadata: dict[str, Any]) -> Iterator[None]:
        del name, metadata
        yield


class GoogleGenAIClient:
    """Adapter from the Google GenAI client to the service interface."""

    def __init__(self, client: genai.Client) -> None:
        self._client = client

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        return self._client.models.generate_content(
            model=model, contents=contents, config=config
        )

    def generate_content_stream(
        self, *, model: str, contents: str, config: Any
    ) -> Iterator[Any]:
        return self._client.models.generate_content_stream(
            model=model, contents=contents, config=config
        )


class LangfuseTracer:
    """Adapter for the optional Langfuse v2 tracing client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @contextmanager
    def trace(self, name: str, metadata: dict[str, Any]) -> Iterator[None]:
        trace = self._client.trace(name=name, metadata=metadata)
        span = trace.span(name=name)
        try:
            yield
        finally:
            try:
                span.end()
            finally:
                self._client.flush()


def create_google_client(settings: Settings) -> GoogleGenAIClient:
    """Create a Vertex AI Google GenAI client from injected settings."""
    return GoogleGenAIClient(
        genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.vertex_ai_location,
        )
    )


def create_tracer(settings: Settings) -> Tracer:
    """Create an optional Langfuse tracer without reading process environment."""
    if not settings.langfuse_enabled:
        return NoOpTracer()
    try:
        from langfuse import Langfuse
    except ImportError:
        return NoOpTracer()
    return LangfuseTracer(
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    )


class GeminiPlanningService:
    """Request structured plans, explanations, and safe feedback operations."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: GeminiClient | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._tracer = tracer

    @property
    def client(self) -> GeminiClient:
        if self._client is None:
            self._client = create_google_client(self._settings)
        return self._client

    @property
    def tracer(self) -> Tracer:
        if self._tracer is None:
            self._tracer = create_tracer(self._settings)
        return self._tracer

    def request_plan(
        self, schema: DatabaseSchema, *, temperature: float = 0.2
    ) -> DatabaseGenerationPlan:
        """Request and schema-validate a generation plan."""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DatabaseGenerationPlan,
            temperature=temperature,
        )
        metadata = _trace_metadata(schema, self._settings.gemini_model)
        with self.tracer.trace("gemini.plan", metadata):
            response = self.client.generate_content(
                model=self._settings.gemini_model,
                contents=_plan_prompt(schema),
                config=config,
            )
        return _parse_plan(response).validate_against(schema)

    def stream_explanation(
        self,
        schema: DatabaseSchema,
        request: str,
        *,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Yield nonempty model text chunks for concise user-facing status."""
        config = types.GenerateContentConfig(temperature=temperature)
        metadata = _trace_metadata(schema, self._settings.gemini_model)
        with self.tracer.trace("gemini.stream", metadata):
            for response in self.client.generate_content_stream(
                model=self._settings.gemini_model,
                contents=(
                    "Give a concise user-facing explanation or status. "
                    f"Request: {request}\nSchema summary: {_schema_payload(schema)}"
                ),
                config=config,
            ):
                text = getattr(response, "text", None)
                if text:
                    yield text

    def request_feedback_operations(
        self,
        schema: DatabaseSchema,
        table_name: str,
        feedback: str,
        *,
        temperature: float = 0.1,
    ) -> list[FeedbackOperation]:
        """Request and validate non-executing feedback operations for one table."""
        try:
            table = schema.table(table_name)
        except KeyError as error:
            raise ValueError(str(error)) from error
        declarations = _feedback_declarations(table)
        config = types.GenerateContentConfig(
            temperature=temperature,
            tools=(
                [types.Tool(function_declarations=declarations)]
                if declarations
                else None
            ),
        )
        metadata = {
            "table": table.name,
            "column_count": len(table.columns),
            "model": self._settings.gemini_model,
        }
        with self.tracer.trace("gemini.feedback", metadata):
            response = self.client.generate_content(
                model=self._settings.gemini_model,
                contents=(
                    "Select only the supplied safe tools to represent this feedback. "
                    f"Feedback: {feedback}"
                ),
                config=config,
            )
        return [
            _parse_feedback_call(call, table)
            for call in (getattr(response, "function_calls", None) or [])
        ]

    def request_query(
        self,
        schema: DatabaseSchema,
        question: str,
        *,
        temperature: float = 0.1,
    ) -> NaturalLanguageQuery:
        """Request a read-only SELECT for a natural-language question.

        # Errors
        Raises `ValueError` when the question is empty, the model returns no
        plan, or the SQL is not a single SELECT of tables in `schema`.
        """
        if not question.strip():
            raise ValueError("question must not be empty")
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=NaturalLanguageQuery,
            temperature=temperature,
        )
        metadata = _trace_metadata(schema, self._settings.gemini_model)
        with self.tracer.trace("gemini.query", metadata):
            response = self.client.generate_content(
                model=self._settings.gemini_model,
                contents=_query_prompt(schema, question),
                config=config,
            )
        plan = _parse_query(response)
        try:
            compile_readonly_select(plan.sql, schema, _QUERY_VALIDATION_SCHEMA)
        except QueryError as error:
            raise ValueError(str(error)) from error
        return plan


def _parse_plan(response: Any) -> DatabaseGenerationPlan:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, DatabaseGenerationPlan):
        return parsed
    if parsed is not None:
        return DatabaseGenerationPlan.model_validate(parsed)
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini returned no structured generation plan")
    return DatabaseGenerationPlan.model_validate(json.loads(text))


def _parse_feedback_call(call: Any, table: TableSchema) -> FeedbackOperation:
    name = getattr(call, "name", None)
    args = dict(getattr(call, "args", None) or {})
    args["table_name"] = table.name
    if name == "regenerate_columns":
        operation: FeedbackOperation = RegenerateColumnsOperation.model_validate(args)
        operation.columns = [
            _editable_column(table, column_name).name
            for column_name in operation.columns
        ]
        return operation
    if name == "transform_values":
        operation = TransformValuesOperation.model_validate(args)
        column = _editable_column(table, operation.column)
        operation.column = column.name
        numeric = column.type.name in {"INT", "DECIMAL"}
        date = column.type.name in {"DATE", "DATETIME"}
        if not numeric and not date:
            raise ValueError(
                f"Column {table.name}.{column.name} must be numeric or date-like"
            )
        if date and operation.transformation != "shift_days":
            raise ValueError("Date columns only support shift_days")
        if numeric and operation.transformation == "shift_days":
            raise ValueError("Numeric columns do not support shift_days")
        return operation
    if name == "replace_categorical_values":
        operation = ReplaceCategoricalValuesOperation.model_validate(args)
        column = _editable_column(table, operation.column)
        operation.column = column.name
        if column.type.name not in {"ENUM", "VARCHAR", "TEXT", "BOOLEAN"}:
            raise ValueError(
                f"Column {table.name}.{column.name} is not categorical"
            )
        if column.type.name == "ENUM":
            declared_values = set(column.type.enum_values)
            invalid_keys = set(operation.replacements) - declared_values
            if invalid_keys:
                raise ValueError(
                    f"Enum replacement key is not declared: {sorted(invalid_keys)[0]}"
                )
            invalid_values = set(operation.replacements.values()) - declared_values
            if invalid_values:
                raise ValueError(
                    f"Enum replacement value is not declared: {sorted(invalid_values)[0]}"
                )
        return operation
    raise ValueError(f"Unknown feedback tool {name!r}")


def _feedback_declarations(table: TableSchema) -> list[types.FunctionDeclaration]:
    editable = [
        column.name for column in table.columns if not _is_key_column(table, column.name)
    ]
    numeric_or_date = [
        column.name
        for column in table.columns
        if column.name in editable
        and column.type.name in {"INT", "DECIMAL", "DATE", "DATETIME"}
    ]
    categorical = [
        column.name
        for column in table.columns
        if column.name in editable
        and column.type.name in {"ENUM", "VARCHAR", "TEXT", "BOOLEAN"}
    ]
    declarations: list[types.FunctionDeclaration] = []
    if editable:
        declarations.append(
            types.FunctionDeclaration(
                name="regenerate_columns",
                description="Regenerate selected non-key columns.",
                parameters={
                    "type": "object",
                    "properties": {
                        "columns": {
                            "type": "array",
                            "items": {"type": "string", "enum": editable},
                            "minItems": 1,
                        },
                        "instruction": {"type": "string"},
                    },
                    "required": ["columns"],
                },
            )
        )
    if numeric_or_date:
        declarations.append(
            types.FunctionDeclaration(
                name="transform_values",
                description="Apply a bounded numeric or date transformation.",
                parameters={
                    "type": "object",
                    "properties": {
                        "column": {"type": "string", "enum": numeric_or_date},
                        "transformation": {
                            "type": "string",
                            "enum": ["add", "multiply", "clamp", "shift_days"],
                        },
                        "value": {
                            "type": "number",
                            "minimum": -1_000_000,
                            "maximum": 1_000_000,
                        },
                        "minimum": {"type": "number"},
                        "maximum": {"type": "number"},
                    },
                    "required": ["column", "transformation"],
                },
            )
        )
    if categorical:
        declarations.append(
            types.FunctionDeclaration(
                name="replace_categorical_values",
                description="Replace categorical values without editing keys.",
                parameters={
                    "type": "object",
                    "properties": {
                        "column": {"type": "string", "enum": categorical},
                        "replacements": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["column", "replacements"],
                },
            )
        )
    return declarations


def _editable_column(table: TableSchema, column_name: str) -> Any:
    try:
        column = table.column(column_name)
    except KeyError as error:
        raise ValueError(str(error)) from error
    if _is_key_column(table, column.name):
        raise ValueError(f"Cannot edit key column {table.name}.{column.name}")
    return column


def _is_key_column(table: TableSchema, column_name: str) -> bool:
    key = column_name.casefold()
    try:
        if table.column(column_name).primary_key:
            return True
    except KeyError:
        return False
    primary_columns = table.primary_key.columns if table.primary_key else []
    foreign_columns = [
        column
        for foreign_key in table.foreign_keys
        for column in foreign_key.columns
    ]
    return key in {
        column.casefold() for column in [*primary_columns, *foreign_columns]
    }


def _trace_metadata(schema: DatabaseSchema, model: str) -> dict[str, Any]:
    return {
        "table_count": len(schema.tables),
        "column_count": sum(len(table.columns) for table in schema.tables),
        "model": model,
    }


def _query_prompt(schema: DatabaseSchema, question: str) -> str:
    tables = ", ".join(table.name for table in schema.tables)
    return (
        "Write one PostgreSQL SELECT that answers the question using only these "
        f"tables: {tables}. Do not modify data. Prefer explicit column lists. "
        f"Question: {question}\nSchema: {_schema_payload(schema)}"
    )


def _parse_query(response: Any) -> NaturalLanguageQuery:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, NaturalLanguageQuery):
        return parsed
    if parsed is not None:
        return NaturalLanguageQuery.model_validate(parsed)
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini returned no structured query")
    return NaturalLanguageQuery.model_validate(json.loads(text))


def _plan_prompt(schema: DatabaseSchema) -> str:
    return (
        "Create a realistic synthetic-data generation plan for this normalized schema. "
        "Respect SQL nullability, types, ranges, categories, and relationships. "
        f"Schema: {_schema_payload(schema)}"
    )


def _schema_payload(schema: DatabaseSchema) -> str:
    return schema.model_dump_json(exclude_none=True)
