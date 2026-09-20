"""Contract tests for Gemini planning and safe feedback requests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import pytest
from google.genai import _transformers as genai_transformers
from pydantic import ValidationError

from data_assistant.gemini_planning import (
    DatabaseGenerationPlan,
    GeminiPlanningService,
    LangfuseTracer,
    NaturalLanguageQuery,
    RegenerateColumnsOperation,
    ReplaceCategoricalValuesOperation,
    TransformValuesOperation,
    create_google_client,
)
from data_assistant.generation import SyntheticDataGenerator
from data_assistant.schema import parse_ddl
from data_assistant.settings import Settings


SCHEMA = parse_ddl(
    """
    CREATE TABLE Customers (
        id INT PRIMARY KEY,
        category ENUM('retail', 'business') NOT NULL,
        score DECIMAL(5, 2),
        joined_on DATE,
        nickname VARCHAR(100)
    );
    """
)


@dataclass
class FakeResponse:
    parsed: Any = None
    text: str | None = None
    function_calls: list[Any] | None = None


@dataclass
class FakeFunctionCall:
    name: str
    args: dict[str, Any]


class FakeClient:
    def __init__(
        self,
        *,
        response: FakeResponse | None = None,
        stream: list[FakeResponse] | None = None,
    ) -> None:
        self.response = response or FakeResponse()
        self.stream = stream or []
        self.calls: list[tuple[str, str, Any]] = []

    def generate_content(self, *, model: str, contents: str, config: Any) -> FakeResponse:
        self.calls.append(("generate", model, config, contents))
        return self.response

    def generate_content_stream(
        self, *, model: str, contents: str, config: Any
    ) -> Iterator[FakeResponse]:
        self.calls.append(("stream", model, config, contents))
        return iter(self.stream)


class RecordingTracer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    @contextmanager
    def trace(self, name: str, metadata: dict[str, Any]) -> Iterator[None]:
        self.events.append((name, metadata))
        yield


def _settings() -> Settings:
    return Settings(
        google_cloud_project="project-123",
        database_url="postgresql://redacted",
        vertex_ai_location="europe-west1",
        gemini_model="gemini-2.5-flash",
    )


def _valid_plan() -> dict[str, Any]:
    return {
        "database_description": "Customer records",
        "tables": [
            {
                "table_name": "Customers",
                "row_count": 25,
                "columns": [
                    {
                        "column_name": "category",
                        "generator": "categorical",
                        "semantic_type": "customer segment",
                        "locale": "en_US",
                        "categories": ["retail", "business"],
                        "distribution": "weighted",
                        "nullable_probability": 0,
                    },
                    {
                        "column_name": "score",
                        "generator": "decimal",
                        "minimum": 0,
                        "maximum": 100,
                        "distribution": "normal",
                        "nullable_probability": 0.1,
                    },
                    {
                        "column_name": "joined_on",
                        "generator": "date",
                        "nullable_probability": 0.1,
                    },
                    {
                        "column_name": "nickname",
                        "generator": "person_name",
                        "locale": "en_US",
                        "nullable_probability": 0.1,
                    },
                ],
            }
        ],
    }


def test_create_google_client_configures_vertex_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class StubClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("google.genai.Client", StubClient)

    create_google_client(_settings())

    assert captured == {
        "vertexai": True,
        "project": "project-123",
        "location": "europe-west1",
    }


def test_request_plan_passes_structured_schema_and_validates_against_database() -> None:
    fake = FakeClient(response=FakeResponse(parsed=_valid_plan()))
    tracer = RecordingTracer()
    service = GeminiPlanningService(_settings(), client=fake, tracer=tracer)

    plan = service.request_plan(SCHEMA, temperature=0.25)

    assert isinstance(plan, DatabaseGenerationPlan)
    assert plan.tables[0].table_name == "Customers"
    _, model, config, _contents = fake.calls[0]
    assert model == "gemini-2.5-flash"
    assert config.response_mime_type == "application/json"
    assert config.response_schema is DatabaseGenerationPlan
    assert config.temperature == 0.25
    assert tracer.events == [
        (
            "gemini.plan",
            {"table_count": 1, "column_count": 5, "model": "gemini-2.5-flash"},
        )
    ]
    assert "CREATE TABLE" not in repr(tracer.events)


def test_generation_plan_schema_converts_with_installed_google_genai_sdk() -> None:
    converted = genai_transformers.t_schema(None, DatabaseGenerationPlan)

    row_count = (
        converted.properties["tables"]
        .items.properties["row_count"]
    )
    assert row_count.minimum == 1
    assert DatabaseGenerationPlan.model_validate(_valid_plan()).tables[0].row_count == 25
    invalid = _valid_plan()
    invalid["tables"][0]["row_count"] = 0
    with pytest.raises(ValidationError):
        DatabaseGenerationPlan.model_validate(invalid)


def test_request_plan_keeps_invented_generator_names_out_of_the_plan() -> None:
    invented = _valid_plan()
    invented["tables"][0]["columns"][0]["generator"] = "sequential_integer"
    service = GeminiPlanningService(
        _settings(), client=FakeClient(response=FakeResponse(parsed=invented))
    )

    plan = service.request_plan(SCHEMA)

    category = plan.tables[0].columns[0]
    assert category.generator == "auto"
    assert category.semantic_type is None
    assert category.distribution == "uniform"
    assert len(SyntheticDataGenerator().generate(SCHEMA, plan, seed=1)["Customers"]) == 25


def test_request_plan_rejects_unknown_columns() -> None:
    invalid = _valid_plan()
    invalid["tables"][0]["columns"][0]["column_name"] = "missing"
    service = GeminiPlanningService(
        _settings(), client=FakeClient(response=FakeResponse(parsed=invalid))
    )

    with pytest.raises(ValueError, match="Unknown column"):
        service.request_plan(SCHEMA)


def test_request_plan_requires_every_table_and_non_key_column_exactly_once() -> None:
    missing_column = _valid_plan()
    missing_column["tables"][0]["columns"].pop()
    service = GeminiPlanningService(
        _settings(), client=FakeClient(response=FakeResponse(parsed=missing_column))
    )
    with pytest.raises(ValueError, match="missing non-key columns.*nickname"):
        service.request_plan(SCHEMA)

    second_table_schema = parse_ddl(
        """
        CREATE TABLE Customers (
            id INT PRIMARY KEY,
            category ENUM('retail', 'business') NOT NULL,
            score DECIMAL(5, 2),
            joined_on DATE,
            nickname VARCHAR(100)
        );
        CREATE TABLE AuditLog (
            id INT PRIMARY KEY,
            message TEXT NOT NULL
        );
        """
    )
    service = GeminiPlanningService(
        _settings(), client=FakeClient(response=FakeResponse(parsed=_valid_plan()))
    )
    with pytest.raises(ValueError, match="missing table plans.*AuditLog"):
        service.request_plan(second_table_schema)

    duplicate_column = _valid_plan()
    duplicate_column["tables"][0]["columns"].append(
        duplicate_column["tables"][0]["columns"][0].copy()
    )
    service = GeminiPlanningService(
        _settings(), client=FakeClient(response=FakeResponse(parsed=duplicate_column))
    )
    with pytest.raises(ValueError, match="Duplicate column plan"):
        service.request_plan(SCHEMA)


def test_request_plan_allows_generator_owned_key_columns_to_be_omitted() -> None:
    service = GeminiPlanningService(
        _settings(), client=FakeClient(response=FakeResponse(parsed=_valid_plan()))
    )

    plan = service.request_plan(SCHEMA)

    assert {column.column_name for column in plan.tables[0].columns} == {
        "category",
        "score",
        "joined_on",
        "nickname",
    }


def test_stream_explanation_yields_only_nonempty_text_chunks() -> None:
    fake = FakeClient(
        stream=[
            FakeResponse(text="Planning"),
            FakeResponse(text=""),
            FakeResponse(text=None),
            FakeResponse(text=" complete"),
        ]
    )
    tracer = RecordingTracer()
    service = GeminiPlanningService(_settings(), client=fake, tracer=tracer)

    chunks = list(service.stream_explanation(SCHEMA, "Explain the plan"))

    assert chunks == ["Planning", " complete"]
    assert fake.calls[0][0] == "stream"
    assert tracer.events[0][0] == "gemini.stream"


def test_feedback_sends_safe_table_specific_tool_declarations_and_parses_calls() -> None:
    fake = FakeClient(
        response=FakeResponse(
            function_calls=[
                FakeFunctionCall(
                    "regenerate_columns",
                    {"columns": ["NICKNAME"], "instruction": "Use short aliases"},
                ),
                FakeFunctionCall(
                    "transform_values",
                    {
                        "column": "SCORE",
                        "transformation": "clamp",
                        "minimum": 1,
                        "maximum": 5,
                    },
                ),
                FakeFunctionCall(
                    "replace_categorical_values",
                    {"column": "CATEGORY", "replacements": {"retail": "business"}},
                ),
            ]
        )
    )
    service = GeminiPlanningService(_settings(), client=fake)

    operations = service.request_feedback_operations(
        SCHEMA, "Customers", "Improve these generated values"
    )

    assert isinstance(operations[0], RegenerateColumnsOperation)
    assert isinstance(operations[1], TransformValuesOperation)
    assert isinstance(operations[2], ReplaceCategoricalValuesOperation)
    assert all(operation.table_name == "Customers" for operation in operations)
    assert [operation.operation_type for operation in operations] == [
        "regenerate_columns",
        "transform_values",
        "replace_categorical_values",
    ]
    assert operations[0].columns == ["nickname"]
    assert operations[1].column == "score"
    assert operations[2].column == "category"
    config = fake.calls[0][2]
    declarations = config.tools[0].function_declarations
    assert {declaration.name for declaration in declarations} == {
        "regenerate_columns",
        "transform_values",
        "replace_categorical_values",
    }
    regenerate = next(item for item in declarations if item.name == "regenerate_columns")
    parameters = regenerate.parameters.model_dump(exclude_none=True)
    assert parameters["properties"]["columns"]["items"]["enum"] == [
        "category",
        "score",
        "joined_on",
        "nickname",
    ]


def test_feedback_omits_tools_with_no_eligible_columns() -> None:
    numeric_schema = parse_ddl(
        "CREATE TABLE Metrics (id INT PRIMARY KEY, amount DECIMAL(8, 2));"
    )
    fake = FakeClient()
    GeminiPlanningService(_settings(), client=fake).request_feedback_operations(
        numeric_schema, "Metrics", "Adjust amounts"
    )
    declarations = fake.calls[0][2].tools[0].function_declarations
    assert {declaration.name for declaration in declarations} == {
        "regenerate_columns",
        "transform_values",
    }

    key_only_schema = parse_ddl("CREATE TABLE KeysOnly (id INT PRIMARY KEY);")
    fake = FakeClient()
    GeminiPlanningService(_settings(), client=fake).request_feedback_operations(
        key_only_schema, "KeysOnly", "Change values"
    )
    assert not fake.calls[0][2].tools


@pytest.mark.parametrize(
    "call, message",
    [
        (FakeFunctionCall("drop_table", {}), "Unknown feedback tool"),
        (
            FakeFunctionCall("regenerate_columns", {"columns": ["id"]}),
            "key column",
        ),
        (
            FakeFunctionCall(
                "transform_values",
                {"column": "nickname", "transformation": "add", "value": 1},
            ),
            "numeric or date",
        ),
        (
            FakeFunctionCall(
                "transform_values",
                {"column": "joined_on", "transformation": "shift_days", "value": 4000},
            ),
            "less than or equal to 3650",
        ),
        (
            FakeFunctionCall(
                "replace_categorical_values",
                {"column": "missing", "replacements": {"a": "b"}},
            ),
            "Unknown column",
        ),
        (
            FakeFunctionCall(
                "replace_categorical_values",
                {"column": "category", "replacements": {"unknown": "retail"}},
            ),
            "replacement key.*unknown",
        ),
        (
            FakeFunctionCall(
                "replace_categorical_values",
                {"column": "category", "replacements": {"retail": "unknown"}},
            ),
            "replacement value.*unknown",
        ),
    ],
)
def test_feedback_rejects_unsafe_or_invalid_calls(
    call: FakeFunctionCall, message: str
) -> None:
    service = GeminiPlanningService(
        _settings(),
        client=FakeClient(response=FakeResponse(function_calls=[call])),
    )

    with pytest.raises(ValueError, match=message):
        service.request_feedback_operations(SCHEMA, "Customers", "Change values")


def test_gemini_errors_are_not_swallowed() -> None:
    class FailingClient(FakeClient):
        def generate_content(self, *, model: str, contents: str, config: Any) -> FakeResponse:
            raise RuntimeError("Gemini unavailable")

    service = GeminiPlanningService(_settings(), client=FailingClient())

    with pytest.raises(RuntimeError, match="Gemini unavailable"):
        service.request_plan(SCHEMA)


def test_langfuse_tracer_ends_span_and_flushes() -> None:
    class Span:
        ended = False

        def end(self) -> None:
            self.ended = True

    class Trace:
        def __init__(self, span: Span) -> None:
            self._span = span

        def span(self, *, name: str) -> Span:
            assert name == "gemini.plan"
            return self._span

    class LangfuseClient:
        def __init__(self) -> None:
            self.span = Span()
            self.flushed = False

        def trace(self, *, name: str, metadata: dict[str, Any]) -> Trace:
            assert name == "gemini.plan"
            assert metadata == {"table_count": 1}
            return Trace(self.span)

        def flush(self) -> None:
            self.flushed = True

    client = LangfuseClient()
    tracer = LangfuseTracer(client)

    with tracer.trace("gemini.plan", {"table_count": 1}):
        pass

    assert client.span.ended
    assert client.flushed


def test_request_query_returns_select_and_rejects_writes() -> None:
    fake = FakeClient(
        response=FakeResponse(
            parsed=NaturalLanguageQuery(
                sql="SELECT category FROM Customers",
                explanation="List customer categories",
            )
        )
    )
    tracer = RecordingTracer()
    service = GeminiPlanningService(_settings(), client=fake, tracer=tracer)

    plan = service.request_query(SCHEMA, "What categories exist?", temperature=0.0)

    assert plan.sql == "SELECT category FROM Customers"
    _, model, config, contents = fake.calls[0]
    assert model == "gemini-2.5-flash"
    assert config.response_schema is NaturalLanguageQuery
    assert config.temperature == 0.0
    assert "What categories exist?" in contents
    assert tracer.events[0][0] == "gemini.query"

    fake.response = FakeResponse(
        parsed={"sql": "DELETE FROM Customers", "explanation": "wipe"}
    )
    with pytest.raises(ValueError):
        service.request_query(SCHEMA, "delete everything")
    with pytest.raises(ValueError, match="empty"):
        service.request_query(SCHEMA, "   ")


def test_query_schema_converts_with_installed_google_genai_sdk() -> None:
    converted = genai_transformers.t_schema(None, NaturalLanguageQuery)

    assert converted.properties["sql"].min_length == 1
    assert converted.properties["explanation"].min_length == 1
