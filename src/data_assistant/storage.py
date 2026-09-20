"""PostgreSQL persistence for generated datasets.

Each dataset is written into its own server-generated schema (``dataset_<uuid>``)
and recorded in one app-owned metadata table. Every identifier is composed with
``psycopg.sql.Identifier`` and every value is bound as a parameter, so uploaded
table, column, and constraint names are never interpolated into SQL. Saves and
updates run as a single transaction, and a dataset is validated against its
schema before anything is written.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, Self
from uuid import UUID, uuid4

import psycopg
from psycopg import sql

from data_assistant.querying import DEFAULT_QUERY_LIMIT, QueryError, compile_readonly_select
from data_assistant.schema import ColumnSchema, DatabaseSchema, ForeignKeySchema, TableSchema
from data_assistant.validation import (
    CheckExpressionError,
    Dataset,
    Violation,
    parse_check_expression,
    validate_dataset,
)


METADATA_TABLE = "data_assistant_datasets"
SCHEMA_PREFIX = "dataset_"
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 1_000


class StorageError(RuntimeError):
    """Raised when a storage operation cannot be completed."""


class DatasetNotFoundError(StorageError):
    """Raised when no dataset is recorded under the requested id."""


class DatasetValidationError(StorageError):
    """Raised when a dataset does not satisfy its schema; carries every violation."""

    def __init__(self, violations: Sequence[Violation]) -> None:
        self.violations = list(violations)
        head = "; ".join(violation.message for violation in self.violations[:3])
        super().__init__(f"{len(self.violations)} dataset violations: {head}")


class _Cursor(Protocol):
    def execute(self, query: Any, params: Any = None) -> Any: ...

    def executemany(self, query: Any, params_seq: Any) -> Any: ...

    def fetchall(self) -> list[tuple[Any, ...]]: ...

    def fetchone(self) -> tuple[Any, ...] | None: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *args: Any) -> None: ...


class _Connection(Protocol):
    def cursor(self) -> _Cursor: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


ConnectFactory = Callable[[], _Connection]


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Metadata of one stored dataset. `created_at` is unknown until it is read back."""

    dataset_id: UUID
    name: str
    schema_name: str
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """One stored dataset with the inputs it was generated from."""

    dataset_id: UUID
    name: str
    schema_name: str
    ddl: str
    instructions: str | None
    schema: DatabaseSchema
    created_at: datetime | None = None

    @property
    def summary(self) -> DatasetSummary:
        return DatasetSummary(
            dataset_id=self.dataset_id,
            name=self.name,
            schema_name=self.schema_name,
            created_at=self.created_at,
        )


@dataclass(frozen=True, slots=True)
class TablePage:
    """One page of rows of a stored table."""

    table: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class QueryPage:
    """Result of one compiled read-only SELECT against a stored dataset."""

    sql: str
    explanation: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


class DatasetRepository:
    """Stores and reads generated datasets through an injected connection factory."""

    def __init__(
        self,
        *,
        connect: ConnectFactory,
        new_dataset_id: Callable[[], UUID] = uuid4,
    ) -> None:
        self._connect = connect
        self._new_dataset_id = new_dataset_id

    def initialize(self) -> None:
        """Create the app-owned metadata table if it does not exist.

        # Errors
        Raises `StorageError` when the database rejects the statement.
        """
        with _clean_errors("could not initialize dataset storage"):
            with self._transaction() as connection, connection.cursor() as cursor:
                cursor.execute(_CREATE_METADATA_TABLE)

    def save_dataset(
        self,
        *,
        name: str,
        ddl: str,
        schema: DatabaseSchema,
        data: Dataset,
        instructions: str | None = None,
    ) -> DatasetSummary:
        """Write a validated dataset into a fresh isolated schema in one transaction.

        # Errors
        Raises `DatasetValidationError` when the data does not satisfy the schema,
        and `StorageError` when the database rejects any statement; a failure
        leaves neither a metadata row nor a dataset schema behind.
        """
        _assert_valid(schema, data)
        dataset_id = self._new_dataset_id()
        schema_name = _schema_name(dataset_id)
        with _clean_errors(f"could not save dataset {name!r}"):
            with self._transaction() as connection, connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name))
                )
                for table in schema.tables:
                    cursor.execute(_create_table(schema_name, table))
                for table in schema.tables:
                    rows = _rows_for(data, table.name)
                    insert = _insert(schema_name, table, rows)
                    if insert is not None:
                        cursor.executemany(*insert)
                for table in schema.tables:
                    for foreign_key in table.foreign_keys:
                        cursor.execute(_add_foreign_key(schema_name, table, foreign_key))
                cursor.execute(
                    _INSERT_METADATA,
                    (
                        dataset_id,
                        name,
                        schema_name,
                        ddl,
                        instructions,
                        schema.model_dump_json(),
                    ),
                )
        return DatasetSummary(dataset_id=dataset_id, name=name, schema_name=schema_name)

    def list_datasets(self) -> list[DatasetSummary]:
        """Return every stored dataset, most recently created first.

        # Errors
        Raises `StorageError` when the metadata table cannot be read.
        """
        with _clean_errors("could not list datasets"):
            with self._transaction() as connection, connection.cursor() as cursor:
                cursor.execute(_SELECT_SUMMARIES)
                rows = cursor.fetchall()
        return [
            DatasetSummary(
                dataset_id=row[0], name=row[1], schema_name=row[2], created_at=row[3]
            )
            for row in rows
        ]

    def load_dataset(self, dataset_id: UUID | str) -> DatasetRecord:
        """Return the metadata, original DDL, instructions, and schema of one dataset.

        # Errors
        Raises `DatasetNotFoundError` when the id is unknown and `StorageError`
        when the metadata table cannot be read.
        """
        identifier = _dataset_id(dataset_id)
        with _clean_errors(f"could not load dataset {identifier}"):
            with self._transaction() as connection, connection.cursor() as cursor:
                return _load_record(cursor, identifier)

    def list_tables(self, dataset_id: UUID | str) -> list[str]:
        """Return the table names of a stored dataset in schema order."""
        return [table.name for table in self.load_dataset(dataset_id).schema.tables]

    def read_table(
        self,
        dataset_id: UUID | str,
        table_name: str,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
    ) -> TablePage:
        """Return one page of rows of a stored table in primary-key order.

        # Errors
        Raises `StorageError` for an unknown dataset or table and for a page
        size outside 1..`MAX_PAGE_SIZE` or a negative offset.
        """
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise StorageError(f"page size must be between 1 and {MAX_PAGE_SIZE}")
        if offset < 0:
            raise StorageError("page offset must not be negative")
        identifier = _dataset_id(dataset_id)
        with _clean_errors(f"could not read table {table_name!r}"):
            with self._transaction() as connection, connection.cursor() as cursor:
                record = _load_record(cursor, identifier)
                table = _table(record, table_name)
                cursor.execute(_count(record.schema_name, table))
                counted = cursor.fetchone()
                cursor.execute(
                    _select_rows(record.schema_name, table, paged=True), (limit, offset)
                )
                rows = cursor.fetchall()
        names = tuple(column.name for column in table.columns)
        return TablePage(
            table=table.name,
            columns=names,
            rows=tuple(dict(zip(names, row)) for row in rows),
            total=int(counted[0]) if counted else 0,
            limit=limit,
            offset=offset,
        )

    def execute_select(
        self,
        dataset_id: UUID | str,
        sql_text: str,
        *,
        explanation: str = "",
        max_rows: int = DEFAULT_QUERY_LIMIT,
    ) -> QueryPage:
        """Compile `sql_text` to a read-only SELECT and run it in the dataset schema.

        # Errors
        Raises `StorageError` when the dataset is unknown, the SQL is not a
        single SELECT of known tables, or the database rejects the statement.
        """
        identifier = _dataset_id(dataset_id)
        with _clean_errors("could not execute query"):
            with self._transaction() as connection, connection.cursor() as cursor:
                record = _load_record(cursor, identifier)
                try:
                    compiled = compile_readonly_select(
                        sql_text,
                        record.schema,
                        record.schema_name,
                        max_rows=max_rows,
                    )
                except QueryError as error:
                    raise StorageError(str(error)) from error
                cursor.execute(
                    sql.SQL("SET LOCAL search_path TO {}, pg_temp").format(
                        sql.Identifier(record.schema_name)
                    )
                )
                cursor.execute(sql.SQL("SET LOCAL transaction_read_only = on"))
                cursor.execute(sql.SQL("SET LOCAL statement_timeout = '15s'"))
                cursor.execute(sql.SQL(compiled.sql))
                fetched = cursor.fetchall()
        names = compiled.columns
        width = len(fetched[0]) if fetched else len(names)
        if len(names) < width:
            names = names + tuple(f"column_{index + 1}" for index in range(len(names), width))
        rows = tuple(
            {names[index]: value for index, value in enumerate(row)} for row in fetched
        )
        return QueryPage(
            sql=compiled.sql,
            explanation=explanation,
            columns=names,
            rows=rows,
        )

    def update_table(
        self,
        dataset_id: UUID | str,
        table_name: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> int:
        """Apply edited rows to one stored table, keyed by their unchanged primary keys.

        ## Returns
        The number of rows whose values differed and were updated.

        # Errors
        Raises `DatasetValidationError` when the dataset reconstructed from the
        edit does not satisfy the schema, and `StorageError` when the table has
        no primary key, when the primary-key values change, or when the database
        rejects a statement.
        """
        identifier = _dataset_id(dataset_id)
        with _clean_errors(f"could not update table {table_name!r}"):
            with self._transaction() as connection, connection.cursor() as cursor:
                record = _load_record(cursor, identifier)
                table = _table(record, table_name)
                if table.primary_key is None:
                    raise StorageError(
                        f"table {table.name!r} has no primary key and cannot be "
                        "updated row by row"
                    )
                keys = [table.column(name).name for name in table.primary_key.columns]
                edited = _normalized_rows(table, rows)
                stored = {
                    other.name: _read_all(cursor, record.schema_name, other)
                    for other in record.schema.tables
                }
                _assert_valid(record.schema, {**stored, table.name: edited})
                _assert_unchanged_keys(table, keys, stored[table.name], edited)
                return _apply_updates(
                    cursor, record.schema_name, table, keys, stored[table.name], edited
                )

    @contextmanager
    def _transaction(self) -> Iterator[_Connection]:
        connection = self._connect()
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()


def default_connect(database_url: str) -> ConnectFactory:
    """Return a psycopg connection factory for the given URL."""

    def connect() -> psycopg.Connection[Any]:
        return psycopg.connect(database_url)

    return connect


_SCHEMA_NAME_PATTERN = re.compile(rf"{SCHEMA_PREFIX}[0-9a-f]{{32}}")
_CREATE_METADATA_TABLE = sql.SQL(
    "CREATE TABLE IF NOT EXISTS {} ("
    '"dataset_id" uuid PRIMARY KEY, '
    '"name" text NOT NULL, '
    '"schema_name" text NOT NULL UNIQUE, '
    '"ddl" text NOT NULL, '
    '"instructions" text, '
    '"schema_json" jsonb NOT NULL, '
    '"created_at" timestamptz NOT NULL DEFAULT now())'
).format(sql.Identifier(METADATA_TABLE))
_INSERT_METADATA = sql.SQL(
    "INSERT INTO {} ({}) VALUES (%s, %s, %s, %s, %s, %s::jsonb)"
).format(
    sql.Identifier(METADATA_TABLE),
    sql.SQL(", ").join(
        sql.Identifier(name)
        for name in ("dataset_id", "name", "schema_name", "ddl", "instructions", "schema_json")
    ),
)
_SELECT_SUMMARIES = sql.SQL("SELECT {} FROM {} ORDER BY {} DESC, {}").format(
    sql.SQL(", ").join(
        sql.Identifier(name) for name in ("dataset_id", "name", "schema_name", "created_at")
    ),
    sql.Identifier(METADATA_TABLE),
    sql.Identifier("created_at"),
    sql.Identifier("dataset_id"),
)
_SELECT_RECORD = sql.SQL("SELECT {}, {}::text, {} FROM {} WHERE {} = %s").format(
    sql.SQL(", ").join(
        sql.Identifier(name)
        for name in ("dataset_id", "name", "schema_name", "ddl", "instructions")
    ),
    sql.Identifier("schema_json"),
    sql.Identifier("created_at"),
    sql.Identifier(METADATA_TABLE),
    sql.Identifier("dataset_id"),
)

_SIMPLE_TYPES = {
    "INT": sql.SQL("integer"),
    "TEXT": sql.SQL("text"),
    "DATE": sql.SQL("date"),
    "DATETIME": sql.SQL("timestamp"),
    "BOOLEAN": sql.SQL("boolean"),
    "ENUM": sql.SQL("text"),
}
_FOREIGN_KEY_ACTIONS = {
    action: sql.SQL(action)
    for action in ("RESTRICT", "CASCADE", "SET NULL", "NO ACTION", "SET DEFAULT")
}
# Mirrors the CHECK grammar of `validation`, which parses but does not re-emit SQL.
_CHECK_TOKEN = re.compile(
    r"(?P<number>\d+(?:\.\d+)?)"
    r"|(?P<string>'(?:''|[^'])*')"
    r"|(?P<quoted>`[^`]+`)"
    r"|(?P<identifier>[A-Za-z_][A-Za-z0-9_$]*)"
    r"|(?P<operator><=|>=|<>|!=|=|<|>)"
    r"|(?P<punct>[(),])"
    r"|(?P<minus>-)"
)
_CHECK_KEYWORDS = {"AND", "OR", "NOT", "IS", "IN", "BETWEEN", "NULL", "TRUE", "FALSE"}


def _schema_name(dataset_id: UUID) -> str:
    name = f"{SCHEMA_PREFIX}{dataset_id.hex}"
    if not _SCHEMA_NAME_PATTERN.fullmatch(name):
        raise StorageError("generated dataset schema name is not a safe identifier")
    return name


def _dataset_id(value: UUID | str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(value)
    except (AttributeError, TypeError, ValueError):
        raise StorageError(f"{value!r} is not a dataset id") from None


@contextmanager
def _clean_errors(message: str) -> Iterator[None]:
    """Replace driver failures with an app error that carries no connection detail."""
    try:
        yield
    except (StorageError, DatasetValidationError):
        raise
    except Exception:
        raise StorageError(f"{message}: the database rejected the operation") from None


def _assert_valid(schema: DatabaseSchema, data: Dataset) -> None:
    """Validate a dataset against its schema before it is written.

    # Errors
    Raises `DatasetValidationError` when any constraint is violated.
    """
    # An unsupported CHECK is a limit of the validator, not a defect of the
    # data, and such a CHECK is left out of the created table as well.
    violations = [
        violation
        for violation in validate_dataset(schema, data)
        if violation.kind != "unsupported_check"
    ]
    if violations:
        raise DatasetValidationError(violations)


def _load_record(cursor: _Cursor, dataset_id: UUID) -> DatasetRecord:
    cursor.execute(_SELECT_RECORD, (dataset_id,))
    row = cursor.fetchone()
    if row is None:
        raise DatasetNotFoundError(f"no dataset is stored under {dataset_id}")
    return DatasetRecord(
        dataset_id=row[0],
        name=row[1],
        schema_name=row[2],
        ddl=row[3],
        instructions=row[4],
        schema=DatabaseSchema.model_validate_json(row[5]),
        created_at=row[6],
    )


def _table(record: DatasetRecord, table_name: str) -> TableSchema:
    try:
        return record.schema.table(table_name)
    except KeyError:
        raise StorageError(
            f"dataset {record.dataset_id} has no table {table_name!r}"
        ) from None


def _create_table(schema_name: str, table: TableSchema) -> sql.Composed:
    """Build the CREATE TABLE of one table, without its foreign keys."""
    items: list[sql.Composable] = [_column_definition(column) for column in table.columns]
    if table.primary_key is not None:
        items.append(
            sql.SQL("PRIMARY KEY ({})").format(_identifiers(table.primary_key.columns))
        )
    items.extend(
        sql.SQL("UNIQUE ({})").format(_identifiers(constraint.columns))
        for constraint in table.unique_constraints
    )
    items.extend(
        sql.SQL("CHECK ({} IN ({}))").format(
            sql.Identifier(column.name),
            sql.SQL(", ").join(sql.Literal(value) for value in column.type.enum_values),
        )
        for column in table.columns
        if column.type.name == "ENUM"
    )
    expressions = [
        *(expression for column in table.columns for expression in column.checks),
        *table.checks,
    ]
    items.extend(
        sql.SQL("CHECK ({})").format(rendered)
        for expression in expressions
        if (rendered := _check_sql(table, expression)) is not None
    )
    return sql.SQL("CREATE TABLE {}.{} ({})").format(
        sql.Identifier(schema_name),
        sql.Identifier(table.name),
        sql.SQL(", ").join(items),
    )


def _column_definition(column: ColumnSchema) -> sql.Composed:
    parts: list[sql.Composable] = [sql.Identifier(column.name), _column_type(column)]
    if not column.nullable:
        parts.append(sql.SQL("NOT NULL"))
    default = _default_sql(column)
    if default is not None:
        parts.append(sql.SQL("DEFAULT {}").format(default))
    return sql.SQL(" ").join(parts)


def _column_type(column: ColumnSchema) -> sql.Composable:
    """Translate one normalized MySQL type into its PostgreSQL equivalent.

    ENUM becomes `text`; its domain is enforced by a CHECK constraint instead.
    """
    if column.type.name == "VARCHAR":
        return sql.SQL("varchar({})").format(sql.Literal(column.type.length))
    if column.type.name == "DECIMAL":
        return sql.SQL("numeric({}, {})").format(
            sql.Literal(column.type.precision), sql.Literal(column.type.scale or 0)
        )
    simple = _SIMPLE_TYPES.get(column.type.name)
    if simple is None:
        raise StorageError(f"SQL type {column.type.name!r} has no PostgreSQL translation")
    return simple


def _default_sql(column: ColumnSchema) -> sql.Composable | None:
    """Return the column default as SQL, or None when it does not port to PostgreSQL."""
    raw = column.default
    if raw is None:
        return None
    if raw.upper() == "CURRENT_TIMESTAMP":
        return (
            sql.SQL("CURRENT_TIMESTAMP") if column.type.name in {"DATE", "DATETIME"} else None
        )
    quoted = len(raw) >= 2 and raw[0] in {"'", '"'} and raw[-1] == raw[0]
    value = _default_value(column, raw[1:-1] if quoted else raw, quoted)
    return None if value is None else sql.Literal(value)


def _default_value(column: ColumnSchema, text: str, quoted: bool) -> Any:
    name = column.type.name
    if name in {"VARCHAR", "TEXT"}:
        return text if quoted else None
    if name == "ENUM":
        return text if quoted and text in column.type.enum_values else None
    try:
        if name == "INT":
            return int(text)
        if name == "DECIMAL":
            return Decimal(text).quantize(Decimal(1).scaleb(-(column.type.scale or 0)))
        if name == "BOOLEAN":
            return {"TRUE": True, "FALSE": False, "1": True, "0": False}.get(text.upper())
        if name == "DATE":
            return date.fromisoformat(text)
        if name == "DATETIME":
            return datetime.fromisoformat(text)
    except (InvalidOperation, ValueError):
        return None
    return None


def _check_sql(table: TableSchema, expression: str) -> sql.Composed | None:
    """Re-emit a CHECK expression with quoted identifiers and bound literals.

    ## Returns
    The translated expression, or None when it falls outside the safely
    supported grammar or names something that is not a column of `table`.
    """
    try:
        parse_check_expression(expression)
    except CheckExpressionError:
        return None

    columns = {column.name.casefold(): column.name for column in table.columns}
    tokens: list[tuple[str, sql.Composable]] = []
    position = 0
    while position < len(expression):
        if expression[position].isspace():
            position += 1
            continue
        match = _CHECK_TOKEN.match(expression, position)
        if match is None:
            return None
        kind, text = match.lastgroup or "", match.group()
        position = match.end()
        if kind == "string":
            tokens.append(("literal", sql.Literal(text[1:-1].replace("''", "'"))))
            continue
        if kind in {"identifier", "quoted"}:
            if kind == "identifier" and text.upper() in _CHECK_KEYWORDS:
                tokens.append(("keyword", sql.SQL(text.upper())))
                continue
            resolved = columns.get((text[1:-1] if kind == "quoted" else text).casefold())
            if resolved is None:
                return None
            tokens.append(("identifier", sql.Identifier(resolved)))
            continue
        tokens.append((text, sql.SQL(text)))
    return _join_tokens(tokens) if tokens else None


def _join_tokens(tokens: Sequence[tuple[str, sql.Composable]]) -> sql.Composed:
    parts: list[sql.Composable] = []
    previous = ""
    for kind, token in tokens:
        if parts and previous != "(" and kind not in {")", ","}:
            parts.append(sql.SQL(" "))
        parts.append(token)
        previous = kind
    return sql.Composed(parts)


def _insert(
    schema_name: str, table: TableSchema, rows: Sequence[Mapping[str, Any]]
) -> tuple[sql.Composed, list[tuple[Any, ...]]] | None:
    """Build the bulk INSERT of one table, or None when it has no rows to write.

    Columns absent from every row are left out so their PostgreSQL default applies.
    """
    if not rows:
        return None
    names = [
        column.name
        for column in table.columns
        if any(_has_value(row, column.name) for row in rows)
    ]
    if not names:
        return None
    statement = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({})").format(
        sql.Identifier(schema_name),
        sql.Identifier(table.name),
        _identifiers(names),
        sql.SQL(", ").join(sql.Placeholder() for _ in names),
    )
    return statement, [tuple(_value(row, name) for name in names) for row in rows]


def _add_foreign_key(
    schema_name: str, table: TableSchema, foreign_key: ForeignKeySchema
) -> sql.Composed:
    parts: list[sql.Composable] = [
        sql.SQL("ALTER TABLE {}.{} ADD FOREIGN KEY ({}) REFERENCES {}.{} ({})").format(
            sql.Identifier(schema_name),
            sql.Identifier(table.name),
            _identifiers(foreign_key.columns),
            sql.Identifier(schema_name),
            sql.Identifier(foreign_key.referenced_table),
            _identifiers(foreign_key.referenced_columns),
        )
    ]
    for event, action in (("DELETE", foreign_key.on_delete), ("UPDATE", foreign_key.on_update)):
        clause = _FOREIGN_KEY_ACTIONS.get(action or "")
        if clause is not None:
            parts.append(sql.SQL("ON {} {}").format(sql.SQL(event), clause))
    return sql.SQL(" ").join(parts)


def _count(schema_name: str, table: TableSchema) -> sql.Composed:
    return sql.SQL("SELECT count(*) FROM {}.{}").format(
        sql.Identifier(schema_name), sql.Identifier(table.name)
    )


def _select_rows(schema_name: str, table: TableSchema, *, paged: bool) -> sql.Composed:
    order = (
        table.primary_key.columns
        if table.primary_key is not None
        else [column.name for column in table.columns]
    )
    statement = sql.SQL("SELECT {} FROM {}.{} ORDER BY {}").format(
        _identifiers([column.name for column in table.columns]),
        sql.Identifier(schema_name),
        sql.Identifier(table.name),
        _identifiers(order),
    )
    if not paged:
        return statement
    return sql.SQL("{} LIMIT %s OFFSET %s").format(statement)


def _read_all(
    cursor: _Cursor, schema_name: str, table: TableSchema
) -> list[dict[str, Any]]:
    cursor.execute(_select_rows(schema_name, table, paged=False))
    names = [column.name for column in table.columns]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _normalized_rows(
    table: TableSchema, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Resolve submitted rows to complete rows keyed by canonical column names.

    # Errors
    Raises `DatasetValidationError` when a row is not a mapping, names a column
    the table does not have, or leaves one of its columns out.
    """
    names = {column.name.casefold(): column.name for column in table.columns}
    violations: list[Violation] = []
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            violations.append(
                Violation(
                    kind="row_shape",
                    table=table.name,
                    row_index=index,
                    message=f"{table.name} row {index} is {type(row).__name__}, "
                    "expected a mapping",
                )
            )
            continue
        resolved: dict[str, Any] = {}
        for key, value in row.items():
            name = names.get(key.casefold()) if isinstance(key, str) else None
            if name is None:
                violations.append(
                    Violation(
                        kind="unknown_column",
                        table=table.name,
                        column=str(key),
                        row_index=index,
                        message=f"{table.name} row {index} has column {key!r} "
                        "that is not in the schema",
                    )
                )
                continue
            resolved[name] = value
        violations.extend(
            Violation(
                kind="missing_column",
                table=table.name,
                column=column.name,
                row_index=index,
                message=f"{table.name} row {index} is missing column {column.name!r}; "
                "an edited table must carry every column",
            )
            for column in table.columns
            if column.name not in resolved
        )
        normalized.append({column.name: resolved.get(column.name) for column in table.columns})
    if violations:
        raise DatasetValidationError(violations)
    return normalized


def _assert_unchanged_keys(
    table: TableSchema,
    key_columns: Sequence[str],
    stored: Sequence[Mapping[str, Any]],
    edited: Sequence[Mapping[str, Any]],
) -> None:
    if set(_keys(key_columns, stored)) != set(_keys(key_columns, edited)):
        raise StorageError(
            f"the primary key values of table {table.name!r} must not change; "
            "rows can only be edited in place"
        )


def _apply_updates(
    cursor: _Cursor,
    schema_name: str,
    table: TableSchema,
    key_columns: Sequence[str],
    stored: Sequence[Mapping[str, Any]],
    edited: Sequence[Mapping[str, Any]],
) -> int:
    value_columns = [
        column.name for column in table.columns if column.name not in set(key_columns)
    ]
    if not value_columns:
        return 0

    by_key = {key: row for key, row in zip(_keys(key_columns, stored), stored)}
    changed = [
        row for key, row in zip(_keys(key_columns, edited), edited) if by_key[key] != row
    ]
    if not changed:
        return 0

    statement = sql.SQL("UPDATE {}.{} SET {} WHERE {}").format(
        sql.Identifier(schema_name),
        sql.Identifier(table.name),
        sql.SQL(", ").join(
            sql.SQL("{} = {}").format(sql.Identifier(name), sql.Placeholder())
            for name in value_columns
        ),
        sql.SQL(" AND ").join(
            sql.SQL("{} = {}").format(sql.Identifier(name), sql.Placeholder())
            for name in key_columns
        ),
    )
    cursor.executemany(
        statement,
        [
            tuple(row[name] for name in (*value_columns, *key_columns))
            for row in changed
        ],
    )
    return len(changed)


def _keys(
    columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> list[tuple[Any, ...]]:
    return [tuple(_value(row, name) for name in columns) for row in rows]


def _identifiers(names: Sequence[str]) -> sql.Composed:
    return sql.SQL(", ").join(sql.Identifier(name) for name in names)


def _rows_for(data: Dataset, table_name: str) -> Sequence[Mapping[str, Any]]:
    key = table_name.casefold()
    for name, rows in data.items():
        if name.casefold() == key:
            return rows
    return []


def _has_value(row: Mapping[str, Any], column_name: str) -> bool:
    key = column_name.casefold()
    return any(isinstance(name, str) and name.casefold() == key for name in row)


def _value(row: Mapping[str, Any], column_name: str) -> Any:
    key = column_name.casefold()
    for name, value in row.items():
        if isinstance(name, str) and name.casefold() == key:
            return value
    return None
