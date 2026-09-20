"""Compile natural-language SQL into a single read-only SELECT."""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from data_assistant.schema import DatabaseSchema

DEFAULT_QUERY_LIMIT = 200
MAX_QUERY_LIMIT = 1_000

_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "dblink",
        "dblink_exec",
        "lo_import",
        "lo_export",
        "lo_from_bytea",
        "pg_sleep",
        "pg_terminate_backend",
        "set_config",
        "pg_reload_conf",
        "copy",
        "query_to_xml",
    }
)

_FORBIDDEN_NODES = tuple(
    node
    for name in (
        "Insert",
        "Update",
        "Delete",
        "Merge",
        "Create",
        "Drop",
        "Alter",
        "TruncateTable",
        "Command",
        "Set",
        "Grant",
        "Copy",
        "Analyze",
        "Transaction",
        "Lock",
        "Pragma",
        "LoadData",
    )
    if (node := getattr(exp, name, None)) is not None
)


class QueryError(ValueError):
    """Raised when a query is not a single read-only SELECT of known tables."""


@dataclass(frozen=True, slots=True)
class CompiledQuery:
    """A validated SELECT scoped to one dataset schema."""

    sql: str
    columns: tuple[str, ...]
    limit: int


def compile_readonly_select(
    sql_text: str,
    schema: DatabaseSchema,
    schema_name: str,
    *,
    max_rows: int = DEFAULT_QUERY_LIMIT,
) -> CompiledQuery:
    """Parse `sql_text` and rewrite it as a limited SELECT of `schema` tables.

    # Errors
    Raises `QueryError` when the text is not one SELECT, names unknown tables,
    or includes a write, catalog, or administrative construct.
    """
    if not 1 <= max_rows <= MAX_QUERY_LIMIT:
        raise QueryError(f"row limit must be between 1 and {MAX_QUERY_LIMIT}")
    statements = [statement for statement in _parse(sql_text) if statement is not None]
    if len(statements) != 1:
        raise QueryError("query must be a single SELECT statement")
    expression = statements[0]
    if isinstance(expression, exp.Query) and expression.args.get("into") is not None:
        raise QueryError("SELECT INTO is not allowed")
    if expression.find(exp.Select) is None:
        raise QueryError("query must be a SELECT")
    _reject_writes(expression)
    _reject_forbidden_functions(expression)
    _qualify_tables(expression, schema, schema_name)
    limited = _apply_limit(expression, max_rows)
    rendered = limited.sql(dialect="postgres", identify=True)
    return CompiledQuery(sql=rendered, columns=_column_names(limited), limit=max_rows)


def _parse(sql_text: str) -> list[exp.Expression | None]:
    try:
        return sqlglot.parse(sql_text, dialect="postgres")
    except ParseError as error:
        raise QueryError("query could not be parsed as PostgreSQL SELECT") from error


def _reject_writes(expression: exp.Expression) -> None:
    forbidden = next(expression.find_all(*_FORBIDDEN_NODES), None)
    if forbidden is not None:
        raise QueryError("only read-only SELECT queries are allowed")


def _reject_forbidden_functions(expression: exp.Expression) -> None:
    for node in expression.find_all(exp.Anonymous, exp.Func):
        name = (node.name or "").casefold()
        if name in _FORBIDDEN_FUNCTIONS:
            raise QueryError(f"function {node.name} is not allowed")


def _qualify_tables(
    expression: exp.Expression, schema: DatabaseSchema, schema_name: str
) -> None:
    allowed = {table.name.casefold(): table.name for table in schema.tables}
    cte_names = {cte.alias_or_name.casefold() for cte in expression.find_all(exp.CTE)}
    for table in expression.find_all(exp.Table):
        name = table.name
        if not name:
            continue
        key = name.casefold()
        if key in cte_names:
            continue
        catalog = (table.catalog or "").casefold()
        db = (table.db or "").casefold()
        if catalog and catalog not in {schema_name.casefold()}:
            raise QueryError(f"catalog {table.catalog} is not allowed")
        if db and db not in {schema_name.casefold()}:
            raise QueryError(f"schema {table.db} is not allowed")
        canonical = allowed.get(key)
        if canonical is None:
            raise QueryError(f"unknown table {name}")
        table.set("this", exp.to_identifier(canonical, quoted=True))
        table.set("db", exp.to_identifier(schema_name, quoted=True))
        table.set("catalog", None)


def _apply_limit(expression: exp.Expression, max_rows: int) -> exp.Expression:
    current = expression.args.get("limit") if isinstance(expression, exp.Query) else None
    value = _limit_value(current)
    capped = max_rows if value is None else min(value, max_rows)
    return expression.limit(capped)


def _limit_value(limit: exp.Limit | None) -> int | None:
    if limit is None:
        return None
    expression = limit.expression
    if isinstance(expression, exp.Literal) and expression.is_int:
        parsed = int(expression.this)
        if parsed < 1:
            raise QueryError("LIMIT must be a positive integer")
        return parsed
    raise QueryError("LIMIT must be a positive integer")


def _column_names(expression: exp.Expression) -> tuple[str, ...]:
    select = expression if isinstance(expression, exp.Select) else expression.find(exp.Select)
    if select is None:
        return ()
    names: list[str] = []
    for item in select.expressions:
        names.append(item.alias_or_name or item.sql(dialect="postgres"))
    return tuple(names)
