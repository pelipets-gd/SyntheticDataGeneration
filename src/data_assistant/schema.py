"""Normalized database schema models and MySQL DDL parsing."""

from __future__ import annotations

import re

import sqlglot
from pydantic import BaseModel, Field
from sqlglot.errors import ParseError


class DDLParseError(ValueError):
    """Raised when DDL cannot be represented as a valid normalized schema."""


class SQLType(BaseModel):
    """A normalized SQL column type."""

    name: str
    length: int | None = None
    precision: int | None = None
    scale: int | None = None
    enum_values: list[str] = Field(default_factory=list)


class ColumnSchema(BaseModel):
    """A normalized table column."""

    name: str
    type: SQLType
    nullable: bool = True
    primary_key: bool = False
    auto_increment: bool = False
    unique: bool = False
    default: str | None = None
    checks: list[str] = Field(default_factory=list)


class PrimaryKeyConstraint(BaseModel):
    """A table primary-key constraint."""

    columns: list[str]
    name: str | None = None


class UniqueConstraint(BaseModel):
    """A table unique constraint."""

    columns: list[str]
    name: str | None = None


class ForeignKeySchema(BaseModel):
    """A normalized foreign-key constraint."""

    columns: list[str]
    referenced_table: str
    referenced_columns: list[str]
    name: str | None = None
    on_delete: str | None = None
    on_update: str | None = None


class TableSchema(BaseModel):
    """A normalized database table."""

    name: str
    columns: list[ColumnSchema]
    primary_key: PrimaryKeyConstraint | None = None
    unique_constraints: list[UniqueConstraint] = Field(default_factory=list)
    foreign_keys: list[ForeignKeySchema] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)

    def column(self, name: str) -> ColumnSchema:
        """Return a column using case-insensitive name resolution."""
        key = name.casefold()
        for column in self.columns:
            if column.name.casefold() == key:
                return column
        raise KeyError(f"Unknown column {name!r} in table {self.name!r}")


class DependencyAnalysis(BaseModel):
    """Insertion ordering and cyclic table groups."""

    insertion_groups: list[list[str]]
    cycles: list[list[str]]


class DatabaseSchema(BaseModel):
    """A normalized collection of database tables."""

    tables: list[TableSchema]

    def table(self, name: str) -> TableSchema:
        """Return a table using case-insensitive name resolution."""
        key = name.casefold()
        for table in self.tables:
            if table.name.casefold() == key:
                return table
        raise KeyError(f"Unknown table {name!r}")

    def analyze_dependencies(self) -> DependencyAnalysis:
        """Return dependency-first insertion groups and strongly connected cycles."""
        display_names = {table.name.casefold(): table.name for table in self.tables}
        graph: dict[str, set[str]] = {key: set() for key in display_names}
        for table in self.tables:
            graph[table.name.casefold()].update(
                foreign_key.referenced_table.casefold() for foreign_key in table.foreign_keys
            )

        components = _strongly_connected_components(graph)
        component_by_node = {
            node: index for index, component in enumerate(components) for node in component
        }
        dependencies: dict[int, set[int]] = {index: set() for index in range(len(components))}
        for node, targets in graph.items():
            source_component = component_by_node[node]
            dependencies[source_component].update(
                component_by_node[target]
                for target in targets
                if component_by_node[target] != source_component
            )

        remaining = set(dependencies)
        insertion_groups: list[list[str]] = []
        while remaining:
            available = sorted(
                (
                    component
                    for component in remaining
                    if not (dependencies[component] & remaining)
                ),
                key=lambda component: min(
                    display_names[node].casefold() for node in components[component]
                ),
            )
            if not available:
                raise RuntimeError("Dependency condensation graph unexpectedly contains a cycle")
            insertion_groups.extend(
                [
                    sorted(
                        (display_names[node] for node in components[component]),
                        key=str.casefold,
                    )
                    for component in available
                ]
            )
            remaining.difference_update(available)

        cycles = [
            sorted((display_names[node] for node in component), key=str.casefold)
            for component in components
            if len(component) > 1
            or any(node in graph[node] for node in component)
        ]
        cycles.sort(key=lambda cycle: [name.casefold() for name in cycle])
        return DependencyAnalysis(insertion_groups=insertion_groups, cycles=cycles)


_SUPPORTED_TYPES = {"INT", "VARCHAR", "TEXT", "DATE", "DATETIME", "DECIMAL", "BOOLEAN", "ENUM"}
_IDENTIFIER = r"(?:`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_$]*)"
_CREATE_PATTERN = re.compile(
    rf"^\s*CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+({_IDENTIFIER})\s*\((.*)\)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_ALTER_PATTERN = re.compile(
    rf"^\s*ALTER\s+TABLE\s+({_IDENTIFIER})\s+ADD\s+(.*)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_FOREIGN_KEY_PATTERN = re.compile(
    rf"^(?:CONSTRAINT\s+({_IDENTIFIER})\s+)?FOREIGN\s+KEY\s*"
    rf"\(([^)]+)\)\s+REFERENCES\s+({_IDENTIFIER})\s*\(([^)]+)\)",
    re.IGNORECASE | re.DOTALL,
)
_INLINE_REFERENCE_PATTERN = re.compile(
    rf"\bREFERENCES\s+({_IDENTIFIER})\s*\(([^)]+)\)",
    re.IGNORECASE | re.DOTALL,
)


def parse_ddl(sql: str) -> DatabaseSchema:
    """Parse MySQL-flavored DDL into a validated normalized schema."""
    if not sql.strip():
        raise DDLParseError("DDL is empty")

    statements = _split_statements(sql)
    try:
        parsed_statements = sqlglot.parse(sql, read="mysql")
    except ParseError as error:
        raise DDLParseError(f"Malformed DDL: {error}") from error
    if len(parsed_statements) != len(statements):
        raise DDLParseError("Malformed DDL: could not split every statement")

    tables: list[TableSchema] = []
    pending_alters: list[tuple[str, str]] = []
    for statement in statements:
        create_match = _CREATE_PATTERN.match(statement)
        if create_match:
            tables.append(_parse_create_table(create_match.group(1), create_match.group(2)))
            continue

        alter_match = _ALTER_PATTERN.match(statement)
        if alter_match:
            pending_alters.append((_identifier(alter_match.group(1)), alter_match.group(2)))
            continue

        raise DDLParseError("Unsupported DDL; expected CREATE TABLE or ALTER TABLE ADD CONSTRAINT")

    if not tables:
        raise DDLParseError("Schema is empty; at least one CREATE TABLE is required")

    schema = DatabaseSchema(tables=tables)
    _validate_unique_names(schema)
    for table_name, addition in pending_alters:
        try:
            table = schema.table(table_name)
        except KeyError as error:
            raise DDLParseError(f"ALTER TABLE references missing table {table_name!r}") from error
        table.foreign_keys.append(_parse_foreign_key(addition))

    _resolve_and_validate_references(schema)
    return schema


def _parse_create_table(raw_name: str, body: str) -> TableSchema:
    table = TableSchema(name=_identifier(raw_name), columns=[])
    primary_keys: list[PrimaryKeyConstraint] = []
    unique_constraints: list[UniqueConstraint] = []

    for definition in _split_top_level(body, ","):
        item = definition.strip()
        if not item:
            raise DDLParseError(f"Malformed empty definition in table {table.name!r}")

        if _FOREIGN_KEY_PATTERN.match(item):
            table.foreign_keys.append(_parse_foreign_key(item))
        elif re.match(r"^(?:CONSTRAINT\s+\S+\s+)?PRIMARY\s+KEY\b", item, re.IGNORECASE):
            primary_keys.append(_parse_primary_key(item))
        elif re.match(r"^(?:CONSTRAINT\s+\S+\s+)?UNIQUE\b", item, re.IGNORECASE):
            unique_constraints.append(_parse_unique_constraint(item))
        elif re.match(r"^(?:CONSTRAINT\s+\S+\s+)?CHECK\b", item, re.IGNORECASE):
            table.checks.append(_check_expression(item))
        else:
            column, inline_foreign_key = _parse_column(item)
            if any(existing.name.casefold() == column.name.casefold() for existing in table.columns):
                raise DDLParseError(
                    f"Duplicate column {column.name!r} in table {table.name!r}"
                )
            table.columns.append(column)
            if inline_foreign_key:
                table.foreign_keys.append(inline_foreign_key)
            if column.primary_key:
                primary_keys.append(PrimaryKeyConstraint(columns=[column.name]))
            if column.unique:
                unique_constraints.append(UniqueConstraint(columns=[column.name]))

    if not table.columns:
        raise DDLParseError(f"Table {table.name!r} has no columns")
    if len(primary_keys) > 1:
        raise DDLParseError(f"Table {table.name!r} has multiple primary-key constraints")
    if primary_keys:
        table.primary_key = primary_keys[0]
        table.primary_key.columns = [
            _table_column_or_error(table, column_name).name
            for column_name in table.primary_key.columns
        ]
        for column_name in table.primary_key.columns:
            column = _table_column_or_error(table, column_name)
            column.primary_key = len(table.primary_key.columns) == 1
            column.nullable = False
    table.unique_constraints = unique_constraints
    for constraint in table.unique_constraints:
        constraint.columns = [
            _table_column_or_error(table, column_name).name
            for column_name in constraint.columns
        ]
        if len(constraint.columns) == 1:
            _table_column_or_error(table, constraint.columns[0]).unique = True
    return table


def _parse_column(definition: str) -> tuple[ColumnSchema, ForeignKeySchema | None]:
    name_match = re.match(rf"^\s*({_IDENTIFIER})\s+(.*)$", definition, re.DOTALL)
    if not name_match:
        raise DDLParseError(f"Malformed column definition: {definition!r}")
    name = _identifier(name_match.group(1))
    sql_type, constraints = _parse_type(name_match.group(2))

    primary_key = bool(re.search(r"\bPRIMARY\s+KEY\b", constraints, re.IGNORECASE))
    nullable = not bool(re.search(r"\bNOT\s+NULL\b", constraints, re.IGNORECASE))
    if primary_key:
        nullable = False
    default = _default_value(constraints)
    checks = _check_expressions(constraints)

    inline_foreign_key = None
    reference_match = _INLINE_REFERENCE_PATTERN.search(constraints)
    if reference_match:
        on_delete, on_update = _parse_foreign_key_actions(
            constraints[reference_match.end() :], strict=True
        )
        inline_foreign_key = ForeignKeySchema(
            columns=[name],
            referenced_table=_identifier(reference_match.group(1)),
            referenced_columns=_identifier_list(reference_match.group(2)),
            on_delete=on_delete,
            on_update=on_update,
        )

    return (
        ColumnSchema(
            name=name,
            type=sql_type,
            nullable=nullable,
            primary_key=primary_key,
            auto_increment=bool(
                re.search(r"\bAUTO_INCREMENT\b", constraints, re.IGNORECASE)
            ),
            unique=bool(re.search(r"\bUNIQUE\b", constraints, re.IGNORECASE)),
            default=default,
            checks=checks,
        ),
        inline_foreign_key,
    )


def _parse_type(definition: str) -> tuple[SQLType, str]:
    match = re.match(r"^\s*([A-Za-z]+)", definition)
    if not match:
        raise DDLParseError(f"Missing SQL type in column definition: {definition!r}")
    type_name = match.group(1).upper()
    if type_name not in _SUPPORTED_TYPES:
        raise DDLParseError(f"unsupported SQL type {type_name!r}")

    position = match.end()
    arguments: str | None = None
    if position < len(definition) and definition[position:].lstrip().startswith("("):
        open_position = position + len(definition[position:]) - len(definition[position:].lstrip())
        arguments, close_position = _parenthesized_content(definition, open_position)
        position = close_position + 1

    sql_type = SQLType(name=type_name)
    if type_name == "VARCHAR":
        if arguments is None or not arguments.strip().isdigit():
            raise DDLParseError("VARCHAR requires an integer length")
        sql_type.length = int(arguments)
    elif type_name == "DECIMAL":
        values = [part.strip() for part in (arguments or "").split(",")]
        if not values[0].isdigit() or (
            len(values) == 2 and not values[1].isdigit()
        ) or len(values) not in {1, 2}:
            raise DDLParseError("DECIMAL requires precision and optional scale")
        sql_type.precision = int(values[0])
        sql_type.scale = int(values[1]) if len(values) == 2 else 0
    elif type_name == "ENUM":
        if arguments is None:
            raise DDLParseError("ENUM requires quoted values")
        sql_type.enum_values = _parse_enum_values(arguments)
        if not sql_type.enum_values:
            raise DDLParseError("ENUM requires at least one value")
    elif arguments is not None:
        raise DDLParseError(f"Unsupported arguments for SQL type {type_name!r}")
    return sql_type, definition[position:]


def _parse_enum_values(arguments: str) -> list[str]:
    values: list[str] = []
    for value in _split_top_level(arguments, ","):
        value = value.strip()
        if len(value) < 2 or value[0] not in {"'", '"'} or value[-1] != value[0]:
            raise DDLParseError(f"ENUM value must be quoted: {value!r}")
        quote = value[0]
        values.append(value[1:-1].replace(quote * 2, quote))
    return values


def _parse_foreign_key(definition: str) -> ForeignKeySchema:
    match = _FOREIGN_KEY_PATTERN.match(definition.strip())
    if not match:
        raise DDLParseError(
            f"Unsupported ALTER/foreign-key definition: {definition.strip()!r}"
        )
    columns = _identifier_list(match.group(2))
    referenced_columns = _identifier_list(match.group(4))
    if len(columns) != len(referenced_columns):
        raise DDLParseError("Foreign key column counts do not match")
    on_delete, on_update = _parse_foreign_key_actions(
        definition.strip()[match.end() :], strict=True
    )
    return ForeignKeySchema(
        name=_identifier(match.group(1)) if match.group(1) else None,
        columns=columns,
        referenced_table=_identifier(match.group(3)),
        referenced_columns=referenced_columns,
        on_delete=on_delete,
        on_update=on_update,
    )


def _resolve_and_validate_references(schema: DatabaseSchema) -> None:
    table_names = {table.name.casefold(): table for table in schema.tables}
    for table in schema.tables:
        local_columns = {column.name.casefold(): column.name for column in table.columns}
        for foreign_key in table.foreign_keys:
            for index, column_name in enumerate(foreign_key.columns):
                resolved = local_columns.get(column_name.casefold())
                if resolved is None:
                    raise DDLParseError(
                        f"Foreign key in {table.name!r} references missing local column "
                        f"{column_name!r}"
                    )
                foreign_key.columns[index] = resolved

            referenced_table = table_names.get(foreign_key.referenced_table.casefold())
            if referenced_table is None:
                raise DDLParseError(
                    f"Foreign key in {table.name!r} references missing table "
                    f"{foreign_key.referenced_table!r}"
                )
            foreign_key.referenced_table = referenced_table.name
            referenced_columns = {
                column.name.casefold(): column.name for column in referenced_table.columns
            }
            for index, column_name in enumerate(foreign_key.referenced_columns):
                resolved = referenced_columns.get(column_name.casefold())
                if resolved is None:
                    raise DDLParseError(
                        f"Foreign key in {table.name!r} references missing column "
                        f"{foreign_key.referenced_table}.{column_name}"
                    )
                foreign_key.referenced_columns[index] = resolved


def _validate_unique_names(schema: DatabaseSchema) -> None:
    seen: dict[str, str] = {}
    for table in schema.tables:
        key = table.name.casefold()
        if key in seen:
            raise DDLParseError(f"Duplicate table {table.name!r}")
        seen[key] = table.name


def _parse_primary_key(definition: str) -> PrimaryKeyConstraint:
    match = re.fullmatch(
        rf"\s*(?:CONSTRAINT\s+({_IDENTIFIER})\s+)?"
        r"PRIMARY\s+KEY\s*\(([^)]+)\)\s*",
        definition,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise DDLParseError(f"Malformed PRIMARY KEY constraint: {definition!r}")
    return PrimaryKeyConstraint(
        name=_identifier(match.group(1)) if match.group(1) else None,
        columns=_identifier_list(match.group(2)),
    )


def _parse_unique_constraint(definition: str) -> UniqueConstraint:
    match = re.fullmatch(
        rf"\s*(?:CONSTRAINT\s+({_IDENTIFIER})\s+)?"
        rf"UNIQUE(?:\s+(?:KEY|INDEX))?(?:\s+({_IDENTIFIER}))?\s*"
        r"\(([^)]+)\)\s*",
        definition,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise DDLParseError(f"Malformed UNIQUE constraint: {definition!r}")
    constraint_name = match.group(1) or match.group(2)
    return UniqueConstraint(
        name=_identifier(constraint_name) if constraint_name else None,
        columns=_identifier_list(match.group(3)),
    )


def _check_expression(definition: str) -> str:
    check_position = re.search(r"\bCHECK\b", definition, re.IGNORECASE)
    if not check_position:
        raise DDLParseError(f"Malformed CHECK constraint: {definition!r}")
    open_position = definition.find("(", check_position.end())
    if open_position < 0:
        raise DDLParseError(f"Malformed CHECK constraint: {definition!r}")
    expression, _ = _parenthesized_content(definition, open_position)
    return expression.strip()


def _check_expressions(definition: str) -> list[str]:
    expressions: list[str] = []
    position = 0
    while match := re.search(r"\bCHECK\b", definition[position:], re.IGNORECASE):
        check_end = position + match.end()
        open_position = check_end
        while open_position < len(definition) and definition[open_position].isspace():
            open_position += 1
        if open_position >= len(definition) or definition[open_position] != "(":
            raise DDLParseError(f"Malformed CHECK constraint: {definition!r}")
        expression, close_position = _parenthesized_content(definition, open_position)
        expressions.append(expression.strip())
        position = close_position + 1
    return expressions


_FOREIGN_KEY_ACTION_PATTERN = re.compile(
    r"\bON\s+(DELETE|UPDATE)\s+"
    r"(RESTRICT|CASCADE|SET\s+NULL|NO\s+ACTION|SET\s+DEFAULT)\b",
    re.IGNORECASE,
)


def _parse_foreign_key_actions(
    suffix: str, *, strict: bool
) -> tuple[str | None, str | None]:
    actions: dict[str, str] = {}
    consumed: list[tuple[int, int]] = []
    for match in _FOREIGN_KEY_ACTION_PATTERN.finditer(suffix):
        event = match.group(1).upper()
        if event in actions:
            raise DDLParseError(f"Duplicate foreign-key ON {event} action")
        actions[event] = " ".join(match.group(2).upper().split())
        consumed.append(match.span())

    if re.search(r"\bON\s+(?:DELETE|UPDATE)\b", suffix, re.IGNORECASE) and not consumed:
        raise DDLParseError(f"Unsupported foreign-key action suffix: {suffix.strip()!r}")
    if strict:
        remainder = list(suffix)
        for start, end in consumed:
            remainder[start:end] = " " * (end - start)
        if "".join(remainder).strip():
            raise DDLParseError(f"Unsupported foreign-key suffix: {suffix.strip()!r}")
    return actions.get("DELETE"), actions.get("UPDATE")


def _default_value(constraints: str) -> str | None:
    match = re.search(
        r"\bDEFAULT\s+("
        r"CURRENT_TIMESTAMP(?:\s*\(\s*\))?"
        r"|'(?:''|[^'])*'"
        r'|"(?:\"\"|[^"])*"'
        r"|[^\s,]+)",
        constraints,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group(1).strip()
    if re.fullmatch(r"CURRENT_TIMESTAMP(?:\s*\(\s*\))?", value, re.IGNORECASE):
        return "CURRENT_TIMESTAMP"
    return value


def _identifier_list(value: str) -> list[str]:
    identifiers = [_identifier(part.strip()) for part in _split_top_level(value, ",")]
    if not identifiers or any(not identifier for identifier in identifiers):
        raise DDLParseError(f"Malformed identifier list: {value!r}")
    return identifiers


def _identifier(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in {'`', '"'} and value[-1] == value[0]:
        return value[1:-1]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", value):
        raise DDLParseError(f"Malformed identifier {value!r}")
    return value


def _table_column_or_error(table: TableSchema, name: str) -> ColumnSchema:
    try:
        return table.column(name)
    except KeyError as error:
        raise DDLParseError(
            f"Constraint in {table.name!r} references missing column {name!r}"
        ) from error


def _parenthesized_content(value: str, open_position: int) -> tuple[str, int]:
    depth = 0
    quote: str | None = None
    index = open_position
    while index < len(value):
        character = value[index]
        if quote:
            if character == quote:
                if index + 1 < len(value) and value[index + 1] == quote:
                    index += 1
                else:
                    quote = None
        elif character in {"'", '"', "`"}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return value[open_position + 1 : index], index
        index += 1
    raise DDLParseError("Unclosed parenthesis in DDL")


def _split_statements(sql: str) -> list[str]:
    uncommented = _strip_line_comments(sql)
    return [part.strip() for part in _split_top_level(uncommented, ";") if part.strip()]


def _strip_line_comments(sql: str) -> str:
    result: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(sql):
        character = sql[index]
        if quote:
            result.append(character)
            if character == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    result.append(sql[index + 1])
                    index += 1
                else:
                    quote = None
        elif character in {"'", '"', "`"}:
            quote = character
            result.append(character)
        elif character == "-" and index + 1 < len(sql) and sql[index + 1] == "-":
            index += 2
            while index < len(sql) and sql[index] != "\n":
                index += 1
            if index < len(sql):
                result.append("\n")
        else:
            result.append(character)
        index += 1
    return "".join(result)


def _split_top_level(value: str, separator: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(value):
        character = value[index]
        if quote:
            if character == quote:
                if index + 1 < len(value) and value[index + 1] == quote:
                    index += 1
                else:
                    quote = None
        elif character in {"'", '"', "`"}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise DDLParseError("Unexpected closing parenthesis in DDL")
        elif character == separator and depth == 0:
            parts.append(value[start:index])
            start = index + 1
        index += 1
    if quote:
        raise DDLParseError("Unclosed quoted value in DDL")
    if depth:
        raise DDLParseError("Unclosed parenthesis in DDL")
    parts.append(value[start:])
    return parts


def _strongly_connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    indices: dict[str, int] = {}
    low_links: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        low_links[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in sorted(graph[node]):
            if target not in indices:
                visit(target)
                low_links[node] = min(low_links[node], low_links[target])
            elif target in on_stack:
                low_links[node] = min(low_links[node], indices[target])

        if low_links[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(component)

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return components
