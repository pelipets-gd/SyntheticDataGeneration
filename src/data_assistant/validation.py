"""Independent constraint validation for generated relational datasets.

CHECK constraints are parsed into a small safe expression tree; SQL is never
evaluated or executed. Literals are coerced into the Python type of the column
they are compared against. Forms outside the supported grammar are reported as
``unsupported_check`` violations, and comparisons whose operand types cannot be
ordered as ``check_unevaluable`` ones, instead of being treated as satisfied.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel

from data_assistant.schema import ColumnSchema, DatabaseSchema, TableSchema


ViolationKind = Literal[
    "missing_table",
    "unknown_table",
    "row_shape",
    "missing_column",
    "unknown_column",
    "not_null",
    "type_mismatch",
    "varchar_length",
    "decimal_precision",
    "enum_value",
    "check_failed",
    "check_unevaluable",
    "unsupported_check",
    "primary_key_null",
    "primary_key_duplicate",
    "unique_duplicate",
    "foreign_key_missing",
]

Dataset = Mapping[str, Sequence[Any]]


class CheckExpressionError(ValueError):
    """Raised when a CHECK expression is outside the safely supported grammar."""


class Violation(BaseModel):
    """One actionable constraint violation located in the dataset."""

    kind: ViolationKind
    table: str
    message: str
    column: str | None = None
    row_index: int | None = None


class DataValidationError(ValueError):
    """Raised when a dataset violates its schema; carries every violation."""

    def __init__(self, violations: Sequence[Violation]) -> None:
        self.violations = list(violations)
        super().__init__(_summarize(self.violations))


@dataclass(frozen=True)
class ColumnConstraint:
    """Value bounds derived from CHECK expressions for a single column.

    Bounds carry the Python type of the column they belong to: `Decimal` for
    INT and DECIMAL, `date` for DATE, and `datetime` for DATETIME.
    """

    minimum: Decimal | date | datetime | None = None
    maximum: Decimal | date | datetime | None = None
    minimum_exclusive: bool = False
    maximum_exclusive: bool = False
    allowed: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class ColumnComparison:
    """A CHECK comparison between two columns of the same row."""

    left: str
    operator: str
    right: str
    expression: str
    conjunct: bool


def validate_dataset(schema: DatabaseSchema, data: Dataset) -> list[Violation]:
    """Check a dataset against every schema constraint and return all violations."""
    violations: list[Violation] = []
    known = {table.name.casefold() for table in schema.tables}
    for name in data:
        if name.casefold() not in known:
            violations.append(
                Violation(
                    kind="unknown_table",
                    table=name,
                    message=f"Table {name!r} is not part of the schema",
                )
            )

    for table in schema.tables:
        rows = _rows_for(data, table.name)
        if rows is None:
            violations.append(
                Violation(
                    kind="missing_table",
                    table=table.name,
                    message=f"Table {table.name!r} has no rows in the dataset",
                )
            )
            continue
        violations.extend(_validate_table(table, rows))

    violations.extend(_validate_foreign_keys(schema, data))
    return violations


def assert_valid_dataset(schema: DatabaseSchema, data: Dataset) -> None:
    """Validate a dataset and raise on the first failing validation pass.

    # Errors
    Raises `DataValidationError` when any constraint is violated.
    """
    violations = validate_dataset(schema, data)
    if violations:
        raise DataValidationError(violations)


def unsupported_check_violations(table: TableSchema) -> list[Violation]:
    """Return one violation per CHECK expression that cannot be safely validated."""
    return _parse_table_checks(table)[1]


def column_comparisons(table: TableSchema) -> list[ColumnComparison]:
    """Return every CHECK comparison whose two operands are columns of `table`.

    Column names are casefolded. `conjunct` is False when the comparison sits
    under OR/NOT, where satisfying it row by row is not a matter of ordering.
    """
    found: list[ColumnComparison] = []
    for _, expression, node in _parse_table_checks(table)[0]:
        top_level = {id(part) for part in _conjuncts(node)}
        for comparison in _column_comparison_nodes(node):
            found.append(
                ColumnComparison(
                    left=comparison.left.name,
                    operator=comparison.operator,
                    right=comparison.right.name,
                    expression=expression,
                    conjunct=id(comparison) in top_level,
                )
            )
    return found


def derive_column_constraint(
    table: TableSchema, column: ColumnSchema
) -> ColumnConstraint:
    """Derive value bounds for one column from the conjunctive CHECKs of its table.

    Expressions that cannot be parsed or coerced, or that constrain the column
    only through OR/NOT, contribute nothing.
    """
    minimum: Decimal | date | datetime | None = None
    maximum: Decimal | date | datetime | None = None
    minimum_exclusive = False
    maximum_exclusive = False
    allowed: tuple[Any, ...] | None = None
    key = column.name.casefold()
    columns = _columns_by_name(table)

    for expression in (*column.checks, *table.checks):
        try:
            node = _coerce_literals(parse_check_expression(expression), columns)
        except CheckExpressionError:
            continue
        for conjunct in _conjuncts(node):
            bound = _bound_from(conjunct, key)
            if bound is None:
                continue
            kind, value = bound
            if kind == "between":
                low, high = value
                if _is_ordered_literal(low) and (minimum is None or low > minimum):
                    minimum, minimum_exclusive = low, False
                if _is_ordered_literal(high) and (maximum is None or high < maximum):
                    maximum, maximum_exclusive = high, False
            elif kind == "allowed":
                allowed = (
                    value
                    if allowed is None
                    else tuple(member for member in allowed if member in value)
                )
            elif kind in {"min", "min_exclusive"}:
                if minimum is None or value > minimum:
                    minimum, minimum_exclusive = value, kind == "min_exclusive"
            elif maximum is None or value < maximum:
                maximum, maximum_exclusive = value, kind == "max_exclusive"

    return ColumnConstraint(
        minimum=minimum,
        maximum=maximum,
        minimum_exclusive=minimum_exclusive,
        maximum_exclusive=maximum_exclusive,
        allowed=allowed,
    )


def _summarize(violations: Sequence[Violation]) -> str:
    head = "; ".join(violation.message for violation in violations[:3])
    if len(violations) > 3:
        return f"{len(violations)} constraint violations: {head}; ..."
    return f"{len(violations)} constraint violations: {head}"


def _rows_for(data: Dataset, table_name: str) -> Sequence[Any] | None:
    key = table_name.casefold()
    for name, rows in data.items():
        if name.casefold() == key:
            return rows
    return None


def _row_value(row: Mapping[str, Any], column_name: str) -> Any:
    key = column_name.casefold()
    for name, value in row.items():
        if isinstance(name, str) and name.casefold() == key:
            return value
    return None


def _has_column(row: Mapping[str, Any], column_name: str) -> bool:
    key = column_name.casefold()
    return any(isinstance(name, str) and name.casefold() == key for name in row)


def _validate_table(table: TableSchema, rows: Sequence[Any]) -> list[Violation]:
    checks, violations = _parse_table_checks(table)

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            violations.append(
                Violation(
                    kind="row_shape",
                    table=table.name,
                    row_index=index,
                    message=f"{table.name} row {index} is {type(row).__name__}, expected a mapping",
                )
            )
            continue
        violations.extend(_validate_row(table, index, row))
        violations.extend(_validate_row_checks(table, index, row, checks))

    violations.extend(_validate_key_uniqueness(table, rows))
    return violations


def _validate_row(table: TableSchema, index: int, row: Mapping[str, Any]) -> list[Violation]:
    violations: list[Violation] = []
    known = {column.name.casefold() for column in table.columns}
    for name in row:
        if not isinstance(name, str) or name.casefold() not in known:
            violations.append(
                Violation(
                    kind="unknown_column",
                    table=table.name,
                    column=str(name),
                    row_index=index,
                    message=(
                        f"{table.name} row {index} has column {name!r} "
                        "that is not in the schema"
                    ),
                )
            )

    for column in table.columns:
        if not _has_column(row, column.name):
            if column.nullable or column.default is not None:
                continue
            violations.append(
                Violation(
                    kind="missing_column",
                    table=table.name,
                    column=column.name,
                    row_index=index,
                    message=(
                        f"{table.name} row {index} is missing required column "
                        f"{column.name!r}, which is NOT NULL and has no default"
                    ),
                )
            )
            continue
        violation = _validate_value(table, column, index, _row_value(row, column.name))
        if violation is not None:
            violations.append(violation)
    return violations


def _validate_value(
    table: TableSchema, column: ColumnSchema, index: int, value: Any
) -> Violation | None:
    location = f"{table.name} row {index} column {column.name!r}"
    if value is None:
        if column.nullable:
            return None
        return Violation(
            kind="not_null",
            table=table.name,
            column=column.name,
            row_index=index,
            message=f"{location} is NULL but the column is NOT NULL",
        )

    expected = _expected_python_type(column)
    if expected is not None and not _matches_type(column, value):
        return Violation(
            kind="type_mismatch",
            table=table.name,
            column=column.name,
            row_index=index,
            message=(
                f"{location} is {type(value).__name__}, expected {expected} "
                f"for SQL type {column.type.name}"
            ),
        )

    if column.type.name == "VARCHAR" and column.type.length is not None:
        if len(value) > column.type.length:
            return Violation(
                kind="varchar_length",
                table=table.name,
                column=column.name,
                row_index=index,
                message=(
                    f"{location} has length {len(value)}, exceeding "
                    f"VARCHAR({column.type.length})"
                ),
            )

    if column.type.name == "DECIMAL":
        problem = _decimal_problem(value, column)
        if problem is not None:
            return Violation(
                kind="decimal_precision",
                table=table.name,
                column=column.name,
                row_index=index,
                message=f"{location} {problem}",
            )

    if column.type.name == "ENUM" and value not in column.type.enum_values:
        return Violation(
            kind="enum_value",
            table=table.name,
            column=column.name,
            row_index=index,
            message=(
                f"{location} is {value!r}, which is not one of "
                f"{', '.join(repr(member) for member in column.type.enum_values)}"
            ),
        )
    return None


def _expected_python_type(column: ColumnSchema) -> str | None:
    return {
        "INT": "int",
        "BOOLEAN": "bool",
        "VARCHAR": "str",
        "TEXT": "str",
        "ENUM": "str",
        "DECIMAL": "decimal.Decimal",
        "DATE": "datetime.date",
        "DATETIME": "datetime.datetime",
    }.get(column.type.name)


def _matches_type(column: ColumnSchema, value: Any) -> bool:
    name = column.type.name
    if name == "INT":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "BOOLEAN":
        return isinstance(value, bool)
    if name in {"VARCHAR", "TEXT", "ENUM"}:
        return isinstance(value, str)
    if name == "DECIMAL":
        return isinstance(value, Decimal)
    if name == "DATE":
        return isinstance(value, date) and not isinstance(value, datetime)
    if name == "DATETIME":
        return isinstance(value, datetime)
    return True


def _decimal_problem(value: Decimal, column: ColumnSchema) -> str | None:
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):
        return f"is {value!r}, which is not a finite decimal"
    precision = column.type.precision or 0
    scale = column.type.scale or 0
    if -exponent > scale:
        return f"has {-exponent} decimal places, exceeding DECIMAL({precision}, {scale})"
    if abs(value) >= Decimal(10) ** (precision - scale):
        return f"has too many integer digits for DECIMAL({precision}, {scale})"
    return None


def _validate_key_uniqueness(table: TableSchema, rows: Sequence[Any]) -> list[Violation]:
    violations: list[Violation] = []
    if table.primary_key is not None:
        columns = table.primary_key.columns
        seen: set[tuple[Any, ...]] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            values = tuple(_row_value(row, name) for name in columns)
            null_columns = [name for name, value in zip(columns, values) if value is None]
            if null_columns:
                violations.append(
                    Violation(
                        kind="primary_key_null",
                        table=table.name,
                        column=null_columns[0] if len(columns) == 1 else None,
                        row_index=index,
                        message=(
                            f"{table.name} row {index} has NULL primary-key column(s) "
                            f"{', '.join(null_columns)}"
                        ),
                    )
                )
                continue
            if values in seen:
                violations.append(
                    Violation(
                        kind="primary_key_duplicate",
                        table=table.name,
                        column=columns[0] if len(columns) == 1 else None,
                        row_index=index,
                        message=(
                            f"{table.name} row {index} repeats primary key "
                            f"({', '.join(columns)}) = {values}"
                        ),
                    )
                )
            seen.add(values)

    for constraint in table.unique_constraints:
        seen_unique: set[tuple[Any, ...]] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            values = tuple(_row_value(row, name) for name in constraint.columns)
            if any(value is None for value in values):
                continue
            if values in seen_unique:
                violations.append(
                    Violation(
                        kind="unique_duplicate",
                        table=table.name,
                        column=(
                            constraint.columns[0] if len(constraint.columns) == 1 else None
                        ),
                        row_index=index,
                        message=(
                            f"{table.name} row {index} repeats unique "
                            f"({', '.join(constraint.columns)}) = {values}"
                        ),
                    )
                )
            seen_unique.add(values)
    return violations


def _validate_foreign_keys(schema: DatabaseSchema, data: Dataset) -> list[Violation]:
    violations: list[Violation] = []
    for table in schema.tables:
        rows = _rows_for(data, table.name)
        if rows is None:
            continue
        for foreign_key in table.foreign_keys:
            parent_rows = _rows_for(data, foreign_key.referenced_table)
            if parent_rows is None:
                continue
            parent_keys = {
                tuple(_row_value(row, name) for name in foreign_key.referenced_columns)
                for row in parent_rows
                if isinstance(row, Mapping)
            }
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    continue
                values = tuple(_row_value(row, name) for name in foreign_key.columns)
                if any(value is None for value in values):
                    continue
                if values in parent_keys:
                    continue
                violations.append(
                    Violation(
                        kind="foreign_key_missing",
                        table=table.name,
                        column=(
                            foreign_key.columns[0] if len(foreign_key.columns) == 1 else None
                        ),
                        row_index=index,
                        message=(
                            f"{table.name} row {index} references "
                            f"{foreign_key.referenced_table}"
                            f"({', '.join(foreign_key.referenced_columns)}) = {values}, "
                            "which does not exist"
                        ),
                    )
                )
    return violations


def _parse_table_checks(
    table: TableSchema,
) -> tuple[list[tuple[str | None, str, Any]], list[Violation]]:
    violations: list[Violation] = []
    columns = _columns_by_name(table)
    expressions: list[tuple[str | None, str]] = [
        (column.name, expression)
        for column in table.columns
        for expression in column.checks
    ]
    expressions.extend((None, expression) for expression in table.checks)

    parsed: list[tuple[str | None, str, Any]] = []
    for column_name, expression in expressions:
        try:
            node = parse_check_expression(expression)
        except CheckExpressionError as error:
            violations.append(_unsupported(table, column_name, expression, str(error)))
            continue
        unknown = sorted(
            name for name in _referenced_columns(node) if name not in columns
        )
        if unknown:
            violations.append(
                _unsupported(
                    table,
                    column_name,
                    expression,
                    f"unknown column(s) {', '.join(unknown)}",
                )
            )
            continue
        try:
            node = _coerce_literals(node, columns)
        except CheckExpressionError as error:
            violations.append(_unsupported(table, column_name, expression, str(error)))
            continue
        parsed.append((column_name, expression, node))
    return parsed, violations


def _unsupported(
    table: TableSchema, column_name: str | None, expression: str, reason: str
) -> Violation:
    return Violation(
        kind="unsupported_check",
        table=table.name,
        column=column_name,
        message=(
            f"{table.name} CHECK ({expression}) is not safely supported and was "
            f"not validated: {reason}"
        ),
    )


def _columns_by_name(table: TableSchema) -> dict[str, ColumnSchema]:
    return {column.name.casefold(): column for column in table.columns}


def _validate_row_checks(
    table: TableSchema,
    index: int,
    row: Mapping[str, Any],
    checks: Sequence[tuple[str | None, str, Any]],
) -> list[Violation]:
    values = {
        name.casefold(): value for name, value in row.items() if isinstance(name, str)
    }
    violations: list[Violation] = []
    for column_name, expression, node in checks:
        result = _evaluate(node, values)
        if result is False:
            violations.append(
                Violation(
                    kind="check_failed",
                    table=table.name,
                    column=column_name,
                    row_index=index,
                    message=f"{table.name} row {index} fails CHECK ({expression})",
                )
            )
        elif result is _INCOMPARABLE:
            violations.append(
                Violation(
                    kind="check_unevaluable",
                    table=table.name,
                    column=column_name,
                    row_index=index,
                    message=(
                        f"{table.name} row {index} cannot evaluate CHECK ({expression}): "
                        "the compared values have incomparable types"
                    ),
                )
            )
    return violations


@dataclass(frozen=True)
class _ColumnRef:
    name: str


@dataclass(frozen=True)
class _Value:
    value: Any


@dataclass(frozen=True)
class _Comparison:
    left: Any
    operator: str
    right: Any


@dataclass(frozen=True)
class _Between:
    operand: Any
    low: Any
    high: Any
    negated: bool


@dataclass(frozen=True)
class _In:
    operand: Any
    values: tuple[Any, ...]
    negated: bool


@dataclass(frozen=True)
class _IsNull:
    operand: Any
    negated: bool


@dataclass(frozen=True)
class _And:
    parts: tuple[Any, ...]


@dataclass(frozen=True)
class _Or:
    parts: tuple[Any, ...]


@dataclass(frozen=True)
class _Not:
    operand: Any


_TOKEN_PATTERN = re.compile(
    r"(?P<number>\d+(?:\.\d+)?)"
    r"|(?P<string>'(?:''|[^'])*')"
    r"|(?P<quoted>`[^`]+`)"
    r"|(?P<identifier>[A-Za-z_][A-Za-z0-9_$]*)"
    r"|(?P<operator><=|>=|<>|!=|=|<|>)"
    r"|(?P<punct>[(),])"
    r"|(?P<minus>-)"
)
_KEYWORDS = {"AND", "OR", "NOT", "IS", "IN", "BETWEEN", "NULL", "TRUE", "FALSE"}


def parse_check_expression(expression: str) -> Any:
    """Parse a CHECK expression into a safe evaluable tree.

    # Errors
    Raises `CheckExpressionError` for any form outside the supported grammar.
    """
    return _Parser(_tokenize(expression)).parse()


def _tokenize(expression: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(expression):
        if expression[position].isspace():
            position += 1
            continue
        match = _TOKEN_PATTERN.match(expression, position)
        if match is None:
            raise CheckExpressionError(
                f"unsupported token at position {position}: {expression[position]!r}"
            )
        kind = match.lastgroup or ""
        tokens.append((kind, match.group()))
        position = match.end()
    if not tokens:
        raise CheckExpressionError("empty CHECK expression")
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self._tokens = tokens
        self._position = 0

    def parse(self) -> Any:
        node = self._or_expression()
        if self._position != len(self._tokens):
            raise CheckExpressionError(
                f"unexpected trailing input at token {self._tokens[self._position][1]!r}"
            )
        return node

    def _or_expression(self) -> Any:
        parts = [self._and_expression()]
        while self._accept_keyword("OR"):
            parts.append(self._and_expression())
        return parts[0] if len(parts) == 1 else _Or(tuple(parts))

    def _and_expression(self) -> Any:
        parts = [self._unary()]
        while self._accept_keyword("AND"):
            parts.append(self._unary())
        return parts[0] if len(parts) == 1 else _And(tuple(parts))

    def _unary(self) -> Any:
        if self._accept_keyword("NOT"):
            return _Not(self._unary())
        return self._predicate()

    def _predicate(self) -> Any:
        if self._peek() == ("punct", "("):
            self._advance()
            node = self._or_expression()
            self._expect_punct(")")
            return node

        operand = self._operand()
        if self._accept_keyword("IS"):
            negated = self._accept_keyword("NOT")
            if not self._accept_keyword("NULL"):
                raise CheckExpressionError("expected NULL after IS")
            return _IsNull(operand, negated)

        negated = self._accept_keyword("NOT")
        if self._accept_keyword("IN"):
            self._expect_punct("(")
            values = [self._operand()]
            while self._peek() == ("punct", ","):
                self._advance()
                values.append(self._operand())
            self._expect_punct(")")
            return _In(operand, tuple(values), negated)
        if self._accept_keyword("BETWEEN"):
            low = self._operand()
            if not self._accept_keyword("AND"):
                raise CheckExpressionError("expected AND in BETWEEN")
            return _Between(operand, low, self._operand(), negated)
        if negated:
            raise CheckExpressionError("NOT must be followed by IN or BETWEEN here")

        token = self._peek()
        if token is None or token[0] != "operator":
            raise CheckExpressionError("expected a comparison operator")
        self._advance()
        return _Comparison(operand, token[1], self._operand())

    def _operand(self) -> Any:
        token = self._peek()
        if token is None:
            raise CheckExpressionError("unexpected end of CHECK expression")
        kind, text = token
        self._advance()
        if kind == "minus":
            following = self._peek()
            if following is None or following[0] != "number":
                raise CheckExpressionError("arithmetic is not supported in CHECK")
            self._advance()
            return _Value(-Decimal(following[1]))
        if kind == "number":
            return _Value(Decimal(text))
        if kind == "string":
            return _Value(text[1:-1].replace("''", "'"))
        if kind == "quoted":
            return _ColumnRef(text[1:-1].casefold())
        if kind == "identifier":
            upper = text.upper()
            if upper == "NULL":
                return _Value(None)
            if upper in {"TRUE", "FALSE"}:
                return _Value(upper == "TRUE")
            if upper in _KEYWORDS:
                raise CheckExpressionError(f"unexpected keyword {text!r}")
            return _ColumnRef(text.casefold())
        raise CheckExpressionError(f"unsupported operand {text!r}")

    def _peek(self) -> tuple[str, str] | None:
        if self._position >= len(self._tokens):
            return None
        return self._tokens[self._position]

    def _advance(self) -> None:
        self._position += 1

    def _accept_keyword(self, keyword: str) -> bool:
        token = self._peek()
        if token is not None and token[0] == "identifier" and token[1].upper() == keyword:
            self._advance()
            return True
        return False

    def _expect_punct(self, character: str) -> None:
        if self._peek() != ("punct", character):
            raise CheckExpressionError(f"expected {character!r} in CHECK expression")
        self._advance()


def _referenced_columns(node: Any) -> set[str]:
    if isinstance(node, _ColumnRef):
        return {node.name}
    if isinstance(node, _Value):
        return set()
    if isinstance(node, _Comparison):
        return _referenced_columns(node.left) | _referenced_columns(node.right)
    if isinstance(node, _Between):
        return (
            _referenced_columns(node.operand)
            | _referenced_columns(node.low)
            | _referenced_columns(node.high)
        )
    if isinstance(node, _In):
        names = _referenced_columns(node.operand)
        for value in node.values:
            names |= _referenced_columns(value)
        return names
    if isinstance(node, (_IsNull, _Not)):
        return _referenced_columns(node.operand)
    names = set()
    for part in node.parts:
        names |= _referenced_columns(part)
    return names


_TYPE_FAMILIES = {
    "INT": "number",
    "DECIMAL": "number",
    "BOOLEAN": "number",
    "VARCHAR": "text",
    "TEXT": "text",
    "ENUM": "text",
    "DATE": "date",
    "DATETIME": "datetime",
}


def _coerce_literals(node: Any, columns: Mapping[str, ColumnSchema]) -> Any:
    """Rewrite every literal compared with a column into that column's Python type.

    # Errors
    Raises `CheckExpressionError` when a literal cannot represent a value of the
    column's SQL type, or when two compared columns hold incomparable types.
    """
    if isinstance(node, (_And, _Or)):
        return type(node)(tuple(_coerce_literals(part, columns) for part in node.parts))
    if isinstance(node, _Not):
        return _Not(_coerce_literals(node.operand, columns))
    if isinstance(node, _Comparison):
        left, right = _coerce_pair(node.left, node.right, columns)
        return _Comparison(left, node.operator, right)
    if isinstance(node, _Between):
        low, high = _coerce_operands(node.operand, (node.low, node.high), columns)
        return _Between(node.operand, low, high, node.negated)
    if isinstance(node, _In):
        return _In(
            node.operand,
            tuple(_coerce_operands(node.operand, node.values, columns)),
            node.negated,
        )
    return node


def _coerce_pair(
    left: Any, right: Any, columns: Mapping[str, ColumnSchema]
) -> tuple[Any, Any]:
    if isinstance(left, _ColumnRef):
        return left, _coerce_operands(left, (right,), columns)[0]
    if isinstance(right, _ColumnRef):
        return _coerce_operands(right, (left,), columns)[0], right
    return left, right


def _coerce_operands(
    reference: Any, operands: Sequence[Any], columns: Mapping[str, ColumnSchema]
) -> list[Any]:
    column = columns.get(reference.name) if isinstance(reference, _ColumnRef) else None
    if column is None:
        return list(operands)
    coerced: list[Any] = []
    for operand in operands:
        if isinstance(operand, _ColumnRef):
            _require_comparable(column, columns.get(operand.name))
            coerced.append(operand)
        elif isinstance(operand, _Value):
            coerced.append(_Value(_coerce_literal(operand.value, column)))
        else:
            coerced.append(operand)
    return coerced


def _require_comparable(left: ColumnSchema, right: ColumnSchema | None) -> None:
    if right is None:
        return
    if _TYPE_FAMILIES.get(left.type.name) != _TYPE_FAMILIES.get(right.type.name):
        raise CheckExpressionError(
            f"columns {left.name} ({left.type.name}) and {right.name} "
            f"({right.type.name}) hold incomparable types"
        )


def _coerce_literal(value: Any, column: ColumnSchema) -> Any:
    if value is None:
        return None
    coerced = _LITERAL_COERCIONS[_TYPE_FAMILIES.get(column.type.name, "text")](value)
    if coerced is _UNCOERCIBLE:
        raise CheckExpressionError(
            f"literal {value!r} is not a valid {column.type.name} value for "
            f"column {column.name}"
        )
    return coerced


def _as_number(value: Any) -> Any:
    if isinstance(value, bool):
        return Decimal(int(value))
    return value if isinstance(value, Decimal) else _UNCOERCIBLE


def _as_text(value: Any) -> Any:
    return value if isinstance(value, str) else _UNCOERCIBLE


def _as_date(value: Any) -> Any:
    if isinstance(value, datetime):
        return _UNCOERCIBLE
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return _UNCOERCIBLE
    try:
        return date.fromisoformat(value)
    except ValueError:
        return _UNCOERCIBLE


def _as_datetime(value: Any) -> Any:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if not isinstance(value, str):
        return _UNCOERCIBLE
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return _UNCOERCIBLE


_UNCOERCIBLE = object()
_INCOMPARABLE = object()
_LITERAL_COERCIONS = {
    "number": _as_number,
    "text": _as_text,
    "date": _as_date,
    "datetime": _as_datetime,
}


def _is_ordered_literal(value: Any) -> bool:
    return isinstance(value, (Decimal, date))


def _evaluate(node: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(node, _And):
        results = [_evaluate(part, values) for part in node.parts]
        if any(result is False for result in results):
            return False
        if any(result is _INCOMPARABLE for result in results):
            return _INCOMPARABLE
        return None if any(result is None for result in results) else True
    if isinstance(node, _Or):
        results = [_evaluate(part, values) for part in node.parts]
        if any(result is True for result in results):
            return True
        if any(result is _INCOMPARABLE for result in results):
            return _INCOMPARABLE
        return None if any(result is None for result in results) else False
    if isinstance(node, _Not):
        result = _evaluate(node.operand, values)
        return result if result is None or result is _INCOMPARABLE else not result
    if isinstance(node, _IsNull):
        is_null = _resolve(node.operand, values) is None
        return is_null != node.negated
    if isinstance(node, _Comparison):
        return _compare(
            _resolve(node.left, values), node.operator, _resolve(node.right, values)
        )
    if isinstance(node, _Between):
        value = _resolve(node.operand, values)
        low = _resolve(node.low, values)
        high = _resolve(node.high, values)
        if value is None or low is None or high is None:
            return None
        results = [_compare(low, "<=", value), _compare(value, "<=", high)]
        if _INCOMPARABLE in results:
            return _INCOMPARABLE
        inside = all(results)
        return inside if not node.negated else not inside
    if isinstance(node, _In):
        value = _resolve(node.operand, values)
        if value is None:
            return None
        results = [
            _compare(value, "=", _resolve(member, values)) for member in node.values
        ]
        if _INCOMPARABLE in results:
            return _INCOMPARABLE
        found = any(results)
        return found if not node.negated else not found
    return None


def _resolve(operand: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(operand, _ColumnRef):
        return values.get(operand.name)
    if isinstance(operand, _Value):
        return operand.value
    return None


def _compare(left: Any, operator: str, right: Any) -> Any:
    if left is None or right is None:
        return None
    try:
        if operator == "=":
            return left == right
        if operator in {"<>", "!="}:
            return left != right
        if operator == ">":
            return left > right
        if operator == ">=":
            return left >= right
        if operator == "<":
            return left < right
        return left <= right
    except TypeError:
        return _INCOMPARABLE


def _column_comparison_nodes(node: Any) -> list[_Comparison]:
    if isinstance(node, _Comparison):
        both_columns = isinstance(node.left, _ColumnRef) and isinstance(
            node.right, _ColumnRef
        )
        return [node] if both_columns else []
    if isinstance(node, (_And, _Or)):
        return [found for part in node.parts for found in _column_comparison_nodes(part)]
    if isinstance(node, _Not):
        return _column_comparison_nodes(node.operand)
    return []


def _conjuncts(node: Any) -> list[Any]:
    if isinstance(node, _And):
        return [part for member in node.parts for part in _conjuncts(member)]
    return [node]


def _bound_from(node: Any, column_key: str) -> tuple[str, Any] | None:
    if isinstance(node, _Comparison):
        return _comparison_bound(node, column_key)
    if (
        isinstance(node, _Between)
        and not node.negated
        and isinstance(node.operand, _ColumnRef)
        and node.operand.name == column_key
        and isinstance(node.low, _Value)
        and isinstance(node.high, _Value)
    ):
        return ("between", (node.low.value, node.high.value))
    if (
        isinstance(node, _In)
        and not node.negated
        and isinstance(node.operand, _ColumnRef)
        and node.operand.name == column_key
        and all(isinstance(member, _Value) for member in node.values)
    ):
        return ("allowed", tuple(member.value for member in node.values))
    return None


_MIRROR = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "=": "=", "<>": "<>", "!=": "!="}
_BOUND_KIND = {">=": "min", ">": "min_exclusive", "<=": "max", "<": "max_exclusive"}


def _comparison_bound(node: _Comparison, column_key: str) -> tuple[str, Any] | None:
    if (
        isinstance(node.left, _ColumnRef)
        and node.left.name == column_key
        and isinstance(node.right, _Value)
    ):
        operator, value = node.operator, node.right.value
    elif (
        isinstance(node.right, _ColumnRef)
        and node.right.name == column_key
        and isinstance(node.left, _Value)
    ):
        operator, value = _MIRROR.get(node.operator, node.operator), node.left.value
    else:
        return None

    if operator == "=":
        return ("allowed", (value,))
    kind = _BOUND_KIND.get(operator)
    if kind is None or not _is_ordered_literal(value):
        return None
    return (kind, value)
