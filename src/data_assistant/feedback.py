"""Apply validated Gemini feedback operations to a generated dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from data_assistant.gemini_planning import (
    FeedbackOperation,
    RegenerateColumnsOperation,
    ReplaceCategoricalValuesOperation,
    TransformValuesOperation,
)
from data_assistant.generation import GeneratedDataset, SyntheticDataGenerator
from data_assistant.schema import DatabaseSchema, TableSchema
from data_assistant.validation import DataValidationError, assert_valid_dataset


class FeedbackError(ValueError):
    """Raised when a feedback operation cannot be applied safely."""


def apply_feedback_operations(
    schema: DatabaseSchema,
    data: Mapping[str, Sequence[Mapping[str, Any]]],
    operations: Sequence[FeedbackOperation],
    *,
    seed: int = 1,
) -> GeneratedDataset:
    """Return a new dataset with operations applied and re-validated.

    Primary-key values are copied unchanged. Foreign-key columns are not
    rewritten by regenerate/replace/transform operations (those tools already
    reject key columns).
    """
    updated: GeneratedDataset = {
        name: [dict(row) for row in rows] for name, rows in data.items()
    }
    for operation in operations:
        table = schema.table(operation.table_name)
        rows = updated[table.name]
        if isinstance(operation, RegenerateColumnsOperation):
            _regenerate(schema, table, rows, operation, seed=seed)
        elif isinstance(operation, TransformValuesOperation):
            _transform(table, rows, operation)
        elif isinstance(operation, ReplaceCategoricalValuesOperation):
            _replace(table, rows, operation)
        else:
            raise FeedbackError(f"Unsupported operation {type(operation)!r}")
    try:
        assert_valid_dataset(schema, updated)
    except DataValidationError as error:
        raise FeedbackError(str(error)) from error
    return updated


def _regenerate(
    schema: DatabaseSchema,
    table: TableSchema,
    rows: list[dict[str, Any]],
    operation: RegenerateColumnsOperation,
    *,
    seed: int,
) -> None:
    replacement = SyntheticDataGenerator().generate(
        schema, row_count=max(len(rows), 1), seed=seed
    )[table.name]
    columns = operation.columns
    for index, row in enumerate(rows):
        source = replacement[index % len(replacement)]
        for column in columns:
            row[column] = source[column]


def _transform(
    table: TableSchema, rows: list[dict[str, Any]], operation: TransformValuesOperation
) -> None:
    column = table.column(operation.column)
    for row in rows:
        current = row[column.name]
        if current is None:
            continue
        row[column.name] = _transform_value(current, operation)


def _transform_value(current: Any, operation: TransformValuesOperation) -> Any:
    if operation.transformation == "add":
        return current + _as_same_numeric(current, operation.value)
    if operation.transformation == "multiply":
        product = current * _as_same_numeric(current, operation.value)
        if isinstance(current, Decimal):
            return Decimal(product)
        return product
    if operation.transformation == "clamp":
        low = _as_same_numeric(current, operation.minimum)
        high = _as_same_numeric(current, operation.maximum)
        return min(max(current, low), high)
    if operation.transformation == "shift_days":
        delta = timedelta(days=int(operation.value or 0))
        if isinstance(current, datetime):
            return current + delta
        if isinstance(current, date):
            return current + delta
        raise FeedbackError(f"shift_days cannot be applied to {type(current).__name__}")
    raise FeedbackError(f"Unsupported transformation {operation.transformation}")


def _as_same_numeric(current: Any, value: float | None) -> Any:
    if value is None:
        raise FeedbackError("numeric transformation is missing a value")
    if isinstance(current, Decimal):
        return Decimal(str(value))
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value) if float(value).is_integer() else value
    if isinstance(current, float):
        return float(value)
    raise FeedbackError(f"cannot apply a numeric transform to {type(current).__name__}")


def _replace(
    table: TableSchema,
    rows: list[dict[str, Any]],
    operation: ReplaceCategoricalValuesOperation,
) -> None:
    column = table.column(operation.column)
    mapping = operation.replacements
    for row in rows:
        current = row[column.name]
        if isinstance(current, str) and current in mapping:
            row[column.name] = mapping[current]
