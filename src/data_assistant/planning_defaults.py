"""Local generation plans used when Gemini is unavailable or as a fallback."""

from __future__ import annotations

from data_assistant.gemini_planning import (
    ColumnGenerationPlan,
    DatabaseGenerationPlan,
    TableGenerationPlan,
)
from data_assistant.schema import DatabaseSchema, TableSchema


def default_generation_plan(schema: DatabaseSchema, row_count: int) -> DatabaseGenerationPlan:
    """Cover every non-key column of every table with an `auto` generator."""
    return DatabaseGenerationPlan(
        database_description="Constraint-safe synthetic data",
        tables=[
            TableGenerationPlan(
                table_name=table.name,
                row_count=row_count,
                columns=[
                    ColumnGenerationPlan(column_name=column.name, generator="auto")
                    for column in table.columns
                    if not _is_key_column(table, column.name)
                ],
            )
            for table in schema.tables
        ],
    )


def _is_key_column(table: TableSchema, column_name: str) -> bool:
    key = column_name.casefold()
    if table.primary_key is not None and any(
        name.casefold() == key for name in table.primary_key.columns
    ):
        return True
    return any(
        any(name.casefold() == key for name in foreign_key.columns)
        for foreign_key in table.foreign_keys
    )
