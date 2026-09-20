"""CSV and ZIP export of generated datasets."""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from data_assistant.schema import DatabaseSchema


def table_csv(schema: DatabaseSchema, table_name: str, rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize one table to UTF-8 CSV bytes using schema column order."""
    table = schema.table(table_name)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    names = [column.name for column in table.columns]
    writer.writerow(names)
    for row in rows:
        writer.writerow([_csv_value(row.get(name)) for name in names])
    return buffer.getvalue().encode("utf-8")


def dataset_zip(
    schema: DatabaseSchema, data: Mapping[str, Sequence[Mapping[str, Any]]]
) -> bytes:
    """Return a ZIP archive with one CSV file per table."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for table in schema.tables:
            archive.writestr(
                f"{table.name}.csv",
                table_csv(schema, table.name, data.get(table.name, ())),
            )
    return buffer.getvalue()


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)
