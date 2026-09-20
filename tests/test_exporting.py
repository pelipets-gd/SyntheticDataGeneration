"""Tests for CSV and ZIP export."""

from __future__ import annotations

import zipfile
from datetime import date
from decimal import Decimal
from io import BytesIO

from data_assistant.exporting import dataset_zip, table_csv
from data_assistant.schema import parse_ddl


SCHEMA = parse_ddl(
    """
    CREATE TABLE Item (
        item_id INT PRIMARY KEY,
        label VARCHAR(40) NOT NULL,
        price DECIMAL(6, 2) NOT NULL,
        due DATE
    );
    """
)
ROWS = {
    "Item": [
        {"item_id": 1, "label": "alpha", "price": Decimal("10.50"), "due": date(2025, 1, 2)}
    ]
}


def test_table_csv_uses_schema_column_order() -> None:
    text = table_csv(SCHEMA, "Item", ROWS["Item"]).decode("utf-8")

    assert text.splitlines()[0] == "item_id,label,price,due"
    assert "10.50" in text
    assert "2025-01-02" in text


def test_dataset_zip_contains_one_csv_per_table() -> None:
    payload = dataset_zip(SCHEMA, ROWS)

    with zipfile.ZipFile(BytesIO(payload)) as archive:
        assert archive.namelist() == ["Item.csv"]
        assert b"alpha" in archive.read("Item.csv")
