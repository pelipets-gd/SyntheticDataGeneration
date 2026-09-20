"""Tests for applying validated feedback operations."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from data_assistant.feedback import FeedbackError, apply_feedback_operations
from data_assistant.gemini_planning import (
    RegenerateColumnsOperation,
    ReplaceCategoricalValuesOperation,
    TransformValuesOperation,
)
from data_assistant.schema import parse_ddl


SCHEMA = parse_ddl(
    """
    CREATE TABLE Item (
        item_id INT PRIMARY KEY,
        label VARCHAR(40) NOT NULL,
        price DECIMAL(6, 2) NOT NULL,
        status ENUM('open', 'closed') NOT NULL,
        due DATE
    );
    """
)

ROWS = {
    "Item": [
        {
            "item_id": 1,
            "label": "alpha",
            "price": Decimal("10.00"),
            "status": "open",
            "due": date(2025, 1, 1),
        },
        {
            "item_id": 2,
            "label": "beta",
            "price": Decimal("20.00"),
            "status": "closed",
            "due": date(2025, 2, 1),
        },
    ]
}


def test_replace_categorical_values_keeps_primary_keys() -> None:
    updated = apply_feedback_operations(
        SCHEMA,
        ROWS,
        [
            ReplaceCategoricalValuesOperation(
                table_name="Item",
                column="status",
                replacements={"open": "closed"},
            )
        ],
    )

    assert [row["item_id"] for row in updated["Item"]] == [1, 2]
    assert updated["Item"][0]["status"] == "closed"
    assert updated["Item"][1]["status"] == "closed"


def test_transform_add_and_shift_days() -> None:
    updated = apply_feedback_operations(
        SCHEMA,
        ROWS,
        [
            TransformValuesOperation(
                table_name="Item", column="price", transformation="add", value=1.5
            ),
            TransformValuesOperation(
                table_name="Item", column="due", transformation="shift_days", value=1
            ),
        ],
    )

    assert updated["Item"][0]["price"] == Decimal("11.50")
    assert updated["Item"][0]["due"] == date(2025, 1, 2)


def test_regenerate_columns_changes_requested_values_only() -> None:
    updated = apply_feedback_operations(
        SCHEMA,
        ROWS,
        [RegenerateColumnsOperation(table_name="Item", columns=["label"])],
        seed=9,
    )

    assert [row["item_id"] for row in updated["Item"]] == [1, 2]
    assert [row["price"] for row in updated["Item"]] == [Decimal("10.00"), Decimal("20.00")]
    assert any(row["label"] != original["label"] for row, original in zip(updated["Item"], ROWS["Item"]))


def test_invalid_transform_is_rejected_before_returning() -> None:
    with pytest.raises(FeedbackError):
        apply_feedback_operations(
            SCHEMA,
            ROWS,
            [
                TransformValuesOperation(
                    table_name="Item",
                    column="price",
                    transformation="add",
                    value=1_000_000,
                )
            ],
        )
