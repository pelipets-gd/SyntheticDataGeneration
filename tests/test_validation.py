"""Unit tests for independent constraint validation of generated datasets."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pytest

from data_assistant.schema import parse_ddl
from data_assistant.validation import (
    DataValidationError,
    Violation,
    assert_valid_dataset,
    validate_dataset,
)


SCHEMA = parse_ddl(
    """
    CREATE TABLE Customers (
        customer_id INT PRIMARY KEY AUTO_INCREMENT,
        email VARCHAR(30) UNIQUE NOT NULL,
        nickname VARCHAR(10),
        tier ENUM('basic', 'gold') NOT NULL,
        balance DECIMAL(5, 2),
        joined_on DATE,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        active BOOLEAN DEFAULT TRUE
    );
    CREATE TABLE Reviews (
        review_id INT PRIMARY KEY,
        customer_id INT,
        rating INT NOT NULL CHECK (rating >= 1 AND rating <= 5),
        FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
    );
    """
)

COMPOSITE_SCHEMA = parse_ddl(
    """
    CREATE TABLE Memberships (
        tenant_id INT,
        user_id INT,
        email VARCHAR(50),
        label VARCHAR(50),
        PRIMARY KEY (tenant_id, user_id),
        UNIQUE (tenant_id, email)
    );
    """
)

TEMPORAL_SCHEMA = parse_ddl(
    """
    CREATE TABLE Bookings (
        booking_id INT PRIMARY KEY,
        start_date DATE NOT NULL CHECK (start_date >= '2020-01-01'),
        logged_at DATETIME NOT NULL CHECK (logged_at < '2024-01-01 00:00:00')
    );
    """
)


def booking(**overrides: Any) -> dict[str, list[dict[str, Any]]]:
    row: dict[str, Any] = {
        "booking_id": 1,
        "start_date": date(2021, 5, 4),
        "logged_at": datetime(2023, 7, 8, 9, 10),
    }
    row.update(overrides)
    return {"Bookings": [row]}


def customer(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "customer_id": 1,
        "email": "ace@example.com",
        "nickname": "ace",
        "tier": "gold",
        "balance": Decimal("12.50"),
        "joined_on": date(2024, 1, 2),
        "created_at": datetime(2024, 1, 2, 3, 4, 5),
        "active": True,
    }
    row.update(overrides)
    return row


def review(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"review_id": 1, "customer_id": 1, "rating": 4}
    row.update(overrides)
    return row


def dataset(
    customers: list[dict[str, Any]] | None = None,
    reviews: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "Customers": [customer()] if customers is None else customers,
        "Reviews": [review()] if reviews is None else reviews,
    }


def kinds(violations: list[Violation]) -> list[str]:
    return [violation.kind for violation in violations]


def test_accepts_constraint_safe_dataset() -> None:
    assert validate_dataset(SCHEMA, dataset()) == []


def test_reports_null_in_not_null_column_with_table_row_and_column() -> None:
    violations = validate_dataset(SCHEMA, dataset(customers=[customer(tier=None)]))

    assert kinds(violations) == ["not_null"]
    assert violations[0].table == "Customers"
    assert violations[0].column == "tier"
    assert violations[0].row_index == 0
    assert "tier" in violations[0].message


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("customer_id", "1"),
        ("joined_on", datetime(2024, 1, 2, 3, 4, 5)),
        ("created_at", date(2024, 1, 2)),
        ("balance", 12.5),
        ("active", 1),
        ("email", 42),
    ],
)
def test_reports_type_mismatch_per_sql_type(column: str, value: Any) -> None:
    violations = validate_dataset(
        SCHEMA, dataset(customers=[customer(**{column: value})], reviews=[])
    )

    assert kinds(violations) == ["type_mismatch"]
    assert violations[0].column == column


def test_reports_value_longer_than_varchar_limit() -> None:
    violations = validate_dataset(SCHEMA, dataset(customers=[customer(nickname="x" * 11)]))

    assert kinds(violations) == ["varchar_length"]
    assert "10" in violations[0].message


@pytest.mark.parametrize("value", [Decimal("1234.56"), Decimal("1.234")])
def test_reports_decimal_outside_declared_precision_or_scale(value: Decimal) -> None:
    violations = validate_dataset(SCHEMA, dataset(customers=[customer(balance=value)]))

    assert kinds(violations) == ["decimal_precision"]


def test_reports_value_outside_enum_members() -> None:
    violations = validate_dataset(SCHEMA, dataset(customers=[customer(tier="platinum")]))

    assert kinds(violations) == ["enum_value"]
    assert "basic" in violations[0].message


def test_reports_duplicate_primary_key_values() -> None:
    violations = validate_dataset(
        SCHEMA, dataset(customers=[customer(), customer(email="b@example.com")])
    )

    assert kinds(violations) == ["primary_key_duplicate"]
    assert violations[0].row_index == 1


def test_reports_null_primary_key_value() -> None:
    violations = validate_dataset(SCHEMA, dataset(customers=[customer(customer_id=None)]))

    assert "primary_key_null" in kinds(violations)


def test_composite_primary_key_is_unique_per_tuple() -> None:
    rows = [
        {"tenant_id": 1, "user_id": 1, "email": "a@x.io", "label": "one"},
        {"tenant_id": 1, "user_id": 2, "email": "b@x.io", "label": "two"},
        {"tenant_id": 1, "user_id": 1, "email": "c@x.io", "label": "three"},
    ]

    violations = validate_dataset(COMPOSITE_SCHEMA, {"Memberships": rows})

    assert kinds(violations) == ["primary_key_duplicate"]
    assert violations[0].row_index == 2


def test_unique_constraint_ignores_null_tuples_but_reports_duplicates() -> None:
    rows = [
        {"tenant_id": 1, "user_id": 1, "email": None, "label": "one"},
        {"tenant_id": 1, "user_id": 2, "email": None, "label": "two"},
        {"tenant_id": 1, "user_id": 3, "email": "dup@x.io", "label": "three"},
        {"tenant_id": 1, "user_id": 4, "email": "dup@x.io", "label": "four"},
    ]

    violations = validate_dataset(COMPOSITE_SCHEMA, {"Memberships": rows})

    assert kinds(violations) == ["unique_duplicate"]
    assert violations[0].row_index == 3


def test_reports_foreign_key_without_matching_parent_row() -> None:
    violations = validate_dataset(SCHEMA, dataset(reviews=[review(customer_id=99)]))

    assert kinds(violations) == ["foreign_key_missing"]
    assert "Customers" in violations[0].message


def test_accepts_null_foreign_key_value() -> None:
    assert validate_dataset(SCHEMA, dataset(reviews=[review(customer_id=None)])) == []


def test_reports_row_failing_a_check_expression() -> None:
    violations = validate_dataset(SCHEMA, dataset(reviews=[review(rating=9)]))

    assert kinds(violations) == ["check_failed"]
    assert violations[0].row_index == 0
    assert "rating" in violations[0].message


def test_check_with_null_operand_passes_sql_null_semantics() -> None:
    schema = parse_ddl(
        "CREATE TABLE Scores (id INT PRIMARY KEY, score INT CHECK (score >= 1));"
    )

    assert validate_dataset(schema, {"Scores": [{"id": 1, "score": None}]}) == []


def test_reports_unsupported_check_form_instead_of_treating_it_as_valid() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Scores (
            id INT PRIMARY KEY,
            score INT NOT NULL,
            bonus INT NOT NULL,
            CHECK ((score + bonus) >= 0)
        );
        """
    )

    violations = validate_dataset(schema, {"Scores": [{"id": 1, "score": 1, "bonus": 2}]})

    assert kinds(violations) == ["unsupported_check"]
    assert violations[0].row_index is None


def test_accepts_rows_inside_date_and_datetime_check_literals() -> None:
    assert validate_dataset(TEMPORAL_SCHEMA, booking()) == []


@pytest.mark.parametrize(
    ("column", "value"),
    [("start_date", date(2019, 12, 31)), ("logged_at", datetime(2024, 6, 1))],
)
def test_reports_row_outside_a_date_or_datetime_check_literal(
    column: str, value: Any
) -> None:
    violations = validate_dataset(TEMPORAL_SCHEMA, booking(**{column: value}))

    assert kinds(violations) == ["check_failed"]
    assert violations[0].row_index == 0


@pytest.mark.parametrize(
    "expression", ["start_date >= 'yesterday'", "start_date >= 20200101"]
)
def test_reports_check_literal_that_is_not_a_date_as_unsupported(
    expression: str,
) -> None:
    schema = parse_ddl(
        f"CREATE TABLE Trips (id INT PRIMARY KEY, start_date DATE, CHECK ({expression}));"
    )

    violations = validate_dataset(schema, {"Trips": [{"id": 1, "start_date": date(2021, 1, 1)}]})

    assert kinds(violations) == ["unsupported_check"]
    assert violations[0].row_index is None


def test_reports_comparison_between_columns_of_different_types_as_unsupported() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Spans (
            id INT PRIMARY KEY,
            on_day DATE,
            at_moment DATETIME,
            CHECK (at_moment >= on_day)
        );
        """
    )
    row = {"id": 1, "on_day": date(2024, 1, 1), "at_moment": datetime(2024, 1, 2)}

    assert kinds(validate_dataset(schema, {"Spans": [row]})) == ["unsupported_check"]


def test_reports_unevaluable_check_instead_of_passing_a_wrongly_typed_value() -> None:
    violations = validate_dataset(TEMPORAL_SCHEMA, booking(start_date="2019-01-01"))

    assert sorted(kinds(violations)) == ["check_unevaluable", "type_mismatch"]
    assert next(
        violation for violation in violations if violation.kind == "check_unevaluable"
    ).row_index == 0


def test_supports_date_literals_in_between_and_in_check_forms() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Shifts (
            id INT PRIMARY KEY,
            shift_date DATE CHECK (shift_date BETWEEN '2024-01-01' AND '2024-01-31'),
            review_at DATETIME CHECK (review_at IN ('2024-02-01 08:00:00', '2024-02-02 08:00:00'))
        );
        """
    )
    inside = {"id": 1, "shift_date": date(2024, 1, 15), "review_at": datetime(2024, 2, 1, 8)}
    outside = {"id": 2, "shift_date": date(2024, 2, 15), "review_at": datetime(2024, 2, 3, 8)}

    assert validate_dataset(schema, {"Shifts": [inside]}) == []
    assert kinds(validate_dataset(schema, {"Shifts": [outside]})) == [
        "check_failed",
        "check_failed",
    ]


def test_supports_between_in_and_is_not_null_check_forms() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Items (
            id INT PRIMARY KEY,
            size INT CHECK (size BETWEEN 1 AND 3),
            grade VARCHAR(2) CHECK (grade IN ('A', 'B')),
            code VARCHAR(4),
            CHECK (code IS NOT NULL)
        );
        """
    )
    valid = {"id": 1, "size": 2, "grade": "A", "code": "ok"}
    invalid = {"id": 2, "size": 7, "grade": "C", "code": None}

    assert validate_dataset(schema, {"Items": [valid]}) == []
    assert kinds(validate_dataset(schema, {"Items": [invalid]})) == [
        "check_failed",
        "check_failed",
        "check_failed",
    ]


@pytest.mark.parametrize(
    ("dropped", "expected"),
    [("nickname", []), ("created_at", []), ("tier", ["missing_column"])],
)
def test_missing_column_reported_only_without_null_or_default_fallback(
    dropped: str, expected: list[str]
) -> None:
    row = customer()
    del row[dropped]

    assert kinds(validate_dataset(SCHEMA, dataset(customers=[row]))) == expected


def test_reports_unknown_table_and_unknown_column() -> None:
    data = dataset()
    data["Ghosts"] = [{"id": 1}]
    data["Customers"] = [customer(surprise="x")]

    assert sorted(kinds(validate_dataset(SCHEMA, data))) == ["unknown_column", "unknown_table"]


def test_reports_table_missing_from_dataset() -> None:
    violations = validate_dataset(SCHEMA, {"Customers": [customer()]})

    assert kinds(violations) == ["missing_table"]
    assert violations[0].table == "Reviews"


def test_reports_row_that_is_not_a_mapping() -> None:
    violations = validate_dataset(SCHEMA, dataset(customers=[["not", "a", "row"]], reviews=[]))

    assert kinds(violations) == ["row_shape"]


def test_assert_valid_dataset_raises_structured_error_with_every_violation() -> None:
    data = dataset(customers=[customer(tier=None, nickname="y" * 20)])

    with pytest.raises(DataValidationError) as error:
        assert_valid_dataset(SCHEMA, data)

    assert sorted(kinds(error.value.violations)) == ["not_null", "varchar_length"]
    assert "Customers" in str(error.value)
