"""Unit tests for deterministic constraint-safe synthetic data generation."""

from __future__ import annotations

import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from faker.providers.person.en_US import Provider as PersonProvider

from data_assistant.gemini_planning import (
    ColumnGenerationPlan,
    DatabaseGenerationPlan,
    TableGenerationPlan,
)
from data_assistant.generation import GenerationError, SyntheticDataGenerator
from data_assistant.schema import DatabaseSchema, TableSchema, parse_ddl
from data_assistant.validation import validate_dataset


PERSON_WORDS = frozenset(PersonProvider.first_names) | frozenset(PersonProvider.last_names)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_NAMES = (
    "company_employee_schema.ddl",
    "restaurants_schema.ddl",
    "library_management_schema.ddl",
)


def fixture_schema(name: str) -> DatabaseSchema:
    return parse_ddl((FIXTURES / name).read_text())


def is_key_column(table: TableSchema, column_name: str) -> bool:
    key = column_name.casefold()
    primary = table.primary_key.columns if table.primary_key else []
    foreign = [column for foreign_key in table.foreign_keys for column in foreign_key.columns]
    return key in {name.casefold() for name in [*primary, *foreign]}


def plan_for(
    schema: DatabaseSchema,
    row_count: int,
    hints: dict[tuple[str, str], ColumnGenerationPlan] | None = None,
) -> DatabaseGenerationPlan:
    hints = hints or {}
    return DatabaseGenerationPlan(
        database_description="test plan",
        tables=[
            TableGenerationPlan(
                table_name=table.name,
                row_count=row_count,
                columns=[
                    hints.get(
                        (table.name, column.name),
                        ColumnGenerationPlan(column_name=column.name, generator="auto"),
                    )
                    for column in table.columns
                    if not is_key_column(table, column.name)
                ],
            )
            for table in schema.tables
        ],
    ).validate_against(schema)


def column_values(
    data: dict[str, list[dict[str, Any]]], table: str, column: str
) -> list[Any]:
    return [row[column] for row in data[table]]


def looks_like_person(value: str) -> bool:
    """Report whether every word of `value` comes from Faker's person vocabulary."""
    words = value.split()
    return bool(words) and all(word in PERSON_WORDS for word in words)


def test_generates_requested_row_count_for_every_table() -> None:
    schema = fixture_schema("restaurants_schema.ddl")

    data = SyntheticDataGenerator().generate(schema, row_count=6, seed=1)

    assert sorted(data) == sorted(table.name for table in schema.tables)
    assert {name: len(rows) for name, rows in data.items()} == {
        table.name: 6 for table in schema.tables
    }


@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
def test_generated_data_passes_independent_validation(fixture: str) -> None:
    schema = fixture_schema(fixture)

    data = SyntheticDataGenerator().generate(schema, row_count=8, seed=3)

    assert validate_dataset(schema, data) == []


@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
def test_generated_data_from_a_plan_passes_validation(fixture: str) -> None:
    schema = fixture_schema(fixture)
    plan = plan_for(schema, row_count=5)

    data = SyntheticDataGenerator().generate(schema, plan, seed=3)

    assert validate_dataset(schema, data) == []
    assert all(len(rows) == 5 for rows in data.values())


def test_same_seed_reproduces_identical_dataset() -> None:
    schema = fixture_schema("restaurants_schema.ddl")
    generator = SyntheticDataGenerator()

    first = generator.generate(schema, row_count=7, seed=42)
    second = SyntheticDataGenerator().generate(schema, row_count=7, seed=42)

    assert first == second


def test_different_seed_changes_generated_values() -> None:
    schema = fixture_schema("restaurants_schema.ddl")
    generator = SyntheticDataGenerator()

    first = generator.generate(schema, row_count=7, seed=1)
    second = generator.generate(schema, row_count=7, seed=2)

    assert column_values(first, "Customers", "email") != column_values(
        second, "Customers", "email"
    )


def test_auto_increment_primary_keys_are_sequential_from_one() -> None:
    schema = fixture_schema("restaurants_schema.ddl")

    data = SyntheticDataGenerator().generate(schema, row_count=5, seed=1)

    assert column_values(data, "Customers", "customer_id") == [1, 2, 3, 4, 5]


def test_foreign_keys_only_reference_generated_parent_rows() -> None:
    schema = fixture_schema("restaurants_schema.ddl")

    data = SyntheticDataGenerator().generate(schema, row_count=9, seed=5)

    customer_ids = set(column_values(data, "Customers", "customer_id"))
    assert set(column_values(data, "Orders", "customer_id")) <= customer_ids


def test_cyclic_relationships_generate_without_broken_references() -> None:
    schema = fixture_schema("library_management_schema.ddl")

    data = SyntheticDataGenerator().generate(schema, row_count=6, seed=11)

    employee_ids = set(column_values(data, "Employees", "employee_id"))
    department_ids = set(column_values(data, "Departments", "department_id"))
    managers = set(column_values(data, "Departments", "manager_id")) - {None}
    departments = set(column_values(data, "Employees", "department_id")) - {None}

    assert managers and managers <= employee_ids
    assert departments and departments <= department_ids


def test_self_referencing_foreign_keys_use_existing_keys() -> None:
    schema = fixture_schema("company_employee_schema.ddl")

    data = SyntheticDataGenerator().generate(schema, row_count=6, seed=13)

    employee_ids = set(column_values(data, "Employees", "employee_id"))
    assert set(column_values(data, "Performance_Reviews", "reviewer_id")) <= employee_ids


def test_check_constrained_column_stays_within_range() -> None:
    schema = fixture_schema("restaurants_schema.ddl")

    data = SyntheticDataGenerator().generate(schema, row_count=50, seed=7)

    assert all(1 <= rating <= 5 for rating in column_values(data, "Reviews", "rating"))


def test_unique_columns_have_no_duplicate_values() -> None:
    schema = fixture_schema("restaurants_schema.ddl")

    data = SyntheticDataGenerator().generate(schema, row_count=200, seed=9)

    emails = column_values(data, "Customers", "email")
    licenses = column_values(data, "Delivery_Drivers", "license_number")
    assert len(set(emails)) == len(emails)
    assert len(set(licenses)) == len(licenses)


def test_varchar_values_never_exceed_declared_length() -> None:
    schema = fixture_schema("restaurants_schema.ddl")

    data = SyntheticDataGenerator().generate(schema, row_count=40, seed=4)

    assert all(len(value) <= 20 for value in column_values(data, "Customers", "phone_number"))
    assert all(len(value) <= 10 for value in column_values(data, "Customers", "zip_code"))


def test_decimal_values_respect_precision_and_scale() -> None:
    schema = parse_ddl(
        "CREATE TABLE Prices (id INT PRIMARY KEY, amount DECIMAL(6, 3) NOT NULL);"
    )

    data = SyntheticDataGenerator().generate(schema, row_count=30, seed=2)

    amounts = column_values(data, "Prices", "amount")
    assert all(isinstance(amount, Decimal) for amount in amounts)
    assert all(-amount.as_tuple().exponent <= 3 for amount in amounts)
    assert all(abs(amount) < Decimal(1000) for amount in amounts)


def test_unbounded_decimals_stay_within_a_plausible_magnitude() -> None:
    schema = parse_ddl(
        "CREATE TABLE Menu (id INT PRIMARY KEY, price DECIMAL(10, 2) NOT NULL);"
    )

    data = SyntheticDataGenerator().generate(schema, row_count=40, seed=6)

    assert all(
        Decimal(0) <= price <= Decimal(10_000)
        for price in column_values(data, "Menu", "price")
    )


def test_plan_bounds_beyond_the_default_magnitude_are_honoured() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Salaries (
            id INT PRIMARY KEY,
            annual_amount DECIMAL(12, 2) NOT NULL,
            headcount INT NOT NULL
        );
        """
    )
    plan = plan_for(
        schema,
        row_count=20,
        hints={
            ("Salaries", "annual_amount"): ColumnGenerationPlan(
                column_name="annual_amount",
                generator="auto",
                minimum=50_000,
                maximum=250_000,
            ),
            ("Salaries", "headcount"): ColumnGenerationPlan(
                column_name="headcount",
                generator="auto",
                minimum=20_000,
                maximum=30_000,
            ),
        },
    )

    data = SyntheticDataGenerator().generate(schema, plan, seed=6)

    assert all(
        Decimal(50_000) <= amount <= Decimal(250_000)
        for amount in column_values(data, "Salaries", "annual_amount")
    )
    assert all(
        20_000 <= headcount <= 30_000
        for headcount in column_values(data, "Salaries", "headcount")
    )


def test_person_names_are_limited_to_tables_describing_people() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Members (member_id INT PRIMARY KEY, name VARCHAR(60) NOT NULL);
        CREATE TABLE Branches (branch_id INT PRIMARY KEY, name VARCHAR(60) NOT NULL);
        CREATE TABLE Menu (menu_id INT PRIMARY KEY, item_name VARCHAR(60) NOT NULL);
        """
    )

    data = SyntheticDataGenerator().generate(schema, row_count=12, seed=7)

    assert all(looks_like_person(name) for name in column_values(data, "Members", "name"))
    assert not any(
        looks_like_person(name) for name in column_values(data, "Branches", "name")
    )
    assert not any(
        looks_like_person(name) for name in column_values(data, "Menu", "item_name")
    )


def test_current_timestamp_default_column_receives_datetime_values() -> None:
    schema = fixture_schema("restaurants_schema.ddl")

    data = SyntheticDataGenerator().generate(schema, row_count=4, seed=6)

    assert all(
        isinstance(value, datetime)
        for value in column_values(data, "Customers", "registration_date")
    )


def test_composite_primary_key_built_from_foreign_keys_stays_unique() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Students (student_id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(100) NOT NULL);
        CREATE TABLE Courses (course_id INT PRIMARY KEY AUTO_INCREMENT,
            title VARCHAR(100) NOT NULL);
        CREATE TABLE Enrollments (
            student_id INT NOT NULL,
            course_id INT NOT NULL,
            grade VARCHAR(2),
            PRIMARY KEY (student_id, course_id),
            FOREIGN KEY (student_id) REFERENCES Students(student_id),
            FOREIGN KEY (course_id) REFERENCES Courses(course_id)
        );
        """
    )

    data = SyntheticDataGenerator().generate(schema, row_count=6, seed=8)

    pairs = [(row["student_id"], row["course_id"]) for row in data["Enrollments"]]
    assert len(set(pairs)) == 6
    assert validate_dataset(schema, data) == []


def test_plan_semantic_generators_drive_generated_values() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Contacts (
            id INT PRIMARY KEY,
            contact_email VARCHAR(255) NOT NULL,
            home_city VARCHAR(100) NOT NULL,
            site VARCHAR(255) NOT NULL,
            employer VARCHAR(255) NOT NULL,
            bio TEXT NOT NULL,
            age INT NOT NULL,
            subscribed BOOLEAN NOT NULL,
            plan_kind ENUM('free', 'paid') NOT NULL
        );
        """
    )
    hints = {
        ("Contacts", name): ColumnGenerationPlan(column_name=name, generator=generator)
        for name, generator in {
            "contact_email": "email",
            "home_city": "city",
            "site": "url",
            "employer": "company",
            "bio": "text",
            "subscribed": "boolean",
            "plan_kind": "enum",
        }.items()
    }
    hints[("Contacts", "age")] = ColumnGenerationPlan(
        column_name="age", generator="integer", minimum=21, maximum=30
    )

    data = SyntheticDataGenerator().generate(
        schema, plan_for(schema, row_count=20, hints=hints), seed=1
    )

    assert all("@" in value for value in column_values(data, "Contacts", "contact_email"))
    assert all(
        value.startswith("http") for value in column_values(data, "Contacts", "site")
    )
    assert all(21 <= value <= 30 for value in column_values(data, "Contacts", "age"))
    assert all(
        isinstance(value, bool) for value in column_values(data, "Contacts", "subscribed")
    )
    assert set(column_values(data, "Contacts", "plan_kind")) <= {"free", "paid"}
    assert all(len(value) > 10 for value in column_values(data, "Contacts", "bio"))
    assert all(value for value in column_values(data, "Contacts", "employer"))
    assert len(set(column_values(data, "Contacts", "home_city"))) > 1


def test_plan_nullable_probability_applies_only_to_nullable_columns() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Notes (
            id INT PRIMARY KEY,
            title VARCHAR(50) NOT NULL,
            body TEXT
        );
        """
    )
    hints = {
        ("Notes", "body"): ColumnGenerationPlan(
            column_name="body", generator="text", nullable_probability=1.0
        )
    }

    data = SyntheticDataGenerator().generate(
        schema, plan_for(schema, row_count=12, hints=hints), seed=1
    )

    assert column_values(data, "Notes", "body") == [None] * 12
    assert all(value is not None for value in column_values(data, "Notes", "title"))


def test_plan_row_count_is_used_per_table() -> None:
    schema = fixture_schema("restaurants_schema.ddl")
    plan = plan_for(schema, row_count=4)
    plan.tables[0].row_count = 9

    data = SyntheticDataGenerator().generate(schema, plan, seed=1)

    assert len(data[plan.tables[0].table_name]) == 9
    assert len(data[plan.tables[1].table_name]) == 4


def test_explicit_row_count_overrides_plan_and_default() -> None:
    schema = fixture_schema("restaurants_schema.ddl")
    plan = plan_for(schema, row_count=7)

    data = SyntheticDataGenerator(default_row_count=3).generate(
        schema, plan, row_count=5, seed=1
    )

    assert all(len(rows) == 5 for rows in data.values())


def test_default_row_count_applies_without_plan_or_override() -> None:
    schema = fixture_schema("restaurants_schema.ddl")

    data = SyntheticDataGenerator(default_row_count=3).generate(schema, seed=1)

    assert all(len(rows) == 3 for rows in data.values())


def test_rejects_row_count_above_configured_maximum() -> None:
    schema = fixture_schema("restaurants_schema.ddl")

    with pytest.raises(GenerationError, match="maximum"):
        SyntheticDataGenerator(max_row_count=10).generate(schema, row_count=11, seed=1)


def test_surfaces_unsupported_check_expression_instead_of_emitting_data() -> None:
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

    with pytest.raises(GenerationError, match="CHECK .* is not safely supported"):
        SyntheticDataGenerator().generate(schema, row_count=3, seed=1)


def test_composite_key_of_foreign_key_and_discriminator_exceeds_parent_rows() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Devices (
            device_id INT PRIMARY KEY AUTO_INCREMENT,
            label VARCHAR(50) NOT NULL
        );
        CREATE TABLE Readings (
            device_id INT NOT NULL,
            sequence_no INT NOT NULL,
            celsius DECIMAL(5, 2) NOT NULL,
            PRIMARY KEY (device_id, sequence_no),
            FOREIGN KEY (device_id) REFERENCES Devices(device_id)
        );
        """
    )
    plan = plan_for(schema, row_count=4)
    next(table for table in plan.tables if table.table_name == "Readings").row_count = 40

    data = SyntheticDataGenerator().generate(schema, plan, seed=3)

    keys = [(row["device_id"], row["sequence_no"]) for row in data["Readings"]]
    assert len(set(keys)) == 40
    assert {device for device, _ in keys} <= set(column_values(data, "Devices", "device_id"))
    assert validate_dataset(schema, data) == []


def test_resolves_foreign_key_chain_through_non_primary_unique_columns() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Leaves (
            leaf_id INT PRIMARY KEY AUTO_INCREMENT,
            code VARCHAR(12) NOT NULL,
            FOREIGN KEY (code) REFERENCES Middles(code)
        );
        CREATE TABLE Middles (
            middle_id INT PRIMARY KEY AUTO_INCREMENT,
            code VARCHAR(12) NOT NULL UNIQUE,
            FOREIGN KEY (code) REFERENCES Roots(code)
        );
        CREATE TABLE Roots (
            root_id INT PRIMARY KEY AUTO_INCREMENT,
            code VARCHAR(12) NOT NULL UNIQUE
        );
        """
    )

    data = SyntheticDataGenerator().generate(schema, row_count=6, seed=5)

    assert set(column_values(data, "Middles", "code")) <= set(
        column_values(data, "Roots", "code")
    )
    assert set(column_values(data, "Leaves", "code")) <= set(
        column_values(data, "Middles", "code")
    )
    assert validate_dataset(schema, data) == []


def test_reports_foreign_key_cycle_that_cannot_be_materialized() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Alpha (
            alpha_id INT PRIMARY KEY AUTO_INCREMENT,
            code VARCHAR(10) NOT NULL UNIQUE,
            FOREIGN KEY (code) REFERENCES Beta(code)
        );
        CREATE TABLE Beta (
            beta_id INT PRIMARY KEY AUTO_INCREMENT,
            code VARCHAR(10) NOT NULL UNIQUE,
            FOREIGN KEY (code) REFERENCES Alpha(code)
        );
        """
    )

    with pytest.raises(GenerationError, match="cycle"):
        SyntheticDataGenerator().generate(schema, row_count=4, seed=1)


def test_satisfies_column_to_column_date_and_numeric_checks() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Bookings (
            booking_id INT PRIMARY KEY AUTO_INCREMENT,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            min_spend DECIMAL(8, 2) NOT NULL,
            max_spend DECIMAL(8, 2) NOT NULL,
            CHECK (end_date >= start_date),
            CHECK (max_spend > min_spend)
        );
        """
    )

    data = SyntheticDataGenerator().generate(schema, row_count=40, seed=4)

    assert all(row["end_date"] >= row["start_date"] for row in data["Bookings"])
    assert all(row["max_spend"] > row["min_spend"] for row in data["Bookings"])
    assert validate_dataset(schema, data) == []


def test_generates_dates_and_datetimes_inside_check_literal_bounds() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Trips (
            trip_id INT PRIMARY KEY AUTO_INCREMENT,
            travel_date DATE NOT NULL
                CHECK (travel_date >= '2024-03-01' AND travel_date <= '2024-03-31'),
            booked_at DATETIME NOT NULL
                CHECK (booked_at BETWEEN '2024-01-01 00:00:00' AND '2024-01-31 23:59:59')
        );
        """
    )

    data = SyntheticDataGenerator().generate(schema, row_count=30, seed=3)

    assert all(
        date(2024, 3, 1) <= value <= date(2024, 3, 31)
        for value in column_values(data, "Trips", "travel_date")
    )
    assert all(
        datetime(2024, 1, 1) <= value <= datetime(2024, 1, 31, 23, 59, 59)
        for value in column_values(data, "Trips", "booked_at")
    )
    assert validate_dataset(schema, data) == []


@pytest.mark.parametrize(
    "expression", ["travel_date >= 'soon'", "travel_date >= 20240301"]
)
def test_rejects_check_comparing_a_date_column_to_a_non_date_literal(
    expression: str,
) -> None:
    schema = parse_ddl(
        f"""
        CREATE TABLE Trips (
            trip_id INT PRIMARY KEY AUTO_INCREMENT,
            travel_date DATE NOT NULL CHECK ({expression})
        );
        """
    )

    with pytest.raises(GenerationError, match="not safely supported"):
        SyntheticDataGenerator().generate(schema, row_count=3, seed=1)


@pytest.mark.parametrize(
    "checks",
    [
        "CHECK (low_mark <= mid_mark), CHECK (mid_mark <= high_mark)",
        "CHECK (mid_mark <= high_mark), CHECK (low_mark <= mid_mark)",
        "CHECK (high_mark >= mid_mark), CHECK (mid_mark >= low_mark)",
    ],
)
def test_chained_column_comparisons_hold_for_every_seed(checks: str) -> None:
    schema = parse_ddl(
        f"""
        CREATE TABLE Marks (
            mark_id INT PRIMARY KEY AUTO_INCREMENT,
            low_mark INT NOT NULL,
            mid_mark INT NOT NULL,
            high_mark INT NOT NULL,
            {checks}
        );
        """
    )

    for seed in range(25):
        data = SyntheticDataGenerator().generate(schema, row_count=12, seed=seed)

        assert all(
            row["low_mark"] <= row["mid_mark"] <= row["high_mark"]
            for row in data["Marks"]
        ), f"seed {seed}"
        assert validate_dataset(schema, data) == []


def test_column_comparison_respects_a_single_column_lower_bound() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Ranges (
            range_id INT PRIMARY KEY AUTO_INCREMENT,
            lo INT NOT NULL CHECK (lo >= 100),
            hi INT NOT NULL,
            CHECK (lo <= hi)
        );
        """
    )

    for seed in range(25):
        data = SyntheticDataGenerator().generate(schema, row_count=12, seed=seed)

        assert all(100 <= row["lo"] <= row["hi"] for row in data["Ranges"]), f"seed {seed}"
        assert validate_dataset(schema, data) == []


def test_chained_date_comparisons_stay_inside_their_literal_bounds() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Stays (
            stay_id INT PRIMARY KEY AUTO_INCREMENT,
            arrival DATE NOT NULL CHECK (arrival >= '2024-01-01'),
            departure DATE NOT NULL CHECK (departure <= '2024-12-31'),
            CHECK (arrival < departure)
        );
        """
    )

    for seed in range(15):
        data = SyntheticDataGenerator().generate(schema, row_count=10, seed=seed)

        assert all(
            date(2024, 1, 1) <= row["arrival"] < row["departure"] <= date(2024, 12, 31)
            for row in data["Stays"]
        ), f"seed {seed}"
        assert validate_dataset(schema, data) == []


def test_reports_column_comparison_that_no_value_can_satisfy() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Ranges (
            range_id INT PRIMARY KEY AUTO_INCREMENT,
            lo INT NOT NULL CHECK (lo >= 100),
            hi INT NOT NULL CHECK (hi <= 50),
            CHECK (lo <= hi)
        );
        """
    )

    with pytest.raises(GenerationError, match=r"Ranges\.lo .*empty range"):
        SyntheticDataGenerator().generate(schema, row_count=3, seed=1)


def test_rejects_column_comparisons_that_form_a_cycle() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Loops (
            loop_id INT PRIMARY KEY AUTO_INCREMENT,
            first_value INT NOT NULL,
            second_value INT NOT NULL,
            CHECK (first_value < second_value),
            CHECK (second_value < first_value)
        );
        """
    )

    with pytest.raises(GenerationError, match="cycle"):
        SyntheticDataGenerator().generate(schema, row_count=3, seed=1)


def test_rejects_overlapping_composite_foreign_keys() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Regions (region_id INT, zone_id INT, PRIMARY KEY (region_id, zone_id));
        CREATE TABLE Zones (zone_id INT, area_id INT, PRIMARY KEY (zone_id, area_id));
        CREATE TABLE Sites (
            site_id INT PRIMARY KEY AUTO_INCREMENT,
            region_id INT NOT NULL,
            zone_id INT NOT NULL,
            area_id INT NOT NULL,
            FOREIGN KEY (region_id, zone_id) REFERENCES Regions(region_id, zone_id),
            FOREIGN KEY (zone_id, area_id) REFERENCES Zones(zone_id, area_id)
        );
        """
    )

    with pytest.raises(GenerationError, match=r"overlapping foreign keys.*zone_id"):
        SyntheticDataGenerator().generate(schema, row_count=4, seed=1)


def test_reports_composite_unique_domain_smaller_than_the_row_count() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Slots (
            slot_id INT PRIMARY KEY AUTO_INCREMENT,
            part_of_day ENUM('morning', 'evening') NOT NULL,
            weekend BOOLEAN NOT NULL,
            UNIQUE (part_of_day, weekend)
        );
        """
    )

    with pytest.raises(GenerationError, match="distinct"):
        SyntheticDataGenerator().generate(schema, row_count=5, seed=1)


def test_rejects_column_to_column_check_over_key_columns() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Spans (
            span_id INT PRIMARY KEY,
            upper_bound INT NOT NULL,
            CHECK (upper_bound >= span_id)
        );
        """
    )

    with pytest.raises(GenerationError, match="key column"):
        SyntheticDataGenerator().generate(schema, row_count=5, seed=1)


def test_plan_categories_restrict_enum_values() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Tickets (
            ticket_id INT PRIMARY KEY AUTO_INCREMENT,
            state ENUM('open', 'closed', 'archived') NOT NULL
        );
        """
    )
    hints = {
        ("Tickets", "state"): ColumnGenerationPlan(
            column_name="state", generator="enum", categories=["open", "closed"]
        )
    }

    data = SyntheticDataGenerator().generate(
        schema, plan_for(schema, row_count=30, hints=hints), seed=2
    )

    assert set(column_values(data, "Tickets", "state")) == {"open", "closed"}


def test_rejects_plan_categories_outside_the_enum_domain() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Tickets (
            ticket_id INT PRIMARY KEY AUTO_INCREMENT,
            state ENUM('open', 'closed') NOT NULL
        );
        """
    )
    hints = {
        ("Tickets", "state"): ColumnGenerationPlan(
            column_name="state", generator="enum", categories=["open", "frozen"]
        )
    }

    with pytest.raises(GenerationError, match="frozen"):
        SyntheticDataGenerator().generate(
            schema, plan_for(schema, row_count=5, hints=hints), seed=2
        )


def test_supports_negative_check_bounds_and_negative_plan_ranges() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Readings (
            reading_id INT PRIMARY KEY AUTO_INCREMENT,
            celsius INT NOT NULL CHECK (celsius <= -10),
            drift DECIMAL(6, 2) NOT NULL CHECK (drift <= -1),
            balance INT NOT NULL
        );
        """
    )
    hints = {
        ("Readings", "balance"): ColumnGenerationPlan(
            column_name="balance", generator="integer", minimum=-40, maximum=-20
        )
    }

    data = SyntheticDataGenerator().generate(
        schema, plan_for(schema, row_count=25, hints=hints), seed=6
    )

    assert all(value <= -10 for value in column_values(data, "Readings", "celsius"))
    assert all(
        value <= Decimal("-1") for value in column_values(data, "Readings", "drift")
    )
    assert all(-40 <= value <= -20 for value in column_values(data, "Readings", "balance"))
    assert validate_dataset(schema, data) == []


def test_intersects_multiple_check_allow_lists() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Grades (
            grade_id INT PRIMARY KEY AUTO_INCREMENT,
            mark VARCHAR(2) NOT NULL CHECK (mark IN ('A', 'B', 'C')),
            CHECK (mark IN ('B', 'C', 'D'))
        );
        """
    )

    data = SyntheticDataGenerator().generate(schema, row_count=30, seed=1)

    assert set(column_values(data, "Grades", "mark")) <= {"B", "C"}


def test_reports_unique_enum_domain_smaller_than_row_count() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Slots (
            slot_id INT PRIMARY KEY AUTO_INCREMENT,
            kind ENUM('morning', 'evening') NOT NULL UNIQUE
        );
        """
    )

    with pytest.raises(GenerationError, match="distinct"):
        SyntheticDataGenerator().generate(schema, row_count=5, seed=1)


def test_integer_primary_key_allocation_respects_check_bounds() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Codes (
            code_id INT PRIMARY KEY CHECK (code_id >= 100),
            label VARCHAR(20) NOT NULL
        );
        """
    )

    data = SyntheticDataGenerator().generate(schema, row_count=5, seed=1)

    assert column_values(data, "Codes", "code_id") == [100, 101, 102, 103, 104]


def test_rejects_unsupported_plan_distribution() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Items (
            item_id INT PRIMARY KEY AUTO_INCREMENT,
            score INT NOT NULL
        );
        """
    )
    hints = {
        ("Items", "score"): ColumnGenerationPlan(
            column_name="score", generator="integer", distribution="normal"
        )
    }

    with pytest.raises(GenerationError, match="distribution"):
        SyntheticDataGenerator().generate(
            schema, plan_for(schema, row_count=3, hints=hints), seed=1
        )


def test_rejects_unknown_plan_generator_hint() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Items (
            item_id INT PRIMARY KEY AUTO_INCREMENT,
            label VARCHAR(40) NOT NULL
        );
        """
    )
    hints = {
        ("Items", "label"): ColumnGenerationPlan(column_name="label", generator="teleport")
    }

    with pytest.raises(GenerationError, match="generator"):
        SyntheticDataGenerator().generate(
            schema, plan_for(schema, row_count=3, hints=hints), seed=1
        )


def test_self_referencing_foreign_key_points_at_rows_of_the_same_table() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Staff (
            staff_id INT PRIMARY KEY AUTO_INCREMENT,
            full_name VARCHAR(100) NOT NULL,
            manager_id INT,
            FOREIGN KEY (manager_id) REFERENCES Staff(staff_id)
        );
        """
    )
    plan = DatabaseGenerationPlan(
        database_description="self referencing staff",
        tables=[
            TableGenerationPlan(
                table_name="Staff",
                row_count=12,
                columns=[
                    ColumnGenerationPlan(column_name="full_name", generator="name"),
                    ColumnGenerationPlan(
                        column_name="manager_id",
                        generator="auto",
                        nullable_probability=0.25,
                    ),
                ],
            )
        ],
    ).validate_against(schema)

    data = SyntheticDataGenerator().generate(schema, plan, seed=7)

    managers = {
        value for value in column_values(data, "Staff", "manager_id") if value is not None
    }
    assert managers
    assert managers <= set(column_values(data, "Staff", "staff_id"))
    assert validate_dataset(schema, data) == []


def test_generates_one_thousand_rows_per_table_quickly() -> None:
    schema = fixture_schema("restaurants_schema.ddl")

    started = time.perf_counter()
    data = SyntheticDataGenerator().generate(schema, row_count=1000, seed=17)
    elapsed = time.perf_counter() - started

    assert all(len(rows) == 1000 for rows in data.values())
    assert validate_dataset(schema, data) == []
    assert elapsed < 20
