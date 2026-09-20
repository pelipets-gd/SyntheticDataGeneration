"""Unit tests for PostgreSQL dataset persistence against a recording connection.

The fake connection records the composed SQL of every statement, so the tests
assert identifier quoting, statement ordering inside the transaction, and the
commit/rollback decision without needing a database.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest

from data_assistant.schema import DatabaseSchema, parse_ddl
from data_assistant.storage import (
    METADATA_TABLE,
    DatasetNotFoundError,
    DatasetRepository,
    DatasetValidationError,
    StorageError,
)


DATASET_ID = UUID("0123456789abcdef0123456789abcdef")
SCHEMA_NAME = f"dataset_{DATASET_ID.hex}"
CREATED_AT = datetime(2025, 3, 1, 9, 30)
LEAKY_FAILURE = (
    'connection failed: postgresql://data_assistant:sup3rsecret@db:5432/app — '
    'FATAL: password authentication failed'
)


@dataclass(frozen=True)
class Statement:
    """One recorded statement: its composed SQL text and bound parameters."""

    text: str
    params: Any


class FakeCursor:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection
        self._rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def execute(self, query: Any, params: Any = None) -> FakeCursor:
        self._rows = self._connection.record(query, params)
        return self

    def executemany(self, query: Any, params_seq: Any) -> None:
        self._connection.record(query, list(params_seq))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None


class FakeConnection:
    """Records statements and replays canned rows for matching queries."""

    def __init__(
        self,
        results: tuple[tuple[str, list[tuple[Any, ...]]], ...] = (),
        fail_on: str | None = None,
    ) -> None:
        self.statements: list[Statement] = []
        self.events: list[str] = []
        self._results = results
        self._fail_on = fail_on

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def record(self, query: Any, params: Any) -> list[tuple[Any, ...]]:
        text = query.as_string()
        self.statements.append(Statement(text, params))
        if self._fail_on is not None and self._fail_on in text:
            raise RuntimeError(LEAKY_FAILURE)
        for needle, rows in self._results:
            if needle in text:
                return list(rows)
        return []

    def commit(self) -> None:
        self.events.append("commit")

    def rollback(self) -> None:
        self.events.append("rollback")

    def close(self) -> None:
        self.events.append("close")

    @property
    def texts(self) -> list[str]:
        return [statement.text for statement in self.statements]


def repository(connection: FakeConnection) -> DatasetRepository:
    def connect() -> FakeConnection:
        connection.events.append("connect")
        return connection

    return DatasetRepository(connect=connect, new_dataset_id=lambda: DATASET_ID)


def cyclic_schema() -> DatabaseSchema:
    return parse_ddl(
        """
        CREATE TABLE Team (
            team_id INT PRIMARY KEY,
            lead_id INT,
            FOREIGN KEY (lead_id) REFERENCES Member(member_id) ON DELETE SET NULL
        );
        CREATE TABLE Member (
            member_id INT PRIMARY KEY,
            team_id INT NOT NULL,
            FOREIGN KEY (team_id) REFERENCES Team(team_id) ON DELETE CASCADE
        );
        """
    )


CYCLIC_DATA = {
    "Team": [{"team_id": 1, "lead_id": 1}],
    "Member": [{"member_id": 1, "team_id": 1}],
}


def catalog_schema() -> DatabaseSchema:
    return parse_ddl(
        """
        CREATE TABLE Item (
            item_id INT NOT NULL,
            region VARCHAR(10) NOT NULL,
            label VARCHAR(50) DEFAULT 'none',
            notes TEXT,
            due DATE,
            created DATETIME DEFAULT CURRENT_TIMESTAMP,
            price DECIMAL(8, 2) NOT NULL DEFAULT 1.50,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            status ENUM('open', 'closed') NOT NULL,
            quantity INT NOT NULL CHECK (quantity > 0),
            PRIMARY KEY (item_id, region),
            UNIQUE (label, notes)
        );
        """
    )


CATALOG_ROW = {
    "item_id": 1,
    "region": "eu",
    "label": "first",
    "notes": "a note",
    "due": date(2025, 5, 1),
    "created": datetime(2025, 5, 1, 12, 0),
    "price": Decimal("10.00"),
    "active": True,
    "status": "open",
    "quantity": 3,
}


def priced_schema() -> DatabaseSchema:
    return parse_ddl(
        """
        CREATE TABLE Item (
            item_id INT PRIMARY KEY,
            price DECIMAL(6, 2) NOT NULL,
            label VARCHAR(20)
        );
        """
    )


PRICED_METADATA_ROW = (
    DATASET_ID,
    "Catalog",
    SCHEMA_NAME,
    "CREATE TABLE Item (item_id INT PRIMARY KEY);",
    "keep it small",
    priced_schema().model_dump_json(),
    CREATED_AT,
)
PRICED_STORED_ROWS = [
    (1, Decimal("10.00"), "first"),
    (2, Decimal("20.00"), "second"),
]


def metadata_result(
    schema: DatabaseSchema | None = None,
) -> tuple[str, list[tuple[Any, ...]]]:
    row = PRICED_METADATA_ROW
    if schema is not None:
        row = (*row[:5], schema.model_dump_json(), row[6])
    return (f'FROM "{METADATA_TABLE}"', [row])


def test_initialize_creates_the_fixed_metadata_table_once_per_transaction() -> None:
    connection = FakeConnection()

    repository(connection).initialize()

    assert connection.texts == [
        'CREATE TABLE IF NOT EXISTS "data_assistant_datasets" ('
        '"dataset_id" uuid PRIMARY KEY, '
        '"name" text NOT NULL, '
        '"schema_name" text NOT NULL UNIQUE, '
        '"ddl" text NOT NULL, '
        '"instructions" text, '
        '"schema_json" jsonb NOT NULL, '
        '"created_at" timestamptz NOT NULL DEFAULT now())'
    ]
    assert connection.events == ["connect", "commit", "close"]


def test_save_dataset_creates_schema_and_tables_then_rows_then_foreign_keys() -> None:
    connection = FakeConnection()

    summary = repository(connection).save_dataset(
        name="Teams", ddl="CREATE TABLE Team ();", schema=cyclic_schema(), data=CYCLIC_DATA
    )

    assert connection.texts == [
        f'CREATE SCHEMA "{SCHEMA_NAME}"',
        f'CREATE TABLE "{SCHEMA_NAME}"."Team" ('
        '"team_id" integer NOT NULL, "lead_id" integer, PRIMARY KEY ("team_id"))',
        f'CREATE TABLE "{SCHEMA_NAME}"."Member" ('
        '"member_id" integer NOT NULL, "team_id" integer NOT NULL, '
        'PRIMARY KEY ("member_id"))',
        f'INSERT INTO "{SCHEMA_NAME}"."Team" ("team_id", "lead_id") VALUES (%s, %s)',
        f'INSERT INTO "{SCHEMA_NAME}"."Member" ("member_id", "team_id") VALUES (%s, %s)',
        f'ALTER TABLE "{SCHEMA_NAME}"."Team" ADD FOREIGN KEY ("lead_id") '
        f'REFERENCES "{SCHEMA_NAME}"."Member" ("member_id") ON DELETE SET NULL',
        f'ALTER TABLE "{SCHEMA_NAME}"."Member" ADD FOREIGN KEY ("team_id") '
        f'REFERENCES "{SCHEMA_NAME}"."Team" ("team_id") ON DELETE CASCADE',
        f'INSERT INTO "{METADATA_TABLE}" ("dataset_id", "name", "schema_name", "ddl", '
        '"instructions", "schema_json") VALUES (%s, %s, %s, %s, %s, %s::jsonb)',
    ]
    assert connection.events == ["connect", "commit", "close"]
    assert summary.schema_name == SCHEMA_NAME
    assert summary.name == "Teams"


def test_save_dataset_binds_rows_as_parameters_never_as_literal_sql() -> None:
    connection = FakeConnection()

    repository(connection).save_dataset(
        name="Teams", ddl="CREATE TABLE Team ();", schema=cyclic_schema(), data=CYCLIC_DATA
    )

    inserts = [
        statement
        for statement in connection.statements
        if statement.text.startswith(f'INSERT INTO "{SCHEMA_NAME}"')
    ]
    assert [statement.params for statement in inserts] == [[(1, 1)], [(1, 1)]]


def test_save_dataset_generates_an_isolated_server_side_schema_name() -> None:
    connection = FakeConnection()

    summary = DatasetRepository(connect=lambda: connection).save_dataset(
        name="Teams", ddl="CREATE TABLE Team ();", schema=cyclic_schema(), data=CYCLIC_DATA
    )

    assert re.fullmatch(r"dataset_[0-9a-f]{32}", summary.schema_name)
    assert summary.dataset_id.hex == summary.schema_name.removeprefix("dataset_")


def test_save_dataset_quotes_hostile_identifiers_instead_of_interpolating_them() -> None:
    schema = parse_ddl('CREATE TABLE `We"ird` (`co"l` INT PRIMARY KEY);')
    connection = FakeConnection()

    repository(connection).save_dataset(
        name="Hostile", ddl="ddl", schema=schema, data={'We"ird': [{'co"l': 1}]}
    )

    assert connection.texts[1] == (
        f'CREATE TABLE "{SCHEMA_NAME}"."We""ird" ("co""l" integer NOT NULL, '
        'PRIMARY KEY ("co""l"))'
    )


def test_save_dataset_translates_every_supported_type_default_and_constraint() -> None:
    connection = FakeConnection()

    repository(connection).save_dataset(
        name="Catalog", ddl="ddl", schema=catalog_schema(), data={"Item": [CATALOG_ROW]}
    )

    assert connection.texts[1] == (
        f'CREATE TABLE "{SCHEMA_NAME}"."Item" ('
        '"item_id" integer NOT NULL, '
        '"region" varchar(10) NOT NULL, '
        '"label" varchar(50) DEFAULT \'none\', '
        '"notes" text, '
        '"due" date, '
        '"created" timestamp DEFAULT CURRENT_TIMESTAMP, '
        '"price" numeric(8, 2) NOT NULL DEFAULT 1.50, '
        '"active" boolean NOT NULL DEFAULT true, '
        '"status" text NOT NULL, '
        '"quantity" integer NOT NULL, '
        'PRIMARY KEY ("item_id", "region"), '
        'UNIQUE ("label", "notes"), '
        'CHECK ("status" IN (\'open\', \'closed\')), '
        'CHECK ("quantity" > 0))'
    )


def test_save_dataset_omits_defaults_and_checks_that_do_not_translate_safely() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Note (
            note_id INT PRIMARY KEY,
            body VARCHAR(80) DEFAULT UUID(),
            CHECK (LENGTH(body) > 2)
        );
        """
    )
    connection = FakeConnection()

    repository(connection).save_dataset(
        name="Notes", ddl="ddl", schema=schema, data={"Note": [{"note_id": 1, "body": "hi"}]}
    )

    assert connection.texts[1] == (
        f'CREATE TABLE "{SCHEMA_NAME}"."Note" ('
        '"note_id" integer NOT NULL, "body" varchar(80), PRIMARY KEY ("note_id"))'
    )


def test_save_dataset_rejects_invalid_data_before_opening_a_connection() -> None:
    connection = FakeConnection()

    with pytest.raises(DatasetValidationError) as error:
        repository(connection).save_dataset(
            name="Teams",
            ddl="ddl",
            schema=cyclic_schema(),
            data={"Team": [{"team_id": 1, "lead_id": 9}], "Member": []},
        )

    assert [violation.kind for violation in error.value.violations] == ["foreign_key_missing"]
    assert connection.events == []


def test_save_dataset_rolls_back_and_hides_driver_detail_when_a_statement_fails() -> None:
    connection = FakeConnection(fail_on="ALTER TABLE")

    with pytest.raises(StorageError) as error:
        repository(connection).save_dataset(
            name="Teams", ddl="ddl", schema=cyclic_schema(), data=CYCLIC_DATA
        )

    assert connection.events == ["connect", "rollback", "close"]
    assert "sup3rsecret" not in str(error.value)
    assert "postgresql://" not in str(error.value)
    assert error.value.__cause__ is None


def test_list_datasets_reads_the_metadata_table_from_every_new_repository() -> None:
    connection = FakeConnection(
        results=((f'FROM "{METADATA_TABLE}"', [(DATASET_ID, "Catalog", SCHEMA_NAME, CREATED_AT)]),)
    )

    first = repository(connection).list_datasets()
    second = repository(connection).list_datasets()

    assert first == second
    assert [(item.name, item.schema_name) for item in first] == [("Catalog", SCHEMA_NAME)]
    assert connection.texts[0] == (
        f'SELECT "dataset_id", "name", "schema_name", "created_at" FROM "{METADATA_TABLE}" '
        'ORDER BY "created_at" DESC, "dataset_id"'
    )


def test_load_dataset_returns_metadata_and_the_normalized_schema() -> None:
    connection = FakeConnection(results=(metadata_result(),))

    record = repository(connection).load_dataset(DATASET_ID)

    assert connection.statements[0].params == (DATASET_ID,)
    assert record.name == "Catalog"
    assert record.instructions == "keep it small"
    assert record.schema == priced_schema()
    assert record.schema.table("Item").column("price").type.precision == 6


def test_load_dataset_raises_a_not_found_error_for_an_unknown_dataset() -> None:
    connection = FakeConnection()

    with pytest.raises(DatasetNotFoundError):
        repository(connection).load_dataset(DATASET_ID)


def test_list_tables_returns_the_table_names_of_the_stored_schema() -> None:
    connection = FakeConnection(results=(metadata_result(cyclic_schema()),))

    assert repository(connection).list_tables(DATASET_ID) == ["Team", "Member"]


def test_read_table_pages_rows_in_primary_key_order_with_bound_limits() -> None:
    connection = FakeConnection(
        results=(
            metadata_result(),
            ("count(*)", [(7,)]),
            (f'FROM "{SCHEMA_NAME}"."Item"', PRICED_STORED_ROWS),
        )
    )

    page = repository(connection).read_table(DATASET_ID, "item", limit=2, offset=4)

    assert connection.texts[2] == (
        f'SELECT "item_id", "price", "label" FROM "{SCHEMA_NAME}"."Item" '
        'ORDER BY "item_id" LIMIT %s OFFSET %s'
    )
    assert connection.statements[2].params == (2, 4)
    assert page.total == 7
    assert page.rows == (
        {"item_id": 1, "price": Decimal("10.00"), "label": "first"},
        {"item_id": 2, "price": Decimal("20.00"), "label": "second"},
    )


def test_read_table_rejects_an_unknown_table_and_a_non_positive_page_size() -> None:
    connection = FakeConnection(results=(metadata_result(),))
    store = repository(connection)

    with pytest.raises(StorageError, match="Ghost"):
        store.read_table(DATASET_ID, "Ghost")
    with pytest.raises(StorageError):
        store.read_table(DATASET_ID, "Item", limit=0)
    with pytest.raises(StorageError):
        store.read_table(DATASET_ID, "Item", offset=-1)


def test_update_table_writes_only_the_changed_rows_by_primary_key() -> None:
    connection = FakeConnection(
        results=(metadata_result(), (f'FROM "{SCHEMA_NAME}"."Item"', PRICED_STORED_ROWS))
    )

    updated = repository(connection).update_table(
        DATASET_ID,
        "Item",
        [
            {"item_id": 1, "price": Decimal("10.00"), "label": "first"},
            {"item_id": 2, "price": Decimal("25.50"), "label": "second"},
        ],
    )

    assert updated == 1
    assert connection.texts[-1] == (
        f'UPDATE "{SCHEMA_NAME}"."Item" SET "price" = %s, "label" = %s WHERE "item_id" = %s'
    )
    assert connection.statements[-1].params == [(Decimal("25.50"), "second", 2)]
    assert connection.events == ["connect", "commit", "close"]
    assert not any(
        word in text for text in connection.texts for word in ("TRUNCATE", "DELETE", "DROP")
    )


def test_update_table_rejects_changes_to_the_primary_key_values() -> None:
    connection = FakeConnection(
        results=(metadata_result(), (f'FROM "{SCHEMA_NAME}"."Item"', PRICED_STORED_ROWS))
    )

    with pytest.raises(StorageError, match="primary key"):
        repository(connection).update_table(
            DATASET_ID,
            "Item",
            [
                {"item_id": 1, "price": Decimal("10.00"), "label": "first"},
                {"item_id": 3, "price": Decimal("20.00"), "label": "second"},
            ],
        )

    assert not any(text.startswith("UPDATE") for text in connection.texts)
    assert "commit" not in connection.events


def test_update_table_validates_the_whole_dataset_before_writing_anything() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Parent (parent_id INT PRIMARY KEY);
        CREATE TABLE Child (
            child_id INT PRIMARY KEY,
            parent_id INT NOT NULL,
            FOREIGN KEY (parent_id) REFERENCES Parent(parent_id)
        );
        """
    )
    connection = FakeConnection(
        results=(
            metadata_result(schema),
            (f'FROM "{SCHEMA_NAME}"."Parent"', [(1,)]),
            (f'FROM "{SCHEMA_NAME}"."Child"', [(1, 1)]),
        )
    )

    with pytest.raises(DatasetValidationError) as error:
        repository(connection).update_table(
            DATASET_ID, "Child", [{"child_id": 1, "parent_id": 99}]
        )

    assert [violation.kind for violation in error.value.violations] == ["foreign_key_missing"]
    assert not any(text.startswith("UPDATE") for text in connection.texts)
    assert connection.events == ["connect", "rollback", "close"]


def test_update_table_reports_unknown_columns_as_violations_before_any_sql() -> None:
    connection = FakeConnection(
        results=(metadata_result(), (f'FROM "{SCHEMA_NAME}"."Item"', PRICED_STORED_ROWS))
    )

    with pytest.raises(DatasetValidationError) as error:
        repository(connection).update_table(
            DATASET_ID, "Item", [{"item_id": 1, "price": Decimal("1.00"), "bogus": 2}]
        )

    assert {violation.kind for violation in error.value.violations} == {
        "unknown_column",
        "missing_column",
    }
    assert not any(text.startswith("UPDATE") for text in connection.texts)


def test_execute_select_sets_read_only_session_then_runs_qualified_sql() -> None:
    connection = FakeConnection(
        results=(
            metadata_result(),
            (f'FROM "{SCHEMA_NAME}"."Item"', [(1, Decimal("10.00"))]),
        )
    )

    page = repository(connection).execute_select(
        DATASET_ID,
        "SELECT item_id, price FROM Item",
        explanation="prices",
        max_rows=25,
    )

    assert connection.texts[1] == f'SET LOCAL search_path TO "{SCHEMA_NAME}", pg_temp'
    assert connection.texts[2] == "SET LOCAL transaction_read_only = on"
    assert connection.texts[3] == "SET LOCAL statement_timeout = '15s'"
    assert f'FROM "{SCHEMA_NAME}"."Item"' in connection.texts[4]
    assert "LIMIT 25" in connection.texts[4].upper()
    assert page.explanation == "prices"
    assert page.rows == ({"item_id": 1, "price": Decimal("10.00")},)


def test_execute_select_rejects_writes_before_running_user_sql() -> None:
    connection = FakeConnection(results=(metadata_result(),))

    with pytest.raises(StorageError, match="SELECT"):
        repository(connection).execute_select(DATASET_ID, "DELETE FROM Item")

    assert not any("DELETE" in text for text in connection.texts)
