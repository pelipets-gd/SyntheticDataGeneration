"""End-to-end storage tests against a real PostgreSQL.

These are skipped unless `TEST_DATABASE_URL` is passed explicitly, so the normal
suite needs no database and no credentials:

    TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/db \\
        python3.12 -m pytest tests -m integration

Every dataset schema the tests create is dropped again on teardown.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from psycopg import sql

from data_assistant.generation import SyntheticDataGenerator
from data_assistant.schema import parse_ddl
from data_assistant.storage import (
    METADATA_TABLE,
    DatasetRepository,
    DatasetSummary,
    StorageError,
    default_connect,
)


DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "").strip()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set"),
]

CYCLIC_DDL = """
CREATE TABLE Team (
    team_id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(40) NOT NULL,
    budget DECIMAL(9, 2) NOT NULL DEFAULT 100.00,
    lead_id INT,
    FOREIGN KEY (lead_id) REFERENCES Member(member_id) ON DELETE SET NULL
);
CREATE TABLE Member (
    member_id INT PRIMARY KEY AUTO_INCREMENT,
    full_name VARCHAR(60) NOT NULL,
    team_id INT NOT NULL,
    role ENUM('lead', 'engineer') NOT NULL,
    FOREIGN KEY (team_id) REFERENCES Team(team_id) ON DELETE CASCADE
);
"""


class _Store:
    """A repository that drops every dataset schema it created on teardown."""

    def __init__(self, dataset_id: UUID | None = None) -> None:
        self.connect = default_connect(DATABASE_URL)
        self.created: list[DatasetSummary] = []
        self.repository = DatasetRepository(
            connect=self.connect,
            new_dataset_id=(lambda: dataset_id) if dataset_id else uuid4,
        )
        self.repository.initialize()

    def save_sample(self, name: str = "Teams") -> DatasetSummary:
        schema = parse_ddl(CYCLIC_DDL)
        data = SyntheticDataGenerator().generate(schema, row_count=5, seed=3)
        summary = self.repository.save_dataset(
            name=name, ddl=CYCLIC_DDL, schema=schema, data=data, instructions="five rows"
        )
        self.created.append(summary)
        return summary

    def schema_exists(self, schema_name: str) -> bool:
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                (schema_name,),
            )
            return cursor.fetchone() is not None

    def cleanup(self) -> None:
        with self.connect() as connection, connection.cursor() as cursor:
            for summary in self.created:
                cursor.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(summary.schema_name)
                    )
                )
                cursor.execute(
                    sql.SQL("DELETE FROM {} WHERE {} = %s").format(
                        sql.Identifier(METADATA_TABLE), sql.Identifier("dataset_id")
                    ),
                    (summary.dataset_id,),
                )
            connection.commit()


@pytest.fixture
def store() -> Iterator[_Store]:
    instance = _Store()
    try:
        yield instance
    finally:
        instance.cleanup()


def test_saves_a_cyclic_dataset_and_reads_it_back(store: _Store) -> None:
    summary = store.save_sample()

    assert summary.dataset_id in {item.dataset_id for item in store.repository.list_datasets()}
    record = store.repository.load_dataset(summary.dataset_id)
    assert record.instructions == "five rows"
    assert [table.name for table in record.schema.tables] == ["Team", "Member"]

    page = store.repository.read_table(summary.dataset_id, "team", limit=2, offset=1)
    assert page.total == 5
    assert len(page.rows) == 2
    assert isinstance(page.rows[0]["budget"], Decimal)
    assert [row["team_id"] for row in page.rows] == sorted(
        row["team_id"] for row in page.rows
    )


def test_updates_one_row_in_place_and_keeps_the_other_tables(store: _Store) -> None:
    summary = store.save_sample()
    page = store.repository.read_table(summary.dataset_id, "Team", limit=5)
    edited = [dict(row) for row in page.rows]
    edited[0]["title"] = "Renamed team"

    updated = store.repository.update_table(summary.dataset_id, "Team", edited)

    assert updated == 1
    after = store.repository.read_table(summary.dataset_id, "Team", limit=5)
    assert after.rows[0]["title"] == "Renamed team"
    assert after.total == page.total
    assert store.repository.read_table(summary.dataset_id, "Member", limit=5).total == 5


def test_a_failed_save_leaves_no_schema_and_no_metadata_row() -> None:
    dataset_id = uuid4()
    store = _Store(dataset_id=dataset_id)
    try:
        first = store.save_sample(name="First")

        with pytest.raises(StorageError):
            store.repository.save_dataset(
                name="Second",
                ddl=CYCLIC_DDL,
                schema=parse_ddl(CYCLIC_DDL),
                data=SyntheticDataGenerator().generate(
                    parse_ddl(CYCLIC_DDL), row_count=5, seed=3
                ),
            )

        stored = [
            item
            for item in store.repository.list_datasets()
            if item.dataset_id == dataset_id
        ]
        assert [item.name for item in stored] == ["First"]
        assert store.schema_exists(first.schema_name)
    finally:
        store.cleanup()
