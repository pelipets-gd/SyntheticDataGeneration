"""Readonly SQL compilation for Talk-to-your-data queries."""

from __future__ import annotations

import pytest

from data_assistant.querying import QueryError, compile_readonly_select
from data_assistant.schema import parse_ddl


SCHEMA = parse_ddl(
    """
    CREATE TABLE Customers (
        id INT PRIMARY KEY,
        category ENUM('retail', 'business') NOT NULL,
        score DECIMAL(5, 2)
    );
    CREATE TABLE Orders (
        id INT PRIMARY KEY,
        customer_id INT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES Customers(id)
    );
    """
)
SCHEMA_NAME = "dataset_" + "ab" * 16


def test_qualifies_known_tables_and_caps_limit() -> None:
    compiled = compile_readonly_select(
        "SELECT category, COUNT(*) AS n FROM Customers GROUP BY category",
        SCHEMA,
        SCHEMA_NAME,
        max_rows=50,
    )

    assert SCHEMA_NAME in compiled.sql
    assert '"Customers"' in compiled.sql
    assert "LIMIT 50" in compiled.sql.upper().replace("\n", " ")
    assert "n" in compiled.columns


@pytest.mark.parametrize(
    "sql_text",
    [
        "INSERT INTO Customers (id, category, score) VALUES (1, 'retail', 1)",
        "UPDATE Customers SET score = 0",
        "DELETE FROM Customers",
        "DROP TABLE Customers",
        "SELECT * FROM Customers; DROP TABLE Customers",
        "SELECT * FROM pg_stat_activity",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT * INTO tmp FROM Customers",
        "SELECT * FROM unknown_table",
        "SELECT * FROM public.Customers",
        "SET search_path TO public",
        "COPY Customers TO STDOUT",
    ],
)
def test_rejects_writes_unknown_tables_and_catalog_functions(sql_text: str) -> None:
    with pytest.raises(QueryError):
        compile_readonly_select(sql_text, SCHEMA, SCHEMA_NAME)


def test_allows_cte_and_join_of_known_tables() -> None:
    compiled = compile_readonly_select(
        """
        WITH top AS (
            SELECT id FROM Customers WHERE category = 'retail'
        )
        SELECT o.id FROM Orders o JOIN top ON o.customer_id = top.id
        """,
        SCHEMA,
        SCHEMA_NAME,
        max_rows=10,
    )

    assert '"Orders"' in compiled.sql
    assert '"Customers"' in compiled.sql
    assert "LIMIT 10" in compiled.sql.upper()


def test_existing_limit_is_capped_not_removed() -> None:
    compiled = compile_readonly_select(
        "SELECT id FROM Customers LIMIT 5000",
        SCHEMA,
        SCHEMA_NAME,
        max_rows=25,
    )

    assert compiled.sql.upper().count("LIMIT") == 1
    assert "LIMIT 25" in compiled.sql.upper()
