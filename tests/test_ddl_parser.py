from pathlib import Path

import pytest

from data_assistant.schema import DDLParseError, parse_ddl


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("fixture", "expected_tables"),
    [
        ("company_employee_schema.ddl", 7),
        ("restaurants_schema.ddl", 7),
        ("library_management_schema.ddl", 9),
    ],
)
def test_parses_supplied_mysql_schemas(fixture: str, expected_tables: int) -> None:
    schema = parse_ddl((FIXTURES / fixture).read_text())

    assert len(schema.tables) == expected_tables


def test_normalizes_representative_column_types_and_constraints() -> None:
    schema = parse_ddl((FIXTURES / "restaurants_schema.ddl").read_text())
    restaurants = schema.table("restaurants")
    customers = schema.table("CUSTOMERS")

    assert restaurants.name == "Restaurants"
    assert restaurants.column("restaurant_id").primary_key
    assert restaurants.column("restaurant_id").auto_increment
    assert restaurants.column("name").nullable is False
    assert restaurants.column("zip_code").nullable is True
    assert restaurants.column("name").type.name == "VARCHAR"
    assert restaurants.column("name").type.length == 255
    assert restaurants.column("rating").type.name == "DECIMAL"
    assert restaurants.column("rating").type.precision == 3
    assert restaurants.column("rating").type.scale == 2
    assert restaurants.column("cuisine_type").type.enum_values == [
        "Italian",
        "Mexican",
        "American",
        "Chinese",
        "Indian",
        "Other",
    ]
    assert customers.column("registration_date").default == "CURRENT_TIMESTAMP"
    assert customers.column("registration_date").type.name == "DATETIME"
    assert customers.column("email").unique
    assert schema.table("menu").column("description").type.name == "TEXT"
    assert schema.table("menu").column("available").type.name == "BOOLEAN"
    assert schema.table("menu").column("available").default == "TRUE"


def test_preserves_checks_and_quoted_enum_defaults() -> None:
    schema = parse_ddl((FIXTURES / "company_employee_schema.ddl").read_text())
    reviews = schema.table("performance_reviews")

    assert "rating >= 1" in reviews.column("rating").checks[0]
    assert "rating <= 5" in reviews.column("rating").checks[0]
    assert reviews.column("review_status").default == "'Draft'"


def test_preserves_composite_primary_key_and_unique_constraints() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Memberships (
            tenant_id INT,
            user_id INT,
            email VARCHAR(255),
            external_id VARCHAR(50),
            CONSTRAINT pk_memberships PRIMARY KEY (tenant_id, user_id),
            CONSTRAINT uq_memberships_email UNIQUE (tenant_id, email),
            UNIQUE (external_id)
        );
        """
    )
    table = schema.table("memberships")

    assert table.primary_key is not None
    assert table.primary_key.name == "pk_memberships"
    assert table.primary_key.columns == ["tenant_id", "user_id"]
    assert [constraint.model_dump() for constraint in table.unique_constraints] == [
        {
            "name": "uq_memberships_email",
            "columns": ["tenant_id", "email"],
        },
        {"name": None, "columns": ["external_id"]},
    ]
    assert table.column("tenant_id").primary_key is False
    assert table.column("user_id").primary_key is False
    assert table.column("tenant_id").nullable is False
    assert table.column("user_id").nullable is False
    assert table.column("tenant_id").unique is False
    assert table.column("email").unique is False
    assert table.column("external_id").unique is True


def test_parses_balanced_nested_check_expressions() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Scores (
            score INT CHECK ((score >= 0) AND (score <= 100 OR score IS NULL)),
            bonus INT,
            CHECK ((score + bonus) >= 0 AND COALESCE(bonus, 0) < 50)
        );
        """
    )
    scores = schema.table("scores")

    assert scores.column("score").checks == [
        "(score >= 0) AND (score <= 100 OR score IS NULL)"
    ]
    assert scores.checks == ["(score + bonus) >= 0 AND COALESCE(bonus, 0) < 50"]


def test_preserves_foreign_key_actions_for_create_and_alter() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Parents (id INT PRIMARY KEY);
        CREATE TABLE Children (
            id INT PRIMARY KEY,
            parent_id INT,
            backup_parent_id INT REFERENCES Parents(id) ON DELETE SET NULL ON UPDATE CASCADE,
            CONSTRAINT fk_parent FOREIGN KEY (parent_id) REFERENCES Parents(id)
                ON DELETE CASCADE ON UPDATE RESTRICT
        );
        ALTER TABLE Children ADD CONSTRAINT fk_backup
            FOREIGN KEY (backup_parent_id) REFERENCES Parents(id)
            ON UPDATE NO ACTION ON DELETE SET NULL;
        """
    )
    foreign_keys = schema.table("children").foreign_keys

    actions = [
        (foreign_key.name, foreign_key.on_delete, foreign_key.on_update)
        for foreign_key in foreign_keys
    ]
    assert actions == [
        (None, "SET NULL", "CASCADE"),
        ("fk_parent", "CASCADE", "RESTRICT"),
        ("fk_backup", "SET NULL", "NO ACTION"),
    ]


def test_line_comments_do_not_remove_double_hyphens_inside_literals() -> None:
    schema = parse_ddl(
        """
        CREATE TABLE Payments (
            method ENUM('card--online', 'cash') DEFAULT 'card--online', -- actual comment
            note VARCHAR(100) DEFAULT '--not a comment'
        );
        """
    )
    payments = schema.table("payments")

    assert payments.column("method").type.enum_values == ["card--online", "cash"]
    assert payments.column("method").default == "'card--online'"
    assert payments.column("note").default == "'--not a comment'"


def test_resolves_alter_forward_self_and_circular_foreign_keys() -> None:
    schema = parse_ddl((FIXTURES / "library_management_schema.ddl").read_text())

    branch_fk = schema.table("library_branches").foreign_keys[0]
    assert branch_fk.name == "FK_Library_Branches_ManagerID"
    assert branch_fk.referenced_table == "Employees"
    assert branch_fk.referenced_columns == ["employee_id"]
    assert schema.table("departments").foreign_keys[0].referenced_table == "Employees"

    synthetic = parse_ddl(
        """
        CREATE TABLE Parent (
            id INT PRIMARY KEY,
            child_id INT,
            CONSTRAINT fk_child FOREIGN KEY (child_id) REFERENCES Child(id)
        );
        CREATE TABLE Child (
            id INT PRIMARY KEY,
            parent_id INT REFERENCES Parent(id),
            mentor_id INT REFERENCES Child(id)
        );
        """
    )

    assert len(synthetic.table("child").foreign_keys) == 2
    assert synthetic.table("child").foreign_keys[1].referenced_table == "Child"
    analysis = synthetic.analyze_dependencies()
    assert analysis.insertion_groups == [["Child", "Parent"]]
    assert analysis.cycles == [["Child", "Parent"]]


def test_dependency_analysis_returns_topological_groups() -> None:
    schema = parse_ddl((FIXTURES / "restaurants_schema.ddl").read_text())
    analysis = schema.analyze_dependencies()
    positions = {
        table.casefold(): index
        for index, group in enumerate(analysis.insertion_groups)
        for table in group
    }

    assert positions["customers"] < positions["orders"]
    assert positions["restaurants"] < positions["menu"]
    assert positions["orders"] < positions["order_items"]
    assert positions["menu"] < positions["order_items"]
    assert analysis.cycles == []


@pytest.mark.parametrize(
    ("ddl", "message"),
    [
        ("", "empty"),
        ("CREATE VIEW v AS SELECT 1", "CREATE TABLE"),
        ("CREATE TABLE broken (id INT", "parenthesis"),
        ("CREATE TABLE t (id INT, x UUID)", "unsupported"),
        (
            "CREATE TABLE parent (id INT PRIMARY KEY);"
            "CREATE TABLE child (parent_id INT REFERENCES parent(id) "
            "ON DELETE CASCADE MATCH FULL);",
            "suffix",
        ),
        (
            "CREATE TABLE child (id INT, parent_id INT, "
            "FOREIGN KEY (parent_id) REFERENCES missing(id));",
            "missing table",
        ),
        (
            "CREATE TABLE parent (id INT);"
            "CREATE TABLE child (id INT, parent_id INT, "
            "FOREIGN KEY (parent_id) REFERENCES parent(missing));",
            "missing column",
        ),
    ],
)
def test_rejects_invalid_or_unsupported_schemas(ddl: str, message: str) -> None:
    with pytest.raises(DDLParseError, match=message):
        parse_ddl(ddl)
