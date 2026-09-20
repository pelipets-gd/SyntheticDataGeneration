"""Deterministic, constraint-safe synthetic data generation.

Keys are allocated before any other value, so foreign keys can be drawn from
real parent tuples even for forward, self, and circular relationships. Columns
that CHECK constraints order against each other are generated in that order,
never repaired afterwards. Every dataset is validated before it is returned.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from functools import partial
from itertools import combinations, product
from typing import Any, TypeVar

from faker import Faker

from data_assistant.gemini_planning import ColumnGenerationPlan, DatabaseGenerationPlan
from data_assistant.schema import (
    ColumnSchema,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)
from data_assistant.validation import (
    ColumnComparison,
    ColumnConstraint,
    DataValidationError,
    assert_valid_dataset,
    column_comparisons,
    derive_column_constraint,
    unsupported_check_violations,
)


GeneratedDataset = dict[str, list[dict[str, Any]]]

DEFAULT_ROW_COUNT = 25
DEFAULT_MAX_ROW_COUNT = 1_000

_MISSING = object()
_DEFAULT_PROBABILITY = 0.25
_RANGE_START = datetime(2015, 1, 1)
_RANGE_END = datetime(2025, 12, 31, 23, 59, 59)
_REFERENCE_NOW = datetime(2025, 6, 1, 12, 0, 0)
_EPOCH = datetime(1970, 1, 1)
_INT_MINIMUM = 0
_INT_MAXIMUM = 10_000
_DECIMAL_MAXIMUM = Decimal(10_000)
_Number = TypeVar("_Number", int, Decimal)
_UNIQUE_ATTEMPTS = 12
_MUTATION_ATTEMPTS = 10_000
_ORDERING_OPERATORS = frozenset({"<", "<=", ">", ">="})
_ORDERED_TYPES = frozenset({"INT", "DECIMAL", "DATE", "DATETIME"})
_SUPPORTED_DISTRIBUTIONS = frozenset({"uniform", "sequential"})
_TYPE_GENERATORS = frozenset(
    {
        "auto",
        "boolean",
        "date",
        "datetime",
        "decimal",
        "default",
        "enum",
        "float",
        "int",
        "integer",
        "number",
        "numeric",
        "random",
        "timestamp",
    }
)


class GenerationError(RuntimeError):
    """Raised when constraint-safe data cannot be produced for a schema."""


class SyntheticDataGenerator:
    """Produces deterministic constraint-safe rows for a normalized schema."""

    def __init__(
        self,
        *,
        default_row_count: int = DEFAULT_ROW_COUNT,
        max_row_count: int = DEFAULT_MAX_ROW_COUNT,
        locale: str = "en_US",
    ) -> None:
        if default_row_count < 1 or max_row_count < 1:
            raise ValueError("row counts must be positive")
        self._default_row_count = default_row_count
        self._max_row_count = max_row_count
        self._locale = locale

    def generate(
        self,
        schema: DatabaseSchema,
        plan: DatabaseGenerationPlan | None = None,
        *,
        row_count: int | None = None,
        seed: int = 0,
    ) -> GeneratedDataset:
        """Generate one dataset keyed by canonical table name.

        # Errors
        Raises `GenerationError` when the schema or plan asks for something this
        generator cannot honour safely, when the request exceeds configured
        limits, or when the produced dataset does not satisfy the schema.
        """
        if plan is not None:
            plan = plan.validate_against(schema)
        counts = self._resolve_row_counts(schema, plan, row_count)
        orders = _preflight(schema, plan, counts)
        data = _Builder(schema, plan, counts, seed, self._locale, orders).build()
        try:
            assert_valid_dataset(schema, data)
        except DataValidationError as error:
            raise GenerationError(
                f"generated dataset does not satisfy the schema: {error}"
            ) from error
        return data

    def _resolve_row_counts(
        self,
        schema: DatabaseSchema,
        plan: DatabaseGenerationPlan | None,
        row_count: int | None,
    ) -> dict[str, int]:
        planned = (
            {table.table_name.casefold(): table.row_count for table in plan.tables}
            if plan is not None
            else {}
        )
        counts: dict[str, int] = {}
        for table in schema.tables:
            count = (
                row_count
                if row_count is not None
                else planned.get(table.name.casefold(), self._default_row_count)
            )
            if count < 1:
                raise GenerationError(f"table {table.name!r} needs at least one row")
            if count > self._max_row_count:
                raise GenerationError(
                    f"table {table.name!r} requests {count} rows, above the configured "
                    f"maximum of {self._max_row_count}"
                )
            counts[table.name] = count
        return counts


@dataclass
class _ColumnSpec:
    generator: str
    constraint: ColumnConstraint
    plan: ColumnGenerationPlan | None
    locale: str | None
    null_probability: float
    default: Any


class _Builder:
    """Single-use builder holding the deterministic state for one dataset."""

    def __init__(
        self,
        schema: DatabaseSchema,
        plan: DatabaseGenerationPlan | None,
        counts: dict[str, int],
        seed: int,
        locale: str,
        orders: dict[str, _ComparisonOrder | None],
    ) -> None:
        self._schema = schema
        self._counts = counts
        self._orders = orders
        self._seed = seed
        self._locale = locale
        self._random = random.Random(seed)
        self._fakers: dict[str, Faker] = {}
        self._plans = _plans_by_column(plan)
        self._data: GeneratedDataset = {
            table.name: [{} for _ in range(counts[table.name])] for table in schema.tables
        }
        self._assigned: dict[str, set[str]] = {table.name: set() for table in schema.tables}

    def build(self) -> GeneratedDataset:
        self._allocate_primary_keys()
        self._generate_plain_columns()
        self._generate_ordered_columns()
        self._assign_foreign_keys()
        return self._ordered_rows()

    def _faker(self, locale: str | None) -> Faker:
        key = locale or self._locale
        cached = self._fakers.get(key)
        if cached is not None:
            return cached
        try:
            faker = Faker(key)
        except (AttributeError, TypeError, ValueError):
            faker = Faker(self._locale)
        faker.seed_instance(self._seed)
        self._fakers[key] = faker
        return faker

    def _allocate_primary_keys(self) -> None:
        for table in self._primary_key_order():
            if table.primary_key is None:
                continue
            self._allocate_table_keys(table, table.primary_key.columns)

    def _primary_key_order(self) -> list[TableSchema]:
        dependencies = {
            table.name: self._key_dependencies(table) for table in self._schema.tables
        }
        remaining = {table.name: table for table in self._schema.tables}
        ordered: list[TableSchema] = []
        while remaining:
            ready = [
                name
                for name in remaining
                if not (dependencies[name] & set(remaining)) - {name}
            ]
            if not ready:
                raise GenerationError(
                    "cannot allocate primary keys: foreign keys inside primary keys form "
                    f"a cycle across {', '.join(sorted(remaining))}"
                )
            for name in ready:
                ordered.append(remaining.pop(name))
        return ordered

    def _key_dependencies(self, table: TableSchema) -> set[str]:
        if table.primary_key is None:
            return set()
        key_columns = {name.casefold() for name in table.primary_key.columns}
        dependencies = {
            foreign_key.referenced_table
            for foreign_key in table.foreign_keys
            if {name.casefold() for name in foreign_key.columns} & key_columns
        }
        if table.name in dependencies:
            raise GenerationError(
                f"table {table.name!r} cannot derive its primary key from a "
                "self-referencing foreign key"
            )
        return dependencies

    def _allocate_table_keys(self, table: TableSchema, columns: Sequence[str]) -> None:
        sources = [
            foreign_key
            for foreign_key in table.foreign_keys
            if {name.casefold() for name in foreign_key.columns}
            <= {name.casefold() for name in columns}
        ]
        if not sources:
            for position, name in enumerate(columns):
                column = table.column(name)
                if position == 0:
                    self._allocate_distinct_column(table, column)
                else:
                    self._generate_column(table, column)
            return

        borrowed = {
            name.casefold() for source in sources for name in source.columns
        }
        discriminators = [
            table.column(name) for name in columns if name.casefold() not in borrowed
        ]
        self._allocate_borrowed_keys(table, sources, discriminators)

    def _allocate_borrowed_keys(
        self,
        table: TableSchema,
        sources: Sequence[ForeignKeySchema],
        discriminators: Sequence[ColumnSchema],
    ) -> None:
        """Fill key columns that a foreign key borrows from referenced tables.

        Without a local discriminator every row needs its own parent
        combination; with one, parent combinations may repeat and the
        discriminator keeps each key tuple unique.
        """
        rows = self._data[table.name]
        candidates = [self._parent_tuples(table, source) for source in sources]
        if discriminators:
            chosen = [
                tuple(self._random.choice(group) for group in candidates) for _ in rows
            ]
        else:
            chosen = self._distinct_combinations(table, candidates, len(rows))

        for row, combination in zip(rows, chosen):
            for source, values in zip(sources, combination):
                for name, value in zip(source.columns, values):
                    row[name] = value
        for source in sources:
            self._assigned[table.name].update(name.casefold() for name in source.columns)
        if discriminators:
            self._allocate_discriminators(table, discriminators, chosen)

    def _distinct_combinations(
        self,
        table: TableSchema,
        candidates: Sequence[Sequence[tuple[Any, ...]]],
        count: int,
    ) -> list[tuple[tuple[Any, ...], ...]]:
        capacity = 1
        for group in candidates:
            capacity *= len(group)
        if capacity < count:
            raise GenerationError(
                f"table {table.name!r} needs {count} distinct key combinations but "
                f"its referenced tables offer only {capacity}"
            )
        if count * 2 >= capacity:
            combinations = [tuple(item) for item in product(*candidates)]
            self._random.shuffle(combinations)
            return combinations[:count]
        return self._sample_distinct_combinations(candidates, count)

    def _allocate_discriminators(
        self,
        table: TableSchema,
        columns: Sequence[ColumnSchema],
        groups: Sequence[tuple[tuple[Any, ...], ...]],
    ) -> None:
        rows = self._data[table.name]
        specs = [self._column_spec(table, column) for column in columns]
        if len(columns) == 1 and columns[0].type.name == "INT":
            self._number_rows_per_group(table, columns[0], specs[0], groups)
        elif len(columns) == 1:
            taken: dict[tuple[Any, ...], set[Any]] = {}
            for index, (row, group) in enumerate(zip(rows, groups)):
                row[columns[0].name] = self._unique_value(
                    partial(self._value_for, table, columns[0], specs[0], index),
                    taken.setdefault(group, set()),
                    columns[0],
                )
        else:
            self._combine_discriminators(table, columns, specs, groups)
        self._mark_assigned(table, columns)

    def _number_rows_per_group(
        self,
        table: TableSchema,
        column: ColumnSchema,
        spec: _ColumnSpec,
        groups: Sequence[tuple[tuple[Any, ...], ...]],
    ) -> None:
        minimum, maximum = _integer_bounds(column, spec)
        minimum = max(minimum, 1)
        counters: dict[tuple[Any, ...], int] = {}
        for row, group in zip(self._data[table.name], groups):
            offset = counters.get(group, 0)
            counters[group] = offset + 1
            value = minimum + offset
            if value > maximum:
                raise GenerationError(
                    f"{table.name}.{column.name} cannot number {offset + 1} rows of one "
                    f"parent within its bounds of {minimum}..{maximum}"
                )
            row[column.name] = value

    def _combine_discriminators(
        self,
        table: TableSchema,
        columns: Sequence[ColumnSchema],
        specs: Sequence[_ColumnSpec],
        groups: Sequence[tuple[tuple[Any, ...], ...]],
    ) -> None:
        taken: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {}
        for index, (row, group) in enumerate(zip(self._data[table.name], groups)):
            seen = taken.setdefault(group, set())
            for _ in range(_UNIQUE_ATTEMPTS):
                values = tuple(
                    self._value_for(table, column, spec, index)
                    for column, spec in zip(columns, specs)
                )
                if values not in seen:
                    break
            else:
                raise GenerationError(
                    f"could not build distinct key columns "
                    f"({', '.join(column.name for column in columns)}) for {table.name}"
                )
            seen.add(values)
            for column, value in zip(columns, values):
                row[column.name] = value

    def _sample_distinct_combinations(
        self, candidates: Sequence[Sequence[tuple[Any, ...]]], count: int
    ) -> list[tuple[tuple[Any, ...], ...]]:
        seen: set[tuple[tuple[Any, ...], ...]] = set()
        chosen: list[tuple[tuple[Any, ...], ...]] = []
        attempts = 0
        while len(chosen) < count and attempts < count * 200:
            attempts += 1
            combination = tuple(self._random.choice(group) for group in candidates)
            if combination in seen:
                continue
            seen.add(combination)
            chosen.append(combination)
        if len(chosen) < count:
            raise GenerationError(
                "could not sample enough distinct key combinations from referenced tables"
            )
        return chosen

    def _parent_tuples(
        self, table: TableSchema, foreign_key: ForeignKeySchema
    ) -> list[tuple[Any, ...]]:
        parent_rows = self._data[foreign_key.referenced_table]
        tuples = [
            tuple(row.get(name, _MISSING) for name in foreign_key.referenced_columns)
            for row in parent_rows
        ]
        usable = [
            item
            for item in tuples
            if all(value is not _MISSING and value is not None for value in item)
        ]
        if not usable:
            raise GenerationError(
                f"table {table.name!r} cannot borrow keys from "
                f"{foreign_key.referenced_table!r}: no usable referenced values"
            )
        return usable

    def _allocate_distinct_column(self, table: TableSchema, column: ColumnSchema) -> None:
        rows = self._data[table.name]
        spec = self._column_spec(table, column)
        if column.type.name == "INT":
            minimum, maximum = _integer_bounds(column, spec)
            start = max(minimum, 1)
            if start + len(rows) - 1 > maximum:
                raise GenerationError(
                    f"{table.name}.{column.name} cannot allocate {len(rows)} sequential "
                    f"keys within its bounds of {start}..{maximum}"
                )
            for offset, row in enumerate(rows):
                row[column.name] = start + offset
            self._assigned[table.name].add(column.name.casefold())
            return

        seen: set[Any] = set()
        for index, row in enumerate(rows):
            row[column.name] = self._unique_value(
                lambda: self._value_for(table, column, spec, index), seen, column
            )
        self._assigned[table.name].add(column.name.casefold())

    def _generate_plain_columns(self) -> None:
        for table in self._schema.tables:
            order = self._orders[table.name]
            deferred = {
                name.casefold()
                for foreign_key in table.foreign_keys
                for name in foreign_key.columns
            }
            if order is not None:
                deferred.update(name.casefold() for name in order.columns)
            for column in table.columns:
                key = column.name.casefold()
                if key in self._assigned[table.name] or key in deferred:
                    continue
                self._generate_column(table, column)

    def _generate_column(self, table: TableSchema, column: ColumnSchema) -> None:
        spec = self._column_spec(table, column)
        seen: set[Any] = set()
        for index, row in enumerate(self._data[table.name]):
            row[column.name] = self._row_value(table, column, spec, index, seen)
        self._assigned[table.name].add(column.name.casefold())

    def _row_value(
        self,
        table: TableSchema,
        column: ColumnSchema,
        spec: _ColumnSpec,
        index: int,
        seen: set[Any],
    ) -> Any:
        if spec.null_probability and self._random.random() < spec.null_probability:
            return None
        if (
            spec.default is not _MISSING
            and not column.unique
            and self._random.random() < _DEFAULT_PROBABILITY
        ):
            return spec.default

        def produce() -> Any:
            return self._value_for(table, column, spec, index)

        if column.unique:
            return self._unique_value(produce, seen, column)
        return produce()

    def _unique_value(
        self, produce: Callable[[], Any], seen: set[Any], column: ColumnSchema
    ) -> Any:
        for _ in range(_UNIQUE_ATTEMPTS):
            value = produce()
            if value not in seen:
                seen.add(value)
                return value
        value = produce()
        for counter in range(1, _MUTATION_ATTEMPTS):
            candidate = _mutate(value, counter, column)
            if candidate is None:
                break
            if candidate not in seen:
                seen.add(candidate)
                return candidate
        raise GenerationError(
            f"could not produce a unique value for column {column.name!r}"
        )

    def _column_spec(self, table: TableSchema, column: ColumnSchema) -> _ColumnSpec:
        plan = self._plans.get((table.name.casefold(), column.name.casefold()))
        return _ColumnSpec(
            generator=_resolve_generator(table, column, plan),
            constraint=derive_column_constraint(table, column),
            plan=plan,
            locale=plan.locale if plan is not None else None,
            null_probability=(
                plan.nullable_probability if plan is not None and column.nullable else 0.0
            ),
            default=_default_python_value(column),
        )

    def _value_for(
        self, table: TableSchema, column: ColumnSchema, spec: _ColumnSpec, index: int
    ) -> Any:
        if column.type.name in _ORDERED_TYPES:
            domain = _ordered_domain(table, column, spec)
            return domain.value(self._pick_tick(domain, domain.low, spec, index))

        allowed = _allowed_values(table, column, spec)
        if allowed:
            return self._random.choice(allowed)
        if column.type.name == "ENUM":
            return self._random.choice(column.type.enum_values)
        if column.type.name == "BOOLEAN":
            return self._random.random() < 0.5
        return _truncate(self._text(column, spec, index), column.type.length)

    def _pick_tick(
        self, domain: _OrderedDomain, floor: int, spec: _ColumnSpec, index: int
    ) -> int:
        """Draw one tick of `domain` at or above `floor`.

        # Errors
        Raises `GenerationError` when no tick of the domain reaches `floor`.
        """
        low = max(domain.low, floor)
        sequential = spec.plan is not None and spec.plan.distribution == "sequential"
        if domain.allowed is not None:
            candidates = [tick for tick in domain.allowed if low <= tick <= domain.high]
            if not candidates:
                raise GenerationError(domain.empty_range_message())
            if sequential:
                return candidates[min(index, len(candidates) - 1)]
            return self._random.choice(candidates)
        if low > domain.high:
            raise GenerationError(domain.empty_range_message())
        if sequential:
            return min(low + index, domain.high)
        return self._random.randint(low, domain.high)

    def _text(self, column: ColumnSchema, spec: _ColumnSpec, index: int) -> str:
        faker = self._faker(spec.locale)
        generator = spec.generator
        if generator == "email":
            return faker.email()
        if generator == "first_name":
            return faker.first_name()
        if generator == "last_name":
            return faker.last_name()
        if generator == "full_name":
            return faker.name()
        if generator == "label":
            return " ".join(faker.words(nb=2)).title()
        if generator == "phone":
            return faker.numerify("###-###-####")
        if generator == "address":
            return faker.street_address()
        if generator == "city":
            return faker.city()
        if generator == "state":
            return faker.state_abbr() if _short(column, 20) else faker.state()
        if generator == "country":
            return faker.country()
        if generator == "zip_code":
            return faker.postcode()
        if generator == "company":
            return faker.company()
        if generator == "url":
            return faker.url()
        if generator == "job":
            return faker.job()
        if generator == "isbn":
            return faker.isbn13()
        if generator == "license":
            return faker.bothify("??######").upper()
        if generator == "opening_hours":
            opening = self._random.randint(6, 11)
            return f"{opening}:00 AM - {self._random.randint(5, 11)}:00 PM"
        if generator == "time":
            return f"{self._random.randint(0, 23):02d}:{self._random.randint(0, 59):02d}"
        if generator == "uuid":
            return faker.uuid4()
        if generator == "word":
            return faker.word()
        if generator == "sentence":
            return faker.sentence(nb_words=5).rstrip(".")
        if generator == "sequence":
            return f"{column.name}-{index + 1}"
        if generator == "text" or column.type.name == "TEXT":
            return faker.paragraph(nb_sentences=3)
        return faker.sentence(nb_words=4).rstrip(".")

    def _assign_foreign_keys(self) -> None:
        """Assign foreign keys in dependency order, deepest chains last.

        A referenced column may itself be a foreign key, so the order is found
        by repeatedly taking every foreign key whose referenced columns already
        hold values. Cycles that no nullable column can break are reported.
        """
        pending = [
            (table, foreign_key)
            for table in self._schema.tables
            for foreign_key in table.foreign_keys
            if not all(
                name.casefold() in self._assigned[table.name]
                for name in foreign_key.columns
            )
        ]
        while pending:
            ready = [item for item in pending if self._is_materialized(item[1])]
            if not ready:
                ready = [self._break_cycle(pending)]
            for table, foreign_key in ready:
                self._assign_foreign_key(table, foreign_key)
            assigned = {id(foreign_key) for _, foreign_key in ready}
            pending = [item for item in pending if id(item[1]) not in assigned]

    def _is_materialized(self, foreign_key: ForeignKeySchema) -> bool:
        assigned = self._assigned[foreign_key.referenced_table]
        return all(
            name.casefold() in assigned for name in foreign_key.referenced_columns
        )

    def _break_cycle(
        self, pending: Sequence[tuple[TableSchema, ForeignKeySchema]]
    ) -> tuple[TableSchema, ForeignKeySchema]:
        for table, foreign_key in pending:
            if all(table.column(name).nullable for name in foreign_key.columns):
                return table, foreign_key
        raise GenerationError(
            "cannot assign foreign keys: "
            + ", ".join(
                f"{table.name}({', '.join(foreign_key.columns)}) -> "
                f"{foreign_key.referenced_table}"
                f"({', '.join(foreign_key.referenced_columns)})"
                for table, foreign_key in pending
            )
            + " form a cycle of NOT NULL references to columns that are themselves "
            "foreign keys"
        )

    def _assign_foreign_key(
        self, table: TableSchema, foreign_key: ForeignKeySchema
    ) -> None:
        rows = self._data[table.name]
        columns = [table.column(name) for name in foreign_key.columns]
        nullable = all(column.nullable for column in columns)
        candidates = [
            item
            for item in (
                tuple(row.get(name, _MISSING) for name in foreign_key.referenced_columns)
                for row in self._data[foreign_key.referenced_table]
            )
            if all(value is not _MISSING and value is not None for value in item)
        ]

        if not candidates:
            if not nullable:
                raise GenerationError(
                    f"{table.name} requires a parent row in "
                    f"{foreign_key.referenced_table} but none is available"
                )
            for row in rows:
                for column in columns:
                    row[column.name] = None
            self._mark_assigned(table, columns)
            return

        distinct = _requires_distinct(table, foreign_key)
        if distinct and len(candidates) < len(rows):
            raise GenerationError(
                f"{table.name} needs {len(rows)} distinct references to "
                f"{foreign_key.referenced_table} but only {len(candidates)} exist"
            )
        ordered = list(candidates)
        if distinct:
            self._random.shuffle(ordered)
        null_probability = max(
            (
                self._null_probability(table, column)
                for column in columns
                if column.nullable
            ),
            default=0.0,
        )

        for index, row in enumerate(rows):
            if nullable and null_probability and self._random.random() < null_probability:
                for column in columns:
                    row[column.name] = None
                continue
            chosen = ordered[index] if distinct else self._random.choice(candidates)
            for column, value in zip(columns, chosen):
                row[column.name] = value
        self._mark_assigned(table, columns)

    def _generate_ordered_columns(self) -> None:
        """Generate every column a CHECK orders against another, lowest first.

        Each column is drawn inside the range left by its own bounds, the
        bounds propagated along the comparison graph, and the values already
        drawn for the row, so no value ever has to be repaired afterwards.
        """
        for table in self._schema.tables:
            order = self._orders[table.name]
            if order is None:
                continue
            columns = [table.column(name) for name in order.columns]
            specs = {column.name: self._column_spec(table, column) for column in columns}
            domains = _resolve_domains(
                table,
                order,
                {
                    column.name: _ordered_domain(table, column, specs[column.name])
                    for column in columns
                },
            )
            for index, row in enumerate(self._data[table.name]):
                self._fill_ordered_row(table, order, columns, specs, domains, index, row)
            self._mark_assigned(table, columns)

    def _fill_ordered_row(
        self,
        table: TableSchema,
        order: _ComparisonOrder,
        columns: Sequence[ColumnSchema],
        specs: dict[str, _ColumnSpec],
        domains: dict[str, _OrderedDomain],
        index: int,
        row: dict[str, Any],
    ) -> None:
        drawn: dict[str, int] = {}
        for column in columns:
            spec = specs[column.name]
            if spec.null_probability and self._random.random() < spec.null_probability:
                row[column.name] = None
                continue
            domain = domains[column.name]
            floor = max(
                (
                    drawn[lower] + (1 if strict else 0)
                    for lower, strict in order.predecessors(column.name)
                    if lower in drawn
                ),
                default=domain.low,
            )
            drawn[column.name] = self._pick_tick(domain, floor, spec, index)
            row[column.name] = domain.value(drawn[column.name])

    def _mark_assigned(self, table: TableSchema, columns: Iterable[ColumnSchema]) -> None:
        self._assigned[table.name].update(column.name.casefold() for column in columns)

    def _null_probability(self, table: TableSchema, column: ColumnSchema) -> float:
        plan = self._plans.get((table.name.casefold(), column.name.casefold()))
        return plan.nullable_probability if plan is not None else 0.0

    def _ordered_rows(self) -> GeneratedDataset:
        ordered: GeneratedDataset = {}
        for table in self._schema.tables:
            names = [column.name for column in table.columns]
            ordered[table.name] = [
                {name: row.get(name) for name in names} for row in self._data[table.name]
            ]
        return ordered


def _preflight(
    schema: DatabaseSchema,
    plan: DatabaseGenerationPlan | None,
    counts: dict[str, int],
) -> dict[str, _ComparisonOrder | None]:
    """Reject requests this generator cannot satisfy before any row is built.

    ## Returns
    The column comparison order of every table, keyed by table name.

    # Errors
    Raises `GenerationError` for unsupported check constraints, irreconcilable
    foreign keys, unsatisfiable unique domains, or plan hints outside the
    supported vocabulary.
    """
    _reject_unsupported_checks(schema)
    orders: dict[str, _ComparisonOrder | None] = {}
    for table in schema.tables:
        _reject_unsupported_comparisons(table)
        _reject_overlapping_foreign_keys(table)
        _reject_exhausted_unique_domains(table, counts[table.name])
        orders[table.name] = _comparison_order(table)
    _reject_unsupported_plan(schema, plan)
    return orders


def _reject_unsupported_checks(schema: DatabaseSchema) -> None:
    unsupported = [
        violation
        for table in schema.tables
        for violation in unsupported_check_violations(table)
    ]
    if unsupported:
        raise GenerationError(
            "cannot generate data for an unsupported check constraint: "
            + "; ".join(violation.message for violation in unsupported)
        )


def _reject_unsupported_comparisons(table: TableSchema) -> None:
    for comparison in column_comparisons(table):
        reason = _comparison_problem(table, comparison)
        if reason is not None:
            raise GenerationError(
                f"cannot generate data for {table.name} CHECK "
                f"({comparison.expression}): {reason}"
            )


def _comparison_problem(table: TableSchema, comparison: ColumnComparison) -> str | None:
    if not comparison.conjunct:
        return "column comparisons under OR/NOT are not supported"
    if comparison.operator not in _ORDERING_OPERATORS:
        return f"operator {comparison.operator!r} between two columns is not supported"
    left = table.column(comparison.left)
    right = table.column(comparison.right)
    if _is_constrained_key(table, left) or _is_constrained_key(table, right):
        return "comparisons over a key column or a unique column are not supported"
    if left.type != right.type:
        return f"{left.name} and {right.name} do not share one comparable SQL type"
    if left.type.name not in _ORDERED_TYPES:
        return f"SQL type {left.type.name} cannot be ordered across columns"
    return None


def _is_constrained_key(table: TableSchema, column: ColumnSchema) -> bool:
    key = column.name.casefold()
    if column.unique:
        return True
    primary = table.primary_key.columns if table.primary_key is not None else []
    names = {name.casefold() for name in primary}
    names.update(
        name.casefold()
        for constraint in table.unique_constraints
        for name in constraint.columns
    )
    names.update(
        name.casefold()
        for foreign_key in table.foreign_keys
        for name in foreign_key.columns
    )
    return key in names


def _reject_overlapping_foreign_keys(table: TableSchema) -> None:
    for first, second in combinations(table.foreign_keys, 2):
        shared = sorted(
            {name.casefold() for name in first.columns}
            & {name.casefold() for name in second.columns}
        )
        if not shared or _references_the_same_tuple(first, second):
            continue
        raise GenerationError(
            f"table {table.name!r} has overlapping foreign keys on "
            f"{', '.join(shared)} that cannot be reconciled: "
            f"{_reference_text(first)} and {_reference_text(second)}"
        )


def _references_the_same_tuple(
    first: ForeignKeySchema, second: ForeignKeySchema
) -> bool:
    return (
        first.referenced_table.casefold() == second.referenced_table.casefold()
        and [name.casefold() for name in first.columns]
        == [name.casefold() for name in second.columns]
        and [name.casefold() for name in first.referenced_columns]
        == [name.casefold() for name in second.referenced_columns]
    )


def _reference_text(foreign_key: ForeignKeySchema) -> str:
    return (
        f"({', '.join(foreign_key.columns)}) -> {foreign_key.referenced_table}"
        f"({', '.join(foreign_key.referenced_columns)})"
    )


def _reject_exhausted_unique_domains(table: TableSchema, count: int) -> None:
    for columns in _unique_column_groups(table):
        capacity = 1
        for name in columns:
            size = _finite_domain_size(table, table.column(name))
            if size is None:
                break
            capacity *= size
        else:
            if capacity < count:
                raise GenerationError(
                    f"{table.name} ({', '.join(columns)}) needs {count} distinct "
                    f"values but its domain offers only {capacity}"
                )


def _unique_column_groups(table: TableSchema) -> list[list[str]]:
    groups = [constraint.columns for constraint in table.unique_constraints]
    if table.primary_key is not None:
        groups.append(table.primary_key.columns)
    return groups


def _finite_domain_size(table: TableSchema, column: ColumnSchema) -> int | None:
    if column.type.name == "ENUM":
        return len(column.type.enum_values)
    if column.type.name == "BOOLEAN":
        return 2
    allowed = derive_column_constraint(table, column).allowed
    return None if allowed is None else len(allowed)


def _reject_unsupported_plan(
    schema: DatabaseSchema, plan: DatabaseGenerationPlan | None
) -> None:
    if plan is None:
        return
    for table_plan in plan.tables:
        table = schema.table(table_plan.table_name)
        for column_plan in table_plan.columns:
            column = table.column(column_plan.column_name)
            _reject_unsupported_column_plan(table, column, column_plan)


def _reject_unsupported_column_plan(
    table: TableSchema, column: ColumnSchema, plan: ColumnGenerationPlan
) -> None:
    location = f"{table.name}.{column.name}"
    if plan.distribution not in _SUPPORTED_DISTRIBUTIONS:
        raise GenerationError(
            f"{location} asks for the {plan.distribution!r} distribution, which is not "
            f"supported; use one of {', '.join(sorted(_SUPPORTED_DISTRIBUTIONS))}"
        )
    for hint in (plan.generator, plan.semantic_type):
        token = _normalize(hint)
        if token and token not in _GENERATOR_ALIASES and token not in _TYPE_GENERATORS:
            raise GenerationError(
                f"{location} asks for the unknown generator {hint!r}"
            )
    if not plan.categories:
        return
    if column.type.name == "ENUM":
        unknown = [
            category
            for category in plan.categories
            if category not in column.type.enum_values
        ]
        if unknown:
            raise GenerationError(
                f"{location} plans categories outside its ENUM domain: "
                f"{', '.join(repr(value) for value in unknown)}"
            )
    elif column.type.name not in {"VARCHAR", "TEXT"}:
        raise GenerationError(
            f"{location} plans categories, which only apply to ENUM and text columns, "
            f"not to SQL type {column.type.name}"
        )


@dataclass(frozen=True)
class _ComparisonOrder:
    """The column order that the column-to-column CHECKs of one table impose.

    `edges` holds `(lower, upper, strict)` triples; `columns` is a topological
    order of them, tie-broken by declaration order so it never depends on how
    the CHECKs were written.
    """

    columns: tuple[str, ...]
    edges: tuple[tuple[str, str, bool], ...]

    def predecessors(self, name: str) -> tuple[tuple[str, bool], ...]:
        return tuple((lower, strict) for lower, upper, strict in self.edges if upper == name)

    def successors(self, name: str) -> tuple[tuple[str, bool], ...]:
        return tuple((upper, strict) for lower, upper, strict in self.edges if lower == name)


@dataclass(frozen=True)
class _OrderedDomain:
    """An orderable column domain projected onto integer ticks of one step.

    One tick is 1 for INT, the smallest representable step for DECIMAL, one day
    for DATE, and one second for DATETIME, so a strict comparison is always
    satisfied by a single tick of separation.
    """

    table: str
    column: ColumnSchema
    low: int
    high: int
    allowed: tuple[int, ...] | None

    def value(self, tick: int) -> Any:
        name = self.column.type.name
        if name == "INT":
            return tick
        if name == "DECIMAL":
            unit = _decimal_unit(self.column)
            return (Decimal(tick) * unit).quantize(unit)
        if name == "DATE":
            return date.fromordinal(tick)
        return _EPOCH + timedelta(seconds=tick)

    def empty_range_message(self) -> str:
        return (
            f"{self.table}.{self.column.name} has an empty range after applying its "
            "checks, plan bounds, and declared type"
        )


def _comparison_order(table: TableSchema) -> _ComparisonOrder | None:
    """Return the generation order implied by the column comparisons of `table`.

    ## Returns
    `None` when no CHECK compares two columns of the table.

    # Errors
    Raises `GenerationError` when the comparisons cannot be ordered.
    """
    edges: set[tuple[str, str, bool]] = set()
    for comparison in column_comparisons(table):
        lower, upper = comparison.left, comparison.right
        if comparison.operator in {">", ">="}:
            lower, upper = upper, lower
        lower, upper = table.column(lower).name, table.column(upper).name
        strict = comparison.operator in {"<", ">"}
        if lower == upper:
            if strict:
                raise GenerationError(
                    f"cannot generate data for {table.name} CHECK "
                    f"({comparison.expression}): a column cannot be strictly ordered "
                    "against itself"
                )
            continue
        edges.add((lower, upper, strict))
    if not edges:
        return None
    return _ComparisonOrder(columns=_ordered_columns(table, edges), edges=tuple(sorted(edges)))


def _ordered_columns(
    table: TableSchema, edges: set[tuple[str, str, bool]]
) -> tuple[str, ...]:
    position = {column.name: index for index, column in enumerate(table.columns)}
    waiting = {
        name: {lower for lower, upper, _ in edges if upper == name}
        for edge in edges
        for name in edge[:2]
    }
    ordered: list[str] = []
    while waiting:
        ready = sorted(
            (name for name, pending in waiting.items() if not pending),
            key=position.__getitem__,
        )
        if not ready:
            raise GenerationError(
                f"cannot generate data for {table.name}: the column comparisons over "
                f"{', '.join(sorted(waiting))} form a cycle"
            )
        for name in ready:
            ordered.append(name)
            del waiting[name]
        for pending in waiting.values():
            pending.difference_update(ready)
    return tuple(ordered)


def _resolve_domains(
    table: TableSchema,
    order: _ComparisonOrder,
    domains: dict[str, _OrderedDomain],
) -> dict[str, _OrderedDomain]:
    """Propagate each column's bounds along the comparison graph, both ways.

    # Errors
    Raises `GenerationError` when propagation leaves a column with no value.
    """
    low = {name: domains[name].low for name in order.columns}
    high = {name: domains[name].high for name in order.columns}
    for name in order.columns:
        for lower, strict in order.predecessors(name):
            low[name] = max(low[name], low[lower] + (1 if strict else 0))
    for name in reversed(order.columns):
        for upper, strict in order.successors(name):
            high[name] = min(high[name], high[upper] - (1 if strict else 0))

    resolved: dict[str, _OrderedDomain] = {}
    for name in order.columns:
        if low[name] > high[name]:
            raise GenerationError(
                f"{table.name}.{name} has an empty range after combining its own "
                f"bounds with the column comparisons of {table.name}"
            )
        resolved[name] = replace(domains[name], low=low[name], high=high[name])
    return resolved


def _ordered_domain(
    table: TableSchema, column: ColumnSchema, spec: _ColumnSpec
) -> _OrderedDomain:
    """Build the tick domain of one INT, DECIMAL, DATE, or DATETIME column.

    # Errors
    Raises `GenerationError` when checks and plan bounds leave no value.
    """
    if column.type.name == "INT":
        low, high = _integer_bounds(column, spec)
    elif column.type.name == "DECIMAL":
        low, high = _decimal_tick_bounds(column, spec)
    else:
        low, high = _temporal_tick_bounds(column, spec)

    values = _allowed_values(table, column, spec)
    allowed = tuple(sorted(_tick(value, column) for value in values)) if values else None
    if allowed is not None:
        low, high = max(low, allowed[0]), min(high, allowed[-1])
    domain = _OrderedDomain(
        table=table.name, column=column, low=low, high=high, allowed=allowed
    )
    if low > high:
        raise GenerationError(domain.empty_range_message())
    return domain


def _decimal_tick_bounds(column: ColumnSchema, spec: _ColumnSpec) -> tuple[int, int]:
    unit = _decimal_unit(column)
    limit = Decimal(10) ** ((column.type.precision or 10) - (column.type.scale or 0)) - unit
    minimum, maximum = _decimal_bounds(spec, unit, -limit, limit)
    return (
        int((minimum / unit).to_integral_value(ROUND_CEILING)),
        int((maximum / unit).to_integral_value(ROUND_FLOOR)),
    )


def _temporal_tick_bounds(column: ColumnSchema, spec: _ColumnSpec) -> tuple[int, int]:
    constraint = spec.constraint
    minimum: int | None = None
    maximum: int | None = None
    if isinstance(constraint.minimum, date):
        minimum = _tick(constraint.minimum, column) + int(constraint.minimum_exclusive)
    if isinstance(constraint.maximum, date):
        maximum = _tick(constraint.maximum, column) - int(constraint.maximum_exclusive)

    planned_start = _plan_date(spec.plan.minimum if spec.plan else None)
    planned_end = _plan_date(spec.plan.maximum if spec.plan else None)
    if planned_start is not None:
        planned = _tick(planned_start, column)
        minimum = planned if minimum is None else max(minimum, planned)
    if planned_end is not None:
        planned = _tick(_end_of_day(planned_end, column), column)
        maximum = planned if maximum is None else min(maximum, planned)

    start, end = _default_temporal_window(column)
    return _open_bounds(minimum, maximum, _tick(start, column), _tick(end, column))


def _default_temporal_window(column: ColumnSchema) -> tuple[date, date]:
    if column.type.name == "DATE":
        return _RANGE_START.date(), _RANGE_END.date()
    if _is_current_timestamp(column):
        return _REFERENCE_NOW - timedelta(days=365), _REFERENCE_NOW
    return _RANGE_START, _RANGE_END


def _end_of_day(value: date, column: ColumnSchema) -> date:
    if column.type.name == "DATE":
        return value
    return datetime.combine(value, datetime.max.time().replace(microsecond=0))


def _tick(value: Any, column: ColumnSchema) -> int:
    name = column.type.name
    if name == "INT":
        return int(value)
    if name == "DECIMAL":
        return int((value / _decimal_unit(column)).to_integral_value(ROUND_FLOOR))
    if name == "DATE":
        return value.toordinal()
    moment = value if isinstance(value, datetime) else datetime.combine(
        value, datetime.min.time()
    )
    return int((moment - _EPOCH).total_seconds())


def _decimal_unit(column: ColumnSchema) -> Decimal:
    return Decimal(1).scaleb(-(column.type.scale or 0))


def _plans_by_column(
    plan: DatabaseGenerationPlan | None,
) -> dict[tuple[str, str], ColumnGenerationPlan]:
    if plan is None:
        return {}
    return {
        (table.table_name.casefold(), column.column_name.casefold()): column
        for table in plan.tables
        for column in table.columns
    }


_GENERATOR_ALIASES = {
    "name": "full_name",
    "full_name": "full_name",
    "person_name": "full_name",
    "first_name": "first_name",
    "given_name": "first_name",
    "last_name": "last_name",
    "surname": "last_name",
    "family_name": "last_name",
    "email": "email",
    "email_address": "email",
    "phone": "phone",
    "phone_number": "phone",
    "telephone": "phone",
    "address": "address",
    "street_address": "address",
    "city": "city",
    "state": "state",
    "state_abbr": "state",
    "province": "state",
    "country": "country",
    "zip": "zip_code",
    "zip_code": "zip_code",
    "postcode": "zip_code",
    "postal_code": "zip_code",
    "company": "company",
    "company_name": "company",
    "employer": "company",
    "organization": "company",
    "url": "url",
    "website": "url",
    "uri": "url",
    "link": "url",
    "text": "text",
    "paragraph": "text",
    "description": "text",
    "sentence": "sentence",
    "title": "sentence",
    "word": "word",
    "job": "job",
    "job_title": "job",
    "isbn": "isbn",
    "label": "label",
    "license": "license",
    "uuid": "uuid",
    "time": "time",
    "sequence": "sequence",
}

_PERSON_WORDS = (
    "applicant",
    "author",
    "client",
    "contact",
    "customer",
    "driver",
    "employee",
    "full_name",
    "guest",
    "manager",
    "member",
    "owner",
    "patient",
    "people",
    "person",
    "staff",
    "student",
    "subscriber",
    "teacher",
    "user",
)

_NAME_HINTS = (
    ("email", "email"),
    ("first_name", "first_name"),
    ("given_name", "first_name"),
    ("middle_name", "first_name"),
    ("last_name", "last_name"),
    ("surname", "last_name"),
    ("phone", "phone"),
    ("fax", "phone"),
    ("zip", "zip_code"),
    ("postal", "zip_code"),
    ("postcode", "zip_code"),
    ("address", "address"),
    ("city", "city"),
    ("state", "state"),
    ("country", "country"),
    ("job_title", "job"),
    ("company", "company"),
    ("employer", "company"),
    ("website", "url"),
    ("url", "url"),
    ("isbn", "isbn"),
    ("license", "license"),
    ("opening_hours", "opening_hours"),
    ("description", "text"),
    ("comment", "text"),
    ("biography", "text"),
    ("note", "text"),
    ("instruction", "text"),
    ("review_text", "text"),
    ("title", "sentence"),
    ("name", "full_name"),
)


def _resolve_generator(
    table: TableSchema, column: ColumnSchema, plan: ColumnGenerationPlan | None
) -> str:
    if plan is not None:
        for hint in (plan.generator, plan.semantic_type):
            token = _GENERATOR_ALIASES.get(_normalize(hint))
            if token is not None:
                return token
    key = column.name.casefold()
    for fragment, token in _NAME_HINTS:
        if fragment not in key:
            continue
        if token == "full_name" and not _describes_people(key, table.name.casefold()):
            return "label"
        return token
    return ""


def _describes_people(*names: str) -> bool:
    return any(word in name for name in names for word in _PERSON_WORDS)


def _normalize(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().casefold().replace(" ", "_").replace("-", "_")


def _allowed_values(
    table: TableSchema, column: ColumnSchema, spec: _ColumnSpec
) -> list[Any]:
    """Return the values a column may take, narrowed by CHECKs and plan categories.

    # Errors
    Raises `GenerationError` when the narrowed domain is empty.
    """
    domains = [
        candidates
        for candidates in (spec.constraint.allowed, _planned_categories(spec))
        if candidates is not None
    ]
    if not domains:
        return []
    values = [
        coerced
        for candidate in domains[0]
        if (coerced := _coerce(candidate, column)) is not _MISSING
        and all(candidate in domain for domain in domains[1:])
    ]
    if not values:
        raise GenerationError(
            f"no value satisfies the declared domain of {table.name}.{column.name}"
        )
    return values


def _planned_categories(spec: _ColumnSpec) -> tuple[str, ...] | None:
    if spec.plan is None or not spec.plan.categories:
        return None
    return tuple(spec.plan.categories)


def _coerce(value: Any, column: ColumnSchema) -> Any:
    type_name = column.type.name
    if type_name == "INT" and isinstance(value, Decimal):
        return int(value)
    if type_name == "DECIMAL" and isinstance(value, Decimal):
        return value.quantize(Decimal(1).scaleb(-(column.type.scale or 0)))
    if type_name in {"VARCHAR", "TEXT"} and isinstance(value, str):
        return value
    if type_name == "ENUM" and value in column.type.enum_values:
        return value
    if type_name == "BOOLEAN" and isinstance(value, bool):
        return value
    if type_name == "DATE" and isinstance(value, date) and not isinstance(value, datetime):
        return value
    if type_name == "DATETIME" and isinstance(value, datetime):
        return value
    return _MISSING


def _integer_bounds(column: ColumnSchema, spec: _ColumnSpec) -> tuple[int, int]:
    constraint = spec.constraint
    minimum: int | None = None
    maximum: int | None = None
    if isinstance(constraint.minimum, Decimal):
        bound = int(constraint.minimum.to_integral_value(ROUND_CEILING))
        minimum = bound + 1 if constraint.minimum_exclusive else bound
    if isinstance(constraint.maximum, Decimal):
        bound = int(constraint.maximum.to_integral_value(ROUND_FLOOR))
        maximum = bound - 1 if constraint.maximum_exclusive else bound
    if spec.plan is not None:
        if isinstance(spec.plan.minimum, (int, float)):
            planned = int(spec.plan.minimum)
            minimum = planned if minimum is None else max(minimum, planned)
        if isinstance(spec.plan.maximum, (int, float)):
            planned = int(spec.plan.maximum)
            maximum = planned if maximum is None else min(maximum, planned)
    return _open_bounds(minimum, maximum, _INT_MINIMUM, _INT_MAXIMUM)


def _decimal_bounds(
    spec: _ColumnSpec, unit: Decimal, floor: Decimal, ceiling: Decimal
) -> tuple[Decimal, Decimal]:
    constraint = spec.constraint
    minimum: Decimal | None = None
    maximum: Decimal | None = None
    if isinstance(constraint.minimum, Decimal):
        minimum = constraint.minimum + (unit if constraint.minimum_exclusive else 0)
    if isinstance(constraint.maximum, Decimal):
        maximum = constraint.maximum - (unit if constraint.maximum_exclusive else 0)
    if spec.plan is not None:
        if isinstance(spec.plan.minimum, (int, float)):
            planned = Decimal(str(spec.plan.minimum))
            minimum = planned if minimum is None else max(minimum, planned)
        if isinstance(spec.plan.maximum, (int, float)):
            planned = Decimal(str(spec.plan.maximum))
            maximum = planned if maximum is None else min(maximum, planned)
    low, high = _open_bounds(minimum, maximum, Decimal(0), _DECIMAL_MAXIMUM)
    return max(low, floor), min(high, ceiling)


def _open_bounds(
    minimum: _Number | None,
    maximum: _Number | None,
    floor: _Number,
    ceiling: _Number,
) -> tuple[_Number, _Number]:
    """Fill in whichever numeric bound the schema and plan leave open.

    The preferred `floor`..`ceiling` window is shifted, never narrowed, when the
    declared bound falls outside it, so negative CHECK and plan bounds keep a
    span to draw from.
    """
    span = ceiling - floor
    if minimum is None:
        minimum = floor if maximum is None or maximum >= floor else maximum - span
    if maximum is None:
        maximum = ceiling if minimum <= ceiling else minimum + span
    return minimum, maximum


def _plan_date(value: float | str | None) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _is_current_timestamp(column: ColumnSchema) -> bool:
    return (column.default or "").upper() == "CURRENT_TIMESTAMP"


def _default_python_value(column: ColumnSchema) -> Any:
    raw = column.default
    if raw is None or _is_current_timestamp(column):
        return _MISSING
    text = raw
    if len(text) >= 2 and text[0] in {"'", '"'} and text[-1] == text[0]:
        text = text[1:-1]
    elif text.upper() == "NULL":
        return None if column.nullable else _MISSING

    type_name = column.type.name
    try:
        if type_name == "BOOLEAN":
            upper = text.upper()
            return upper in {"TRUE", "1"} if upper in {"TRUE", "FALSE", "0", "1"} else _MISSING
        if type_name == "INT":
            return int(text)
        if type_name == "DECIMAL":
            return Decimal(text).quantize(Decimal(1).scaleb(-(column.type.scale or 0)))
        if type_name == "DATE":
            return date.fromisoformat(text)
        if type_name == "DATETIME":
            return datetime.fromisoformat(text)
    except (ValueError, InvalidOperation):
        return _MISSING

    if type_name == "ENUM":
        return text if text in column.type.enum_values else _MISSING
    if type_name in {"VARCHAR", "TEXT"}:
        return _truncate(text, column.type.length)
    return _MISSING


def _truncate(value: str, length: int | None) -> str:
    return value if length is None or len(value) <= length else value[:length]


def _short(column: ColumnSchema, threshold: int) -> bool:
    return column.type.length is not None and column.type.length < threshold


def _mutate(value: Any, counter: int, column: ColumnSchema) -> Any:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        suffix = str(counter)
        length = column.type.length
        if length is None:
            return value + suffix
        if length <= len(suffix):
            return None
        return value[: length - len(suffix)] + suffix
    if isinstance(value, int):
        return value + counter
    if isinstance(value, Decimal):
        return value + Decimal(counter).scaleb(-(column.type.scale or 0))
    if isinstance(value, datetime):
        return value + timedelta(seconds=counter)
    if isinstance(value, date):
        return value + timedelta(days=counter)
    return None


def _requires_distinct(table: TableSchema, foreign_key: ForeignKeySchema) -> bool:
    columns = {name.casefold() for name in foreign_key.columns}
    if table.primary_key is not None and columns == {
        name.casefold() for name in table.primary_key.columns
    }:
        return True
    return any(
        columns == {name.casefold() for name in constraint.columns}
        for constraint in table.unique_constraints
    )
