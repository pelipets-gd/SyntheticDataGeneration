# Synthetic Data Assistant

Dockerized Streamlit app that generates constraint-safe synthetic relational data from MySQL-style DDL using Gemini on Vertex AI and PostgreSQL storage, then answers natural-language questions with a compiled read-only SELECT. It includes settings, health checks, DDL parsing, structured Gemini planning, NL query compilation, safe feedback-operation contracts, deterministic data generation, independent dataset validation, and PostgreSQL dataset persistence.

Configuration is read from **process environment variables** and **Google Application Default Credentials (ADC)**. Do not add `.env` or `.env.*` files to this project.

## Prerequisites

- Docker and Docker Compose
- Python 3.12+ (for local unit tests)
- [Google Cloud SDK](https://cloud.google.com/sdk) with access to a GCP project that can call Vertex AI (Gemini)
- Optional: [Langfuse](https://langfuse.com/) project keys for tracing

## Google Cloud / Vertex AI authentication

Authenticate with ADC on your machine (credentials stay outside the repo):

```bash
gcloud auth application-default login
```

Export your GCP project id before starting Docker Compose:

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
```

Optional overrides:

| Variable | Purpose | Default |
|----------|---------|---------|
| `VERTEX_AI_LOCATION` | Vertex AI region | `us-central1` |
| `GEMINI_MODEL` | Gemini model id | `gemini-2.5-flash` |

The app container sets `GOOGLE_GENAI_USE_VERTEXAI=true` and mounts your local gcloud config read-only so the `google-genai` client can use ADC. Override the mount with `ADC_MOUNT` if your ADC files live elsewhere.

## PostgreSQL (Docker Compose)

Compose defines `postgres` and `app` services. Postgres credentials are **local dev placeholders** (`data_assistant` / `data_assistant_dev`) — suitable for a laptop only, not production secrets.

| Variable | Purpose | Default |
|----------|---------|---------|
| `POSTGRES_USER` | DB user | `data_assistant` |
| `POSTGRES_PASSWORD` | DB password | `data_assistant_dev` |
| `POSTGRES_DB` | Database name | `data_assistant` |
| `POSTGRES_PORT` | Host port mapping | `5432` |
| `STREAMLIT_PORT` | Streamlit host port | `8501` |
| `DATABASE_URL` | App DB URL (set automatically in Compose) | built from Postgres vars |

## Optional Langfuse

Tracing is enabled only when both keys are set:

| Variable | Purpose |
|----------|---------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse API host (e.g. cloud or self-hosted URL) |

Leave these unset for local runs without observability.

## Run with Docker

From this directory:

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
docker compose up --build
```

Open Streamlit at [http://localhost:8501](http://localhost:8501) (or the port you mapped).

## Data generation and validation

Two modules turn a parsed schema, plus an optional Gemini plan, into rows and then prove those rows satisfy the schema.

`generator_vocabulary.py` holds the generator names and distributions both sides agree on. The planning prompt lists them, and any name the model invents anyway is replaced with `auto` before the plan reaches the generator, so one hallucinated hint falls back to column-name inference instead of failing the run.

### `generation.py` — `SyntheticDataGenerator`

`generate(schema, plan=None, row_count=None, seed=0)` returns `{table_name: [row, ...]}`. The same seed always produces the same dataset, and the generator validates its own output before returning; anything that cannot be produced safely raises `GenerationError` instead of emitting invalid rows.

The build runs in fixed phases:

1. **Preflight** rejects, before any row exists: CHECK expressions outside the supported grammar, CHECK literals that do not fit their column's type, comparisons between columns of unrelated types, column comparisons over key or unique columns, comparison chains that form a cycle or leave a column with an empty range, any unique constraint — single-column or composite — whose finite domain is smaller than the requested row count, overlapping composite foreign keys that cannot be reconciled, plan distributions other than `uniform` and `sequential`, unknown generator or semantic-type hints, and plan categories outside a column's ENUM domain.
2. **Primary keys** are allocated first, tables ordered so a key that borrows from a foreign key is filled after its parent. Integer keys are numbered sequentially from the lowest value the CHECK bounds allow. A composite key built from foreign keys draws distinct parent combinations; when the key also carries a local discriminator column, parent rows may repeat and the discriminator is numbered per parent, so a child table can hold far more rows than its parent.
3. **Plain columns** are drawn from the column's declared domain: CHECK bounds and `IN` lists (intersected across every CHECK on the column and its table), plan minimum/maximum, categories, locale and null probability, column defaults, and a semantic generator resolved from the plan or the column name. Negative bounds are honoured on both the CHECK and the plan side.
4. **Foreign keys** are assigned by fixed point: every key whose referenced columns already hold values is filled, and the pass repeats, so chains of arbitrary depth resolve even when they run through non-primary UNIQUE columns. A remaining cycle is broken through a nullable column; when no nullable column can break it, `GenerationError` names every edge of the cycle.
5. **Column comparisons** such as `CHECK (end_date >= start_date)` or `CHECK (a <= b)` plus `CHECK (b <= c)` are satisfied by construction rather than by repairing finished rows. The comparisons of a table form a graph, its columns are sorted topologically, and each column's bounds are propagated along the edges; every value is then drawn inside the range left by its own CHECK bounds, the propagated bounds, and the values already drawn for the row. INT, DECIMAL, DATE, and DATETIME share one tick-based domain, so chains of any length hold for every seed and for any order the CHECKs are declared in.

### `validation.py` — independent verification

`validate_dataset(schema, data)` re-checks a dataset without trusting the generator and returns every `Violation` it finds; `assert_valid_dataset` raises `DataValidationError` carrying the same list. Covered: missing or unknown tables and columns, row shape, NOT NULL, Python/SQL type match, VARCHAR length, DECIMAL precision and scale, ENUM membership, CHECK results, primary-key nullability and uniqueness, UNIQUE constraints, and foreign-key targets.

CHECK expressions are parsed into a small expression tree and evaluated in Python — SQL is never executed. Literals are coerced to their column's type while the expression is parsed, so `CHECK (start_date >= '2024-01-01')` compares dates rather than a date to a string; a literal that cannot be coerced, and a comparison between columns of unrelated types, are reported as `unsupported_check`. NULL follows three-valued logic, so an unknown result never counts as a failure, but a comparison that cannot be evaluated at all — a value whose Python type does not match its column, for instance — is reported as `check_unevaluable` instead of quietly passing as SQL-unknown. Any expression outside the supported grammar (arithmetic, function calls, unknown columns) is reported as an `unsupported_check` violation rather than silently treated as satisfied, which is also what makes the generator refuse such a schema during preflight.

## Storage architecture

### `storage.py` — `DatasetRepository`

`DatasetRepository(connect=...)` persists generated datasets in PostgreSQL through psycopg 3. The connection factory is injected (same shape as the health check), so the repository never reads configuration itself and can be driven by a fake connection in tests.

**Layout.** `initialize()` creates one app-owned metadata table, `data_assistant_datasets` (`dataset_id`, `name`, `schema_name`, `ddl`, `instructions`, `schema_json`, `created_at`). Each saved dataset then gets its own PostgreSQL schema named `dataset_<uuid>` — generated by the app, never derived from uploaded text — so datasets are isolated and a table called `users` in one upload cannot collide with another. Every identifier is composed with `psycopg.sql.Identifier` and every value is bound as a parameter; uploaded table, column, and constraint names are never interpolated into SQL.

**Type translation.** The normalized MySQL types map to PostgreSQL as:

| Normalized | PostgreSQL |
|------------|------------|
| `INT` | `integer` |
| `VARCHAR(n)` | `varchar(n)` |
| `TEXT` | `text` |
| `DATE` | `date` |
| `DATETIME` | `timestamp` |
| `DECIMAL(p, s)` | `numeric(p, s)` |
| `BOOLEAN` | `boolean` |
| `ENUM(...)` | `text` plus a `CHECK (col IN (...))` |

NOT NULL, primary keys (including composite), single and composite UNIQUE constraints, and foreign-key `ON DELETE` / `ON UPDATE` actions are preserved. Defaults are emitted only when they port cleanly — a quoted literal for text and ENUM columns, a number for `INT` and `DECIMAL`, a boolean word, an ISO date or timestamp, and `CURRENT_TIMESTAMP` — so a MySQL-only default such as `UUID()` is dropped rather than mistranslated. CHECK constraints are re-emitted from the same safe grammar `validation.py` parses: the expression is re-tokenized, identifiers are resolved against the table and quoted, literals are bound, and anything outside the grammar (arithmetic, function calls, unknown columns) is left out instead of being copied verbatim. `AUTO_INCREMENT` is not translated because every row carries its generated key.

**Saving.** `save_dataset(name=..., ddl=..., schema=..., data=..., instructions=...)` validates the data with `assert_valid_dataset` **before** opening a connection, then runs one transaction: `CREATE SCHEMA` → `CREATE TABLE` for each table *without* foreign keys → bulk `INSERT` of every row → `ALTER TABLE ... ADD FOREIGN KEY` → the metadata row. Deferring foreign keys to the end is what lets forward references and circular references (`Team.lead_id → Member`, `Member.team_id → Team`) load without ordering games. Any failure rolls the whole thing back, so a partial save leaves neither a metadata row nor a dataset schema.

**Reading.** `list_datasets()` returns the stored summaries, so datasets survive a restart; `load_dataset(id)` returns the human name, original DDL, instructions, and the normalized `DatabaseSchema` rebuilt from JSON; `list_tables(id)` and `read_table(id, table, limit=..., offset=...)` page rows in primary-key order with the limit and offset bound as parameters.

**Updating.** `update_table(id, table, rows)` runs in one transaction: it reads the whole stored dataset, substitutes the edited table, and validates the complete reconstruction before touching anything. It then requires the primary-key values to be unchanged and issues `UPDATE ... WHERE <primary key>` for the rows that actually differ. Nothing is truncated or deleted, so dependent rows in other tables are never cascaded away.

**Errors.** Failures surface as `StorageError`, `DatasetNotFoundError`, or `DatasetValidationError` (which carries the `Violation` list). Driver exceptions are replaced rather than chained, so connection strings, passwords, and raw server messages never reach the caller.

## Local tests

Install dependencies and run pytest (package root is `src/` via `pyproject.toml`):

```bash
python3.12 -m pip install -r requirements.txt
python3.12 -m pytest tests/ -q
```

Settings tests set environment variables in-process; no credential files are required for unit tests. The storage unit tests drive a recording fake connection, so they assert statement order, identifier quoting, and rollback behaviour without a database.

### Storage integration tests (optional)

`tests/test_storage_integration.py` exercises a real PostgreSQL. It is marked `postgres` and skips unless `TEST_DATABASE_URL` is passed explicitly, so the normal suite needs no database and no credentials:

```bash
TEST_DATABASE_URL="postgresql://data_assistant:data_assistant_dev@127.0.0.1:5432/data_assistant" \
  python3.12 -m pytest tests/test_storage_integration.py -m postgres -q
```

Start the database first with `docker compose up -d postgres`. The tests save a cyclic sample dataset, read it back, and leave the row in the metadata table for inspection.

## Streamlit UI

`app.py` is the interface:

- Sidebar tabs: **Data Generation** and **Talk to your data**
- Upload `.sql` / `.ddl` / `.txt`, optional prompt, temperature, and rows per table
- Generate with Gemini streaming + structured planning when Vertex AI is available, otherwise a local plan
- Per-table preview, textual feedback, CSV and ZIP download
- **Talk to your data**: pick a saved dataset, ask a natural-language question; Gemini proposes a SELECT, sqlglot rewrites it onto `dataset_<uuid>` with a row cap, and Postgres runs it with `search_path`, `transaction_read_only`, and `statement_timeout`

```
app.py                 # Streamlit UI
src/data_assistant/    # settings, schema, Gemini planning, querying,
                       # generation, validation, storage, feedback, export, health
```
