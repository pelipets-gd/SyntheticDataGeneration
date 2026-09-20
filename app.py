"""Data Assistant Streamlit UI — synthetic data generation and NL queries."""

from __future__ import annotations

from typing import Any

import streamlit as st

from data_assistant.exporting import dataset_zip, table_csv
from data_assistant.feedback import FeedbackError, apply_feedback_operations
from data_assistant.gemini_planning import GeminiPlanningService
from data_assistant.generation import GenerationError, SyntheticDataGenerator
from data_assistant.health import check_postgres_health, default_postgres_connect
from data_assistant.planning_defaults import default_generation_plan
from data_assistant.schema import DDLParseError, parse_ddl
from data_assistant.settings import load_settings
from data_assistant.storage import DatasetRepository, StorageError, default_connect

st.set_page_config(page_title="Data Assistant", layout="wide", initial_sidebar_state="expanded")

_ALLOWED_UPLOADS = (".sql", ".txt", ".ddl")


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "page": "Data Generation",
        "schema": None,
        "data": None,
        "ddl": "",
        "dataset_id": None,
        "dataset_name": "Generated dataset",
        "preview_table": None,
        "status_log": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _repository():
    settings = st.session_state.settings
    store = DatasetRepository(connect=default_connect(settings.database_url))
    try:
        store.initialize()
    except StorageError:
        return None
    return store


def _sidebar() -> str:
    st.sidebar.title("Data Assistant")
    return st.sidebar.radio(
        "Navigation",
        ("Data Generation", "Talk to your data"),
        label_visibility="collapsed",
        index=0 if st.session_state.page == "Data Generation" else 1,
    )


def _generation_page() -> None:
    st.markdown("### Prompt")
    prompt = st.text_input(
        "Prompt",
        placeholder="Enter your prompt here...",
        label_visibility="collapsed",
    )

    uploaded = st.file_uploader(
        "Upload DDL Schema",
        type=["sql", "txt", "ddl"],
        help="Supported formats: SQL, TXT, DDL",
    )
    st.caption("Supported formats: SQL, JSON")
    if uploaded is not None:
        st.session_state.ddl = uploaded.getvalue().decode("utf-8", errors="replace")
        st.caption(f"Loaded `{uploaded.name}`")

    with st.expander("Advanced Parameters", expanded=True):
        left, right = st.columns(2)
        with left:
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        with right:
            row_count = st.number_input("Rows per table", min_value=1, max_value=1000, value=25, step=1)
        max_tokens = st.number_input("Max Tokens", min_value=16, max_value=8192, value=100, step=1)
        dataset_name = st.text_input("Dataset name", value=st.session_state.dataset_name)

    if st.button("Generate", type="primary"):
        _run_generation(prompt, temperature, int(row_count), dataset_name, int(max_tokens))

    if st.session_state.status_log:
        st.caption(st.session_state.status_log)

    if st.session_state.data and st.session_state.schema is not None:
        _preview_and_feedback()


def _run_generation(
    prompt: str, temperature: float, row_count: int, dataset_name: str, max_tokens: int
) -> None:
    ddl = st.session_state.ddl
    if not ddl.strip():
        st.error("Upload a DDL schema file (.sql, .txt, or .ddl) before generating.")
        return
    try:
        schema = parse_ddl(ddl)
    except DDLParseError as error:
        st.error(str(error))
        return

    settings = st.session_state.settings
    planner = GeminiPlanningService(settings)
    status_bits: list[str] = []
    status = st.empty()
    try:
        for chunk in planner.stream_explanation(
            schema,
            prompt or "Generate realistic relational sample data.",
            temperature=temperature,
        ):
            status_bits.append(chunk)
            status.write("".join(status_bits))
    except Exception:  # noqa: BLE001 — local generation still proceeds
        status.info("Gemini streaming is unavailable; generating with the local planner.")

    try:
        plan = planner.request_plan(schema, temperature=temperature)
    except Exception:  # noqa: BLE001
        plan = default_generation_plan(schema, row_count)
        status.warning("Gemini structured planning failed; used the local generation plan.")

    try:
        data = SyntheticDataGenerator(max_row_count=1000).generate(
            schema, plan, row_count=row_count, seed=7
        )
    except GenerationError as error:
        st.error(str(error))
        return

    st.session_state.schema = schema
    st.session_state.data = data
    st.session_state.dataset_name = dataset_name
    st.session_state.preview_table = schema.tables[0].name
    st.session_state.status_log = "".join(status_bits) or f"Generated {row_count} rows per table."
    _persist(dataset_name, ddl, prompt, schema, data)
    _ = max_tokens


def _persist(name: str, ddl: str, prompt: str, schema, data) -> None:
    store = _repository()
    if store is None:
        st.warning("PostgreSQL is unavailable; the dataset is kept in this session only.")
        st.session_state.dataset_id = None
        return
    try:
        summary = store.save_dataset(
            name=name, ddl=ddl, schema=schema, data=data, instructions=prompt or None
        )
    except Exception as error:  # noqa: BLE001
        st.warning(f"Could not store the dataset: {error}")
        st.session_state.dataset_id = None
        return
    st.session_state.dataset_id = summary.dataset_id
    st.success(f"Stored as `{summary.schema_name}`.")


def _preview_and_feedback() -> None:
    schema = st.session_state.schema
    data = st.session_state.data
    names = [table.name for table in schema.tables]
    header, selector = st.columns([3, 1])
    with header:
        st.markdown("### Data Preview")
    with selector:
        table_name = st.selectbox(
            "Table",
            names,
            index=names.index(st.session_state.preview_table)
            if st.session_state.preview_table in names
            else 0,
            label_visibility="collapsed",
        )
        st.session_state.preview_table = table_name

    st.dataframe(data[table_name], use_container_width=True, hide_index=True)

    csv_col, zip_col = st.columns(2)
    with csv_col:
        st.download_button(
            "Download CSV",
            data=table_csv(schema, table_name, data[table_name]),
            file_name=f"{table_name}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with zip_col:
        st.download_button(
            "Download ZIP",
            data=dataset_zip(schema, data),
            file_name=f"{st.session_state.dataset_name.replace(' ', '_')}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    feedback_col, submit_col = st.columns([5, 1])
    with feedback_col:
        feedback = st.text_input(
            "Feedback",
            placeholder="Enter quick edit instructions...",
            label_visibility="collapsed",
            key="feedback_text",
        )
    with submit_col:
        submitted = st.button("Submit", type="primary", use_container_width=True)

    if submitted:
        _apply_feedback(table_name, feedback)


def _apply_feedback(table_name: str, feedback: str) -> None:
    if not feedback.strip():
        st.error("Enter feedback before submitting.")
        return
    schema = st.session_state.schema
    settings = st.session_state.settings
    planner = GeminiPlanningService(settings)
    try:
        operations = planner.request_feedback_operations(schema, table_name, feedback)
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not interpret feedback: {error}")
        return
    if not operations:
        st.info("No safe edits were produced for that instruction.")
        return
    try:
        updated = apply_feedback_operations(schema, st.session_state.data, operations, seed=11)
    except FeedbackError as error:
        st.error(str(error))
        return
    st.session_state.data = updated
    store = _repository()
    if store is not None and st.session_state.dataset_id is not None:
        try:
            store.update_table(st.session_state.dataset_id, table_name, updated[table_name])
        except StorageError as error:
            st.warning(f"Preview updated, but PostgreSQL was not: {error}")
    st.success("Applied feedback and re-validated the dataset.")
    st.rerun()


def _talk_page() -> None:
    st.markdown("### Talk to your data")
    store = _repository()
    if store is None:
        st.warning("PostgreSQL is unavailable, so stored datasets cannot be queried.")
        return
    try:
        datasets = store.list_datasets()
    except StorageError as error:
        st.error(str(error))
        return
    if not datasets:
        st.write("No saved datasets yet. Generate data in the Data Generation tab.")
        return

    labels = {
        f"{item.name} ({item.schema_name})": item.dataset_id for item in datasets
    }
    selected = st.selectbox("Dataset", list(labels))
    question = st.text_area(
        "Question",
        placeholder="Ask a question about the selected dataset...",
        height=100,
    )
    asked = st.button("Ask", type="primary", use_container_width=True)
    if not asked:
        return
    if not question.strip():
        st.warning("Enter a question.")
        return
    record = store.load_dataset(labels[selected])
    try:
        plan = GeminiPlanningService(st.session_state.settings).request_query(
            record.schema, question
        )
        page = store.execute_select(
            record.dataset_id, plan.sql, explanation=plan.explanation
        )
    except (ValueError, StorageError) as error:
        st.error(str(error))
        return
    st.caption(page.explanation)
    st.code(page.sql, language="sql")
    st.dataframe(list(page.rows), use_container_width=True, hide_index=True)


def main() -> None:
    _init_state()
    try:
        st.session_state.settings = load_settings()
    except ValueError as error:
        st.error(str(error))
        st.stop()

    health = check_postgres_health(
        connect=default_postgres_connect(st.session_state.settings.database_url)
    )
    if not health.ok:
        st.sidebar.warning(health.detail)

    page = _sidebar()
    st.session_state.page = page
    if page == "Data Generation":
        _generation_page()
    else:
        _talk_page()


if __name__ == "__main__":
    main()
else:
    main()
