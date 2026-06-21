"""
app.py
-------
Streamlit GUI for the Automated Literature Review & Gap Analysis Assistant.

Run with:
    streamlit run app.py

Layout:
    Sidebar  -- API key entry, instructions, processing trigger
    Tab 1    -- Upload & Process (PDF upload, pipeline progress)
    Tab 2    -- Comparative Table (the required 5/6-column table, exportable)
    Tab 3    -- Gap Analysis (theme clusters, contradictions, confidence flags)
    Tab 4    -- Ask Your Papers (RAG chat over the corpus)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.gemini_client import GeminiClient, GeminiAPIError
from src.pipeline import run_pipeline
from src.rag_chat import answer_question
from src.report_export import export_csv_bytes, export_docx_bytes

st.set_page_config(page_title=config.APP_TITLE, page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "saved_upload_paths" not in st.session_state:
    st.session_state.saved_upload_paths = []
if "saved_upload_names" not in st.session_state:
    st.session_state.saved_upload_names = []


# ---------------------------------------------------------------------------
# Sidebar -- configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title(config.APP_TITLE)
    st.caption("NLP / GenAi course project")

    st.markdown("### Gemini API Key")
    api_key_input = st.text_input(
        "Enter your Gemini API key",
        type="password",
        help="Get a free key at https://aistudio.google.com/apikey. "
        "It is only stored in this browser session, never written to disk.",
    )
    st.caption("Free tier is sufficient for this app; large batches may take a few minutes due to rate limits.")

    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "1. Upload 5-20 research paper PDFs\n"
        "2. Click **Process Papers**\n"
        "3. Review the auto-generated comparative table\n"
        "4. Check the Gap Analysis tab for cross-paper themes\n"
        "5. Export to CSV / Word for your report\n"
        "6. Ask follow-up questions in **Ask Your Papers**"
    )

    st.divider()
   


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_upload, tab_table, tab_gaps, tab_chat = st.tabs(
    ["1. Upload & Process", "2. Comparative Table", "3. Gap Analysis", "4. Ask Your Papers"]
)

# ---------------------------------------------------------------------------
# TAB 1 -- Upload & Process
# ---------------------------------------------------------------------------
with tab_upload:
    st.header("Upload your papers")
    st.write(
        "Upload the PDFs you want included in your literature review. "
        "For a course Comparative Analysis Table you typically need 15-20 studies, "
        "but you can process them in smaller batches if you prefer."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Only text-based PDFs are supported (scanned/image-only PDFs without OCR will fail).",
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        process_clicked = st.button("Process Papers", type="primary", use_container_width=True)
    with col_b:
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) ready to process.")

    if process_clicked:
        if not api_key_input:
            st.error("Please enter your Gemini API key in the sidebar first.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF.")
        elif len(uploaded_files) < 2:
            st.warning(
                "Gap analysis works best with 2+ papers (it looks for patterns ACROSS papers). "
                "Proceeding with a single paper will skip cross-paper gap analysis."
            )

    if process_clicked and api_key_input and uploaded_files:
        saved_paths = []
        saved_names = []
        for uf in uploaded_files:
            dest = config.UPLOAD_DIR / uf.name
            dest.write_bytes(uf.getvalue())
            saved_paths.append(str(dest))
            saved_names.append(uf.name)
        st.session_state.saved_upload_paths = saved_paths
        st.session_state.saved_upload_names = saved_names

        try:
            client = GeminiClient(api_key=api_key_input)
        except GeminiAPIError as exc:
            st.error(f"Could not initialize Gemini client: {exc}")
            st.stop()

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _progress_callback(stage, detail, fraction):
            progress_bar.progress(min(max(fraction, 0.0), 1.0))
            status_text.info(f"**{stage}** — {detail}")

        with st.spinner("Running pipeline... this can take 1-3 minutes for larger batches."):
            try:
                result = run_pipeline(
                    client,
                    saved_paths,
                    original_names=saved_names,
                    progress_callback=_progress_callback,
                )
                st.session_state.pipeline_result = result
                st.session_state.chat_history = []
            except GeminiAPIError as exc:
                st.error(f"Gemini API error: {exc}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error while processing papers: {exc}")
                st.stop()

        status_text.success("Processing complete! See the other tabs for results.")

    result = st.session_state.pipeline_result
    if result:
        st.divider()
        st.subheader("Processing summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Papers processed", len(result.papers))
        c2.metric(
            "Successful extractions",
            sum(1 for f in result.extracted_fields if not f.extraction_error),
        )
        c3.metric(
            "Gap clusters found",
            len(result.gap_result.gap_clusters) if result.gap_result and not result.gap_result.error else 0,
        )

        if result.extraction_errors:
            with st.expander(f"{len(result.extraction_errors)} PDF(s) failed to extract text", expanded=False):
                for name, msg in result.extraction_errors:
                    st.write(f"- **{name}**: {msg}")

        if result.warnings:
            for w in result.warnings:
                st.warning(w)


# ---------------------------------------------------------------------------
# TAB 2 -- Comparative Table
# ---------------------------------------------------------------------------
with tab_table:
    st.header("Comparative Analysis Table")
    result = st.session_state.pipeline_result

    if not result or not result.table_rows:
        st.info("Process some papers in Tab 1 first to see the comparative table here.")
    else:
        st.write(
            "This table mirrors the required column schema: **Reference, Title, "
            "Proposed Solution, Limitations/Gaps, Our Contribution**. The "
            "**Confidence** column flags rows the LLM was less certain about -- "
            "review any row below 60 before using it in your report."
        )

        df = pd.DataFrame(result.table_rows)

        def _highlight_low_confidence(row):
            if row.get("Confidence", 100) < 60:
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df.style.apply(_highlight_low_confidence, axis=1),
            use_container_width=True,
            height=min(70 + 40 * len(df), 600),
        )

        low_conf_count = sum(1 for r in result.table_rows if r.get("Confidence", 100) < 60)
        if low_conf_count:
            st.warning(
                f"{low_conf_count} row(s) have confidence below 60 -- these are highlighted "
                "above. Please double-check them against the original PDF before submission."
            )

        st.divider()
        st.subheader("Export for your report")
        col1, col2 = st.columns(2)

        with col1:
            csv_bytes = export_csv_bytes(result.table_rows)
            st.download_button(
                "Download as CSV",
                data=csv_bytes,
                file_name="comparative_analysis_table.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            try:
                docx_bytes = export_docx_bytes(
                    result.table_rows,
                    title="Comparative Analysis Table",
                    subtitle="Generated by the Automated Literature Review & Gap Analysis Assistant",
                )
                st.download_button(
                    "Download as Word (.docx)",
                    data=docx_bytes,
                    file_name="comparative_analysis_table.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            except RuntimeError as exc:
                st.caption(f"Word export unavailable ({exc}). Use CSV export instead.")


# ---------------------------------------------------------------------------
# TAB 3 -- Gap Analysis
# ---------------------------------------------------------------------------
with tab_gaps:
    st.header("Cross-Paper Gap Analysis")
    result = st.session_state.pipeline_result

    if not result or not result.gap_result:
        st.info("Process some papers in Tab 1 first to see gap analysis here.")
    elif result.gap_result.error:
        st.warning(f"Gap analysis could not be completed: {result.gap_result.error}")
    else:
        gap_result = result.gap_result

        st.subheader(f"Thematic gap clusters ({len(gap_result.gap_clusters)})")
        st.caption(
            "Each cluster groups a recurring limitation across multiple papers -- "
            "these are strong candidates for your report's 'Contribution and Motivation' section."
        )
        if not gap_result.gap_clusters:
            st.write("No recurring gap themes were identified.")
        for cluster in gap_result.gap_clusters:
            with st.expander(f"📌 {cluster.theme}  ({len(cluster.papers)} paper(s))"):
                st.write(cluster.description)
                papers_line = ", ".join(cluster.papers) if cluster.papers else "—"
                st.markdown(f"**Papers exhibiting this gap:** {papers_line}")
                st.markdown(f"**Suggested direction:** {cluster.suggested_contribution}")

        st.divider()
        st.subheader(f"Cross-paper contradictions ({len(gap_result.contradictions)})")
        st.caption("Papers that appear to disagree on the same sub-problem -- useful for your Discussion section.")
        if not gap_result.contradictions:
            st.write("No clear contradictions were detected among the processed papers.")
        for c in gap_result.contradictions:
            st.markdown(f"**{c.paper_a}** vs **{c.paper_b}**")
            st.write(c.issue)
            st.markdown("---")


# ---------------------------------------------------------------------------
# TAB 4 -- Ask Your Papers (RAG chat)
# ---------------------------------------------------------------------------
with tab_chat:
    st.header("Ask Your Papers")
    result = st.session_state.pipeline_result

    if not result or not result.vector_store:
        st.info("Process some papers in Tab 1 first to enable Q&A over your corpus.")
    else:
        st.caption(
            "Ask questions across all uploaded papers. Answers are grounded only in "
            "the retrieved excerpts and cite the source file name."
        )

        paper_names = sorted({p.file_name for p in result.papers})
        filter_choice = st.selectbox(
            "Restrict search to a specific paper (optional)",
            options=["All papers"] + paper_names,
        )
        paper_filter = None if filter_choice == "All papers" else filter_choice

        for turn in st.session_state.chat_history:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])
                if turn.get("sources"):
                    with st.expander("Sources"):
                        for s in turn["sources"]:
                            st.markdown(f"**{s['file_name']}**: {s['snippet']}...")

        question = st.chat_input("Ask a question about your uploaded papers...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            try:
                client = GeminiClient(api_key=api_key_input)
                with st.spinner("Searching your papers..."):
                    rag_result = answer_question(
                        client,
                        result.vector_store,
                        question,
                        paper_filter=paper_filter,
                    )
                if rag_result.error:
                    answer_text = f"Sorry, I ran into an error: {rag_result.error}"
                    sources = []
                else:
                    answer_text = rag_result.answer
                    sources = rag_result.sources
            except GeminiAPIError as exc:
                answer_text = f"Sorry, I ran into an error: {exc}"
                sources = []

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer_text, "sources": sources}
            )
            with st.chat_message("assistant"):
                st.write(answer_text)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"**{s['file_name']}**: {s['snippet']}...")
