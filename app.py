import importlib
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import parser as _parser_mod
import rag_engine as _rag_mod

importlib.reload(_parser_mod)
importlib.reload(_rag_mod)

from config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    TOP_K_RESULTS,
)
from parser import parse_financial_file, ParsedReport, PARSER_VERSION
from rag_engine import FinancialRAGEngine
from sample_generator import generate_sample_excel_bytes

st.set_page_config(
    page_title="FinSight AI — Financial Report RAG & Insight Bot",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading HuggingFace Embedding Model...")
def get_rag_engine(embedding_model: str, ollama_url: str, ollama_model: str) -> FinancialRAGEngine:
    return FinancialRAGEngine(
        embedding_model_name=embedding_model,
        ollama_base_url=ollama_url,
        ollama_model=ollama_model,
    )


def init_session_state():
    if "parsed_reports" not in st.session_state:
        st.session_state.parsed_reports = []
    if "indexed_filenames" not in st.session_state:
        st.session_state.indexed_filenames = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "executive_summary" not in st.session_state:
        st.session_state.executive_summary = ""


init_session_state()

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("📈 FinSight RAG Bot")
    st.caption("Local Open-Source Financial Report Q&A & Ratio Analysis")

    st.subheader("1. Upload Financial Report(s)")
    uploaded_files = st.file_uploader(
        "Upload PDF, Excel (.xlsx/.xls), or CSV",
        type=["pdf", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
    )

    if st.button("🚀 Load Sample FY2026 Report", use_container_width=True):
        sample_bytes = generate_sample_excel_bytes()
        sample_report = parse_financial_file(sample_bytes, "Apex_Cloud_FY2026_Financial_Report.xlsx")
        st.session_state.parsed_reports = [sample_report]
        st.session_state.indexed_filenames = [sample_report.filename]
        st.session_state.executive_summary = ""
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.subheader("2. Local LLM & Embeddings")
    ollama_url = st.text_input("Ollama Base URL", value=DEFAULT_OLLAMA_BASE_URL)
    embedding_model = st.selectbox(
        "HuggingFace Embedding Model",
        options=["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
        index=0,
    )

    engine = get_rag_engine(embedding_model, ollama_url, DEFAULT_OLLAMA_MODEL)
    ollama_online, available_models = engine.check_ollama_status()

    if ollama_online:
        st.success(f"🟢 Ollama Connected ({len(available_models)} model(s))")
        selected_model = st.selectbox(
            "Ollama Model",
            options=available_models if available_models else [DEFAULT_OLLAMA_MODEL],
        )
    else:
        st.warning("🟠 Ollama server not detected — Hybrid Retrieval & Rule-Based CFO Analytics active")
        selected_model = st.text_input("Ollama Model Name", value=DEFAULT_OLLAMA_MODEL)
        with st.expander("How to connect local Ollama"):
            st.code("ollama serve\nollama pull llama3.2", language="bash")

    engine.ollama_model = selected_model
    engine.ollama_base_url = ollama_url.rstrip("/")

    temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    top_k = st.slider("Top-K Retrieved Chunks", min_value=2, max_value=10, value=TOP_K_RESULTS)

# Process uploaded files if changed or if parser version was upgraded
if uploaded_files:
    current_names = sorted([f.name for f in uploaded_files])
    needs_reparse = (
        current_names != st.session_state.indexed_filenames
        or any(getattr(r, "parser_version", 0) != PARSER_VERSION for r in st.session_state.parsed_reports)
    )
    if needs_reparse:
        reports = []
        with st.spinner("Parsing tables & financial statements..."):
            for uf in uploaded_files:
                rep = parse_financial_file(uf.getvalue(), uf.name)
                reports.append(rep)
        st.session_state.parsed_reports = reports
        st.session_state.indexed_filenames = current_names
        st.session_state.executive_summary = ""

# Ensure reports are indexed in the vector store
if st.session_state.parsed_reports:
    total_chunks = engine.index_reports(st.session_state.parsed_reports)
else:
    total_chunks = 0

# --- Header ---
st.title("📊 Financial Report RAG & Insight Analysis Bot")
if not st.session_state.parsed_reports:
    st.info(
        "👋 **Get Started:** Upload a financial report (PDF 10-K/10-Q, Earnings Release, Excel, or CSV) in the sidebar, "
        "or click **'🚀 Load Sample FY2026 Report'** in the sidebar to explore an interactive example immediately."
    )
    st.stop()

active_files_badge = ", ".join([f"`{r.filename}`" for r in st.session_state.parsed_reports])
st.markdown(f"**Loaded Report(s):** {active_files_badge} &nbsp;|&nbsp; **Indexed Chunks & Tables:** `{total_chunks}`")

tab_dashboard, tab_chat, tab_inspector = st.tabs(
    [
        "📊 Executive Summary & Key Financial Ratios",
        "💬 Financial RAG Q&A Chat",
        "📑 Extracted Tables & Chunks Inspector",
    ]
)

# ==============================================================================
# TAB 1: EXECUTIVE SUMMARY & KEY FINANCIAL RATIOS DASHBOARD
# ==============================================================================
with tab_dashboard:
    primary_report: ParsedReport = st.session_state.parsed_reports[0]
    if len(st.session_state.parsed_reports) > 1:
        selected_rep_name = st.selectbox(
            "Select Report for Ratio Dashboard",
            options=[r.filename for r in st.session_state.parsed_reports],
        )
        primary_report = next(r for r in st.session_state.parsed_reports if r.filename == selected_rep_name)

    metrics = primary_report.extracted_metrics
    ratios = primary_report.computed_ratios
    curr_sym = getattr(primary_report, "currency_symbol", "$")
    unit_lbl = getattr(primary_report, "unit_label", "")
    unit_suffix = f" ({curr_sym.strip()} {unit_lbl})".strip() if unit_lbl else f" ({curr_sym.strip()})"

    st.subheader(f"1. Core Financial Statement Metrics {unit_suffix}")
    kpi_cols = st.columns(4)
    kpi_items = [
        ("Revenue", metrics.get("Revenue")),
        ("Gross Profit", metrics.get("Gross Profit")),
        ("Operating Income", metrics.get("Operating Income")),
        ("Net Income", metrics.get("Net Income")),
        ("Total Assets", metrics.get("Total Assets")),
        ("Total Liabilities", metrics.get("Total Liabilities")),
        ("Shareholders Equity", metrics.get("Shareholders Equity")),
        ("Operating Cash Flow", metrics.get("Operating Cash Flow")),
    ]
    for idx, (label, val) in enumerate(kpi_items):
        with kpi_cols[idx % 4]:
            formatted = f"{curr_sym}{val:,.2f}" if val is not None else "Not Detected"
            st.metric(label=label, value=formatted)

    st.divider()
    st.subheader("2. Automated Financial Ratios & Visual Breakdown")
    ratio_cols = st.columns(4)
    ratio_items = list(ratios.items())
    for idx, (r_label, r_val) in enumerate(ratio_items):
        with ratio_cols[idx % 4]:
            unit = "%" if "(%)" in r_label else "x"
            formatted_r = f"{r_val:,.2f}{unit}" if r_val is not None else "N/A"
            st.metric(label=r_label, value=formatted_r)

    # Interactive Charts if metrics exist
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        prof_data = {
            k: v
            for k, v in [
                ("Revenue", metrics.get("Revenue")),
                ("Gross Profit", metrics.get("Gross Profit")),
                ("Operating Income", metrics.get("Operating Income")),
                ("Net Income", metrics.get("Net Income")),
            ]
            if v is not None
        }
        if prof_data:
            df_prof = pd.DataFrame({"Metric": list(prof_data.keys()), "Amount": list(prof_data.values())})
            fig_prof = px.bar(
                df_prof,
                x="Metric",
                y="Amount",
                text_auto=".2s",
                title="Profitability Waterfall / Comparison",
                color="Metric",
            )
            fig_prof.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig_prof, use_container_width=True)

    with chart_col2:
        margin_data = {
            k: v
            for k, v in [
                ("Gross Margin (%)", ratios.get("Gross Margin (%)")),
                ("Operating Margin (%)", ratios.get("Operating Margin (%)")),
                ("Net Profit Margin (%)", ratios.get("Net Profit Margin (%)")),
                ("ROE (%)", ratios.get("Return on Equity - ROE (%)")),
                ("ROA (%)", ratios.get("Return on Assets - ROA (%)")),
            ]
            if v is not None
        }
        if margin_data:
            df_margin = pd.DataFrame({"Ratio": list(margin_data.keys()), "Percentage (%)": list(margin_data.values())})
            fig_margin = px.bar(
                df_margin,
                x="Percentage (%)",
                y="Ratio",
                orientation="h",
                text_auto=".1f",
                title="Profitability & Return Ratios (%)",
                color="Ratio",
            )
            fig_margin.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig_margin, use_container_width=True)

    st.divider()
    st.subheader("3. AI Executive Summary & Deep-Dive Insight Analysis")
    gen_col, _ = st.columns([1, 3])
    with gen_col:
        regenerate = st.button("✨ Generate / Refresh Executive Analysis", type="primary", use_container_width=True)

    if regenerate or not st.session_state.executive_summary:
        with st.spinner("Analyzing financial statements and generating executive insights..."):
            st.session_state.executive_summary = engine.generate_executive_insights(
                st.session_state.parsed_reports, temperature=temperature
            )

    st.markdown(st.session_state.executive_summary)


# ==============================================================================
# TAB 2: FINANCIAL RAG Q&A CHAT
# ==============================================================================
with tab_chat:
    st.subheader("Ask Questions & Get Grounded Financial Insights")

    st.caption("Suggested Analytical Questions:")
    q_cols = st.columns(4)
    suggested_questions = [
        "What drove revenue growth and margin expansion?",
        "Analyze the company's liquidity, cash position, and current ratio.",
        "How much did R&D and Sales & Marketing expenses grow YoY?",
        "What are the primary risk factors and future guidance?",
    ]
    triggered_question = None
    for idx, sq in enumerate(suggested_questions):
        with q_cols[idx]:
            if st.button(sq, use_container_width=True):
                triggered_question = sq

    # Display existing chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📚 View {len(msg['citations'])} Grounded Source Citations & Tables"):
                    for c in msg["citations"]:
                        st.markdown(
                            f"**[{c['rank']}] `{c['source']}` — {c['location']}** "
                            f"*(Type: `{c['type']}` | Hybrid Score: `{c['score']}`)*"
                        )
                        st.code(c["snippet"], language="markdown")

    user_input = st.chat_input("Ask about revenue drivers, margins, balance sheet items, cash flow, or risks...")
    active_query = triggered_question or user_input

    if active_query:
        st.session_state.chat_history.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant financial tables & synthesizing insight..."):
                result = engine.answer_question(
                    question=active_query,
                    reports=st.session_state.parsed_reports,
                    chat_history=st.session_state.chat_history[:-1],
                    top_k=top_k,
                    temperature=temperature,
                )
            st.markdown(result["answer"])
            if result["citations"]:
                with st.expander(f"📚 View {len(result['citations'])} Grounded Source Citations & Tables"):
                    for c in result["citations"]:
                        st.markdown(
                            f"**[{c['rank']}] `{c['source']}` — {c['location']}** "
                            f"*(Type: `{c['type']}` | Hybrid Score: `{c['score']}`)*"
                        )
                        st.code(c["snippet"], language="markdown")

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "citations": result["citations"],
            }
        )
        if triggered_question:
            st.rerun()


# ==============================================================================
# TAB 3: PARSED TABLES & CHUNKS INSPECTOR
# ==============================================================================
with tab_inspector:
    st.subheader("Table-Aware Extraction Inspector")
    for rep in st.session_state.parsed_reports:
        st.markdown(f"### 📄 `{rep.filename}` — {len(rep.tables)} Table(s), {len(rep.chunks)} Indexed Chunk(s)")
        if rep.tables:
            for t in rep.tables:
                with st.expander(f"📊 {t['title']} ({t['page_or_sheet']})", expanded=False):
                    st.dataframe(t["dataframe"], use_container_width=True)
        with st.expander("🔍 View Raw Indexed Text & Table Chunks"):
            for ch in rep.chunks:
                st.markdown(f"**Chunk #{ch.chunk_id}** (`{ch.page_or_sheet}` | `{ch.chunk_type}`)")
                st.code(ch.text, language="markdown")
