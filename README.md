# 📈 FinSight AI — Financial Report RAG & Insight Analysis Bot

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B.svg)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/LLM-Local%20Ollama-black.svg)](https://ollama.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS%20%2B%20HuggingFace-00ADD8.svg)](https://github.com/facebookresearch/faiss)

A local, privacy-first **Financial Report RAG (Retrieval-Augmented Generation) & Automated Ratio Analysis Bot** built with **Python**, **Streamlit**, **HuggingFace Sentence-Transformers + FAISS**, **pdfplumber / pandas**, and **Ollama**.

FinSight AI ingests corporate financial reports—including **PDF interim/annual reports (10-K, 10-Q, half-year reviews)**, **Excel (`.xlsx`/`.xls`)**, and **CSV** statements—extracts both **bordered and borderless accounting tables**, detects the reporting currency and scale (e.g., `PKR in '000`, `$ in Millions`), computes core financial ratios automatically, and lets you chat with your financial statements using grounded citations.

---

## ✨ Key Features

### 1. Table-Aware Financial Document Parser (`parser.py`)
- **Ruled & Borderless Accounting Table Extraction:** Combines `pdfplumber` table detection with an intelligent borderless financial statement parser (`_extract_borderless_table_from_text`) that reconstructs multi-column statements (*Statement of Financial Position / Balance Sheet*, *Statement of Profit or Loss*, *Statement of Changes in Equity*, *Statement of Cash Flows*, and *Segment Notes*) into structured DataFrames and Markdown tables.
- **Automatic Currency & Unit Scale Detection:** Automatically detects currency (`PKR`, `$`, `€`, `£`) and unit scale (`in '000`, `in Millions`) and prioritizes formal financial statements over executive summary snapshots so all metrics share a consistent scale.
- **Multi-Sheet Excel & CSV Ingestion:** Preserves column headers across table chunks and automatically selects the latest fiscal period column in multi-year spreadsheets.

### 2. Automated Executive Dashboard & Financial Ratios (`app.py`, `parser.py`)
- **Core Statement KPIs Extracted Automatically:**
  - **Income Statement:** Revenue, Cost of Goods Sold (COGS), Gross Profit, Operating Income (EBIT), Net Income (Profit After Taxation)
  - **Balance Sheet:** Total Assets, Current Assets, Total Liabilities, Current Liabilities, Non-Current Liabilities, Shareholders' Equity
  - **Cash Flow Statement:** Operating Cash Flow, Capital Expenditures (CapEx), and auto-derived Free Cash Flow (FCF)
- **8 Key Financial Ratios Computed Automatically:**
  - **Profitability:** Gross Margin (`%`), Operating Margin (`%`), Net Profit Margin (`%`)
  - **Returns:** Return on Equity — ROE (`%`), Return on Assets — ROA (`%`)
  - **Liquidity & Leverage:** Current Ratio (`x`), Debt-to-Equity (`x`), Debt-to-Assets (`%`)
- **Interactive Visualizations:** Renders Plotly profitability comparison bars and margin/return ratio breakdowns.

### 3. Hybrid Financial RAG Engine (`rag_engine.py`)
- **70% Dense Semantic + 30% Lexical & Table-Boosted Retrieval:** Embeds document chunks and Markdown tables with HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`) indexed in **FAISS**, combined with keyword overlap and structured table boosting so line-item queries retrieve exact statement rows.
- **Local Ollama LLM Integration + Deterministic CFO Fallback:** Connects to your local Ollama server (`http://localhost:11434`, default model `llama3.2:1b` / `llama3.2`) for grounded financial analysis with `[Page / Sheet]` citations, and seamlessly falls back to a rule-based CFO analytical engine if Ollama is offline.

### 4. Built-In 1-Click Sample Financial Report (`sample_generator.py`)
- Includes a ready-to-use multi-sheet financial report (**Apex Cloud Technologies Inc. — FY2026 Financial Report**) with Income Statement, Balance Sheet, Cash Flow Statement, and MD&A Commentary so you can test the app with a single click.

---

## 🗂️ Project Structure

```text
financial-rag-bot/
├── app.py                 # Streamlit UI (Dashboard, RAG Q&A Chat, Table & Chunk Inspector)
├── parser.py              # PDF (bordered & borderless tables), Excel & CSV parser + metric/ratio extractor
├── rag_engine.py          # Hybrid FAISS + lexical RAG engine and Ollama LLM client
├── sample_generator.py    # 1-click sample multi-sheet Excel financial report generator
├── config.py              # Default model, URL, and chunking configuration
├── requirements.txt       # Python dependencies
├── .streamlit/config.toml # Streamlit server settings
└── README.md              # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository & Create a Virtual Environment

```powershell
git clone https://github.com/<your-username>/financial-rag-bot.git
cd financial-rag-bot

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Start Local Ollama (Recommended)

Install [Ollama](https://ollama.com/) and pull a lightweight local model such as `llama3.2:1b` or `llama3.2`:

```powershell
ollama serve
ollama pull llama3.2:1b
```

*(Note: If Ollama is not running, FinSight AI automatically switches to its built-in Hybrid Retrieval & Rule-Based CFO Analytics mode so all dashboards, ratios, and table lookups still work offline.)*

### 3. Run the Application

```powershell
streamlit run app.py
```

Open **`http://localhost:8501`** in your browser:
1. Upload any **PDF**, **Excel (`.xlsx`/`.xls`)**, or **CSV** financial statement in the left sidebar—or click **"🚀 Load Sample FY2026 Report"**.
2. Explore **Tab 1 (`📊 Executive Summary & Key Financial Ratios`)** for automated KPI cards, financial ratios, Plotly charts, and the AI Executive Summary.
3. Switch to **Tab 2 (`💬 Financial RAG Q&A Chat`)** to ask custom questions with expandable source citations and table snippets.
4. Use **Tab 3 (`📑 Extracted Tables & Chunks Inspector`)** to inspect every structured DataFrame and indexed text/table chunk extracted from your report.

---

## ⚙️ Configuration (`config.py` / Environment Variables)

You can customize defaults via environment variables or directly in the Streamlit sidebar:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server endpoint |
| `OLLAMA_MODEL` | `llama3.2:1b` | Default Ollama model tag |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace SentenceTransformer model (`all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`) |
| `CHUNK_SIZE` | `900` | Character length per text chunk |
| `CHUNK_OVERLAP` | `150` | Character overlap between adjacent chunks |
| `TOP_K_RESULTS` | `5` | Default number of hybrid-retrieved chunks passed to the LLM |

---

## 📄 License

This project is open-source and available under the MIT License.
