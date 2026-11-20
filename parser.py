import io
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import pdfplumber


@dataclass
class DocumentChunk:
    text: str
    source: str
    page_or_sheet: str
    chunk_type: str  # "text" or "table"
    chunk_id: int = 0


PARSER_VERSION = 2


@dataclass
class ParsedReport:
    filename: str
    chunks: List[DocumentChunk] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    extracted_metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    computed_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    currency_symbol: str = "$"
    unit_label: str = ""
    parser_version: int = PARSER_VERSION


def dataframe_to_markdown(df: pd.DataFrame, title: str = "") -> str:
    """Convert a pandas DataFrame into a clean Markdown table representation."""
    df_clean = df.fillna("").astype(str)
    # Remove completely empty rows and columns
    df_clean = df_clean.loc[~(df_clean == "").all(axis=1)]
    df_clean = df_clean.loc[:, ~(df_clean == "").all(axis=0)]
    if df_clean.empty:
        return ""

    headers = [str(col).strip().replace("\n", " ") for col in df_clean.columns]
    lines = []
    if title:
        lines.append(f"### {title}")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df_clean.iterrows():
        cells = [str(val).strip().replace("\n", " ") for val in row.values]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Split prose text into overlapping chunks while respecting paragraph boundaries."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                # Keep overlap from the end of previous chunk
                tail = current[-overlap:] if len(current) > overlap else current
                current = f"{tail}\n\n{para}".strip()
            else:
                # Hard-split very long paragraph
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end])
                    start += chunk_size - overlap
                current = ""

    if current:
        chunks.append(current)
    return chunks


def parse_pdf(file_bytes: bytes, filename: str, chunk_size: int = 900, overlap: int = 150) -> ParsedReport:
    """Parse PDF with table-aware extraction using pdfplumber."""
    report = ParsedReport(filename=filename)
    full_text_parts = []
    chunk_counter = 0

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            page_label = f"Page {page_idx}"
            page_text = page.extract_text() or ""

            if page_text.strip():
                full_text_parts.append(f"--- {page_label} ---\n{page_text}")
                for sub_text in chunk_text(page_text, chunk_size=chunk_size, overlap=overlap):
                    report.chunks.append(
                        DocumentChunk(
                            text=f"[Source: {filename} | {page_label}]\n{sub_text}",
                            source=filename,
                            page_or_sheet=page_label,
                            chunk_type="text",
                            chunk_id=chunk_counter,
                        )
                    )
                    chunk_counter += 1

            # Extract tables on this page
            extracted_tables = page.extract_tables() or []
            for t_idx, raw_table in enumerate(extracted_tables, start=1):
                if not raw_table or len(raw_table) < 2:
                    continue
                cleaned_rows = [
                    [cell.strip() if isinstance(cell, str) else "" for cell in row]
                    for row in raw_table
                    if any(cell is not None and str(cell).strip() != "" for cell in row)
                ]
                if len(cleaned_rows) < 2:
                    continue

                header = cleaned_rows[0]
                # Deduplicate or fill empty headers
                header = [h if h else f"Col_{i+1}" for i, h in enumerate(header)]
                df = pd.DataFrame(cleaned_rows[1:], columns=header)
                table_title = f"{page_label} - Table {t_idx}"
                md_table = dataframe_to_markdown(df, title=table_title)

                if md_table:
                    report.tables.append(
                        {
                            "title": table_title,
                            "page_or_sheet": page_label,
                            "dataframe": df,
                            "markdown": md_table,
                        }
                    )
                    full_text_parts.append(md_table)
                    report.chunks.append(
                        DocumentChunk(
                            text=f"[Source: {filename} | {table_title} (Structured Table)]\n{md_table}",
                            source=filename,
                            page_or_sheet=page_label,
                            chunk_type="table",
                            chunk_id=chunk_counter,
                        )
                    )
                    chunk_counter += 1

            # Fallback: extract borderless financial tables from page text when pdfplumber finds none
            if not extracted_tables and page_text.strip():
                text_table_df, text_table_title = _extract_borderless_table_from_text(page_text, page_label)
                if text_table_df is not None and not text_table_df.empty:
                    md_table = dataframe_to_markdown(text_table_df, title=text_table_title)
                    if md_table:
                        report.tables.append(
                            {
                                "title": text_table_title,
                                "page_or_sheet": page_label,
                                "dataframe": text_table_df,
                                "markdown": md_table,
                            }
                        )
                        full_text_parts.append(md_table)
                        report.chunks.append(
                            DocumentChunk(
                                text=f"[Source: {filename} | {text_table_title} (Structured Table)]\n{md_table}",
                                source=filename,
                                page_or_sheet=page_label,
                                chunk_type="table",
                                chunk_id=chunk_counter,
                            )
                        )
                        chunk_counter += 1

    report.raw_text = "\n\n".join(full_text_parts)
    report.currency_symbol, report.unit_label = detect_currency_and_unit(report.raw_text)
    report.extracted_metrics = extract_financial_metrics(report.raw_text)
    report.computed_ratios = compute_financial_ratios(report.extracted_metrics)
    return report


def parse_excel_or_csv(file_bytes: bytes, filename: str) -> ParsedReport:
    """Parse Excel (.xlsx/.xls) or CSV files into table chunks and extract financial metrics."""
    report = ParsedReport(filename=filename)
    full_text_parts = []
    chunk_counter = 0

    lower_name = filename.lower()
    if lower_name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        sheets = {"CSV_Data": df}
    else:
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)

    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        df_clean = df.dropna(how="all").dropna(axis=1, how="all")
        if df_clean.empty:
            continue

        md_table = dataframe_to_markdown(df_clean, title=f"Sheet: {sheet_name}")
        report.tables.append(
            {
                "title": f"Sheet: {sheet_name}",
                "page_or_sheet": sheet_name,
                "dataframe": df_clean,
                "markdown": md_table,
            }
        )
        full_text_parts.append(md_table)

        # Split large tables by groups of 35 rows so headers stay attached to each chunk
        rows_per_chunk = 35
        for start_idx in range(0, len(df_clean), rows_per_chunk):
            sub_df = df_clean.iloc[start_idx : start_idx + rows_per_chunk]
            sub_title = f"{sheet_name} (Rows {start_idx + 1}-{min(start_idx + rows_per_chunk, len(df_clean))})"
            sub_md = dataframe_to_markdown(sub_df, title=sub_title)
            report.chunks.append(
                DocumentChunk(
                    text=f"[Source: {filename} | {sub_title} (Structured Table)]\n{sub_md}",
                    source=filename,
                    page_or_sheet=sheet_name,
                    chunk_type="table",
                    chunk_id=chunk_counter,
                )
            )
            chunk_counter += 1

    report.raw_text = "\n\n".join(full_text_parts)
    report.currency_symbol, report.unit_label = detect_currency_and_unit(report.raw_text)
    report.extracted_metrics = extract_financial_metrics(report.raw_text, tables=report.tables)
    report.computed_ratios = compute_financial_ratios(report.extracted_metrics)
    return report


def _extract_borderless_table_from_text(page_text: str, page_label: str):
    """
    Detect and convert borderless financial statement tables (e.g., Balance Sheet,
    Profit or Loss, Cash Flows, Changes in Equity, Segment Reporting) into a DataFrame.
    """
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if len(lines) < 5:
        return None, ""

    # Identify statement title if present
    title = f"{page_label} - Financial Table"
    for ln in lines[:8]:
        if any(
            kw in ln.lower()
            for kw in [
                "statement of financial position",
                "statement of profit or loss",
                "statement of comprehensive income",
                "statement of changes in equity",
                "statement of cash flows",
                "segment analysis",
                "geographical segments",
                "property, plant and equipment",
                "transactions with related parties",
            ]
        ):
            title = f"{page_label} - {ln}"
            break

    num_token_re = re.compile(r"^(?:\(\d[\d,]*\.?\d*\)|\-?\d[\d,]*\.?\d*|\-)$")
    parsed_rows = []
    max_nums = 0

    for ln in lines:
        tokens = ln.split()
        if len(tokens) < 2:
            continue
        # Count trailing numeric tokens (or dashes representing zero)
        num_vals = []
        idx = len(tokens) - 1
        while idx >= 0 and num_token_re.match(tokens[idx]):
            num_vals.insert(0, tokens[idx])
            idx -= 1
        # Must have at least 1 numeric value with a comma/decimal and a text label
        if num_vals and any(any(c.isdigit() for c in v) and ("," in v or "." in v) for v in num_vals):
            label = " ".join(tokens[: idx + 1]).strip()
            # Ignore pure date/header lines or share count descriptions
            if not label:
                label = "Subtotal / Total"
            parsed_rows.append((label, num_vals))
            if len(num_vals) > max_nums:
                max_nums = len(num_vals)

    if len(parsed_rows) < 4 or max_nums == 0:
        return None, ""

    # Cap columns to at most 6 value columns
    max_cols = min(max_nums, 6)
    records = []
    for label, vals in parsed_rows:
        trimmed = vals[-max_cols:]
        padded = trimmed + [""] * (max_cols - len(trimmed))
        records.append([label] + padded)

    col_names = ["Line Item"] + [f"Period_{i+1}" for i in range(max_cols)]
    df = pd.DataFrame(records, columns=col_names)
    return df, title


def detect_currency_and_unit(text: str):
    """Detect report currency and unit scale (e.g. PKR in '000, $ in Millions)."""
    currency = "$"
    unit_label = ""

    if re.search(r"\b(?:rupees|pkr|rs\.)\b", text, flags=re.IGNORECASE):
        currency = "PKR "
    elif re.search(r"\b(?:eur|€)\b", text, flags=re.IGNORECASE):
        currency = "€"
    elif re.search(r"\b(?:gbp|£)\b", text, flags=re.IGNORECASE):
        currency = "£"

    if re.search(r"rupees\s+in\s+['’`]?000['’`]?", text, flags=re.IGNORECASE) or re.search(
        r"in\s+thousands", text, flags=re.IGNORECASE
    ):
        unit_label = "in '000"
    elif re.search(r"in\s+millions|\$\s*in\s*millions|pkr\s+million", text, flags=re.IGNORECASE):
        unit_label = "in Millions"

    return currency, unit_label


def parse_financial_file(file_bytes: bytes, filename: str) -> ParsedReport:
    """Dispatch parser based on file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return parse_pdf(file_bytes, filename)
    elif lower.endswith((".xlsx", ".xls", ".csv")):
        return parse_excel_or_csv(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file format for {filename}. Please upload PDF, XLSX, XLS, or CSV.")


def _parse_number(raw_val: str) -> Optional[float]:
    """Convert financial number strings like '$1,245.50', '(320.4)', '45.2%' into float."""
    if not raw_val:
        return None
    s = raw_val.strip()
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        val = float(s)
        return -val if is_negative else val
    except ValueError:
        return None


def extract_financial_metrics(text: str, tables: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Optional[float]]:
    """
    Extract core financial statement metrics from parsed text and structured tables.
    Supports both multi-period Excel/Markdown tables (picking the latest period column)
    and formal PDF financial statements (Income Statement, Balance Sheet, Cash Flows).
    """
    num_pat = r"([\(\$]?\-?\d[\d,]*\.?\d*\)?)"

    # Ordered patterns: specific formal statement line items first, generic fallbacks second
    patterns = {
        "Revenue": [
            rf"revenue\s+from\s+contracts\s+with\s+customers(?:\s*-\s*net)?[^\d\n\-\(]{{1,35}}{num_pat}",
            rf"(?:total\s+revenue|net\s+revenue|revenues|total\s+sales|net\s+sales)[^\d\n\-\(]{{1,40}}{num_pat}",
        ],
        "Cost of Goods Sold": [
            rf"(?:cost\s+of\s+goods\s+sold|cost\s+of\s+revenue|cost\s+of\s+sales|cogs)[^\d\n\-\(]{{1,40}}{num_pat}",
        ],
        "Gross Profit": [
            rf"gross\s+profit(?!\s+as\s+%|\s*[\-–]\s*pkr\s+million)[^\d\n\-\(]{{1,35}}{num_pat}",
            rf"(?:gross\s+profit|gross\s+margin\s+dollars)[^\d\n\-\(]{{1,40}}{num_pat}",
        ],
        "Operating Income": [
            rf"operating\s+(?:profit|income)(?!\s+as\s+%|\s*[\-–]\s*pkr\s+million|\s+before)[^\d\n\-\(]{{1,35}}{num_pat}",
            rf"(?:operating\s+income|operating\s+profit|ebit\b)[^\d\n\-\(]{{1,40}}{num_pat}",
        ],
        "Net Income": [
            rf"profit\s+after\s+taxation[^\d\n\-\(]{{1,35}}{num_pat}",
            rf"(?:net\s+income|net\s+earnings|net\s+profit(?:\s+after\s+tax)?)[^\d\n\-\(]{{1,40}}{num_pat}",
        ],
        "Total Assets": [
            rf"total\s+assets[^\d\n\-\(]{{1,40}}{num_pat}",
        ],
        "Current Assets": [
            rf"total\s+current\s+assets[^\d\n\-\(]{{1,40}}{num_pat}",
            rf"unclaimed\s+dividend\s+[\d,]+\s+[\d,]+\s+{num_pat}",
            rf"cash\s+and\s+bank\s+balances\s+[\d,]+\s+[\d,]+\s*\n\s*{num_pat}",
        ],
        "Non-Current Liabilities": [
            rf"total\s+non[\-\s]current\s+liabilities[^\d\n\-\(]{{1,40}}{num_pat}",
            rf"long[\-\s]term\s+employee\s+benefits\s+[\d,]+\s+[\d,]+\s*\n\s*{num_pat}",
        ],
        "Current Liabilities": [
            rf"total\s+current\s+liabilities[^\d\n\-\(]{{1,40}}{num_pat}",
            rf"unclaimed\s+dividend\s+[\d,]+\s+[\d,]+\s+[\d,]+\s+[\d,]+\s*\n\s*{num_pat}",
        ],
        "Total Liabilities": [
            rf"total\s+liabilities(?!\s+and\s+(?:shareholders|stockholders|equity))[^\d\n\-\(]{{1,40}}{num_pat}",
        ],
        "Shareholders Equity": [
            rf"(?:total\s+shareholders['’]?\s+equity|total\s+stockholders['’]?\s+equity|shareholders['’]?\s+equity)[^\d\n\-\(]{{1,40}}{num_pat}",
            rf"total\s+equity(?!\s+and\s+liabilities)[^\d\n\-\(]{{1,40}}{num_pat}",
            rf"accumulated\s+profits\s*-\s*revenue\s+reserve\s+[\d,]+\s+[\d,]+\s*\n\s*{num_pat}",
            rf"balance\s+as\s+at\s+june\s+30,\s+2026\s+\(un-audited\)(?:\s+[\d,]+){{4}}\s+{num_pat}",
        ],
        "Operating Cash Flow": [
            rf"net\s+cash\s+(?:\(used\s+in\)\s*/\s*generated\s+from|generated\s+from|from|used\s+in)\s+operating\s+activities[^\d\n\-\(]{{1,35}}{num_pat}",
            rf"(?:cash\s+(?:flows?\s+)?from\s+operating\s+activities|operating\s+cash\s+flow)[^\d\n\-\(]{{1,40}}{num_pat}",
        ],
        "Capital Expenditures": [
            rf"(?:purchase\s+of\s+property,\s+plant\s+and\s+equipment|capital\s+expenditures?(?:\s*\(capex\))?)[^\d\n\-\(]{{1,35}}{num_pat}",
        ],
        "Free Cash Flow": [
            rf"free\s+cash\s+flow[^\d\n\-\(]{{1,40}}{num_pat}",
        ],
    }

    extracted: Dict[str, Optional[float]] = {}

    # 1. If structured Excel/CSV tables are provided with multi-year columns (e.g. FY2025, FY2026),
    # prefer the latest period column for matching line items
    table_overrides: Dict[str, float] = {}
    if tables:
        label_map = {
            "Revenue": ["total revenue", "net revenue", "revenue from contracts with customers", "net sales"],
            "Cost of Goods Sold": ["cost of goods sold", "cost of sales", "cost of revenue"],
            "Gross Profit": ["gross profit"],
            "Operating Income": ["operating income", "operating profit"],
            "Net Income": ["net income", "profit after taxation", "net profit"],
            "Total Assets": ["total assets"],
            "Current Assets": ["total current assets"],
            "Total Liabilities": ["total liabilities"],
            "Current Liabilities": ["total current liabilities"],
            "Shareholders Equity": ["total shareholders equity", "total stockholders equity", "total equity"],
            "Operating Cash Flow": ["cash from operating activities", "net cash (used in) / generated from operating activities"],
            "Capital Expenditures": ["capital expenditures", "purchase of property, plant and equipment"],
            "Free Cash Flow": ["free cash flow"],
        }
        for tbl in tables:
            df = tbl.get("dataframe")
            if df is None or df.empty or len(df.columns) < 2:
                continue
            # Check if this is an Excel/CSV table with explicit year headers (e.g. FY2025, FY2026)
            year_cols = [c for c in df.columns[1:] if re.search(r"202\d", str(c)) and "change" not in str(c).lower()]
            if not year_cols:
                continue
            # Pick the highest year column (e.g. FY2026 over FY2025)
            target_col = sorted(year_cols, key=lambda c: re.findall(r"202\d", str(c))[-1])[-1]
            first_col = df.columns[0]
            for _, row in df.iterrows():
                item_label = str(row[first_col]).strip().lower()
                for metric_key, aliases in label_map.items():
                    if metric_key not in table_overrides and any(item_label == a or item_label.startswith(a + " ") for a in aliases):
                        parsed_v = _parse_number(str(row[target_col]))
                        if parsed_v is not None:
                            table_overrides[metric_key] = parsed_v

    for metric_name, regex_list in patterns.items():
        if metric_name in table_overrides:
            extracted[metric_name] = table_overrides[metric_name]
            continue
        val = None
        for pattern in regex_list:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                parsed = _parse_number(match.group(1))
                if parsed is not None:
                    val = parsed
                    break
        extracted[metric_name] = val

    # Derive Gross Profit if Revenue and COGS exist
    if extracted.get("Gross Profit") is None and extracted.get("Revenue") is not None and extracted.get("Cost of Goods Sold") is not None:
        extracted["Gross Profit"] = extracted["Revenue"] - abs(extracted["Cost of Goods Sold"])

    # Derive Total Liabilities if missing:
    # 1) Non-Current Liabilities + Current Liabilities
    if extracted.get("Total Liabilities") is None:
        ncl = extracted.get("Non-Current Liabilities")
        cl = extracted.get("Current Liabilities")
        if ncl is not None and cl is not None:
            extracted["Total Liabilities"] = ncl + cl
        elif extracted.get("Total Assets") is not None and extracted.get("Shareholders Equity") is not None:
            diff = extracted["Total Assets"] - extracted["Shareholders Equity"]
            if diff > 0:
                extracted["Total Liabilities"] = diff

    # Derive Free Cash Flow if missing: Operating Cash Flow - |Capital Expenditures|
    if extracted.get("Free Cash Flow") is None and extracted.get("Operating Cash Flow") is not None and extracted.get("Capital Expenditures") is not None:
        extracted["Free Cash Flow"] = extracted["Operating Cash Flow"] - abs(extracted["Capital Expenditures"])

    return extracted


def compute_financial_ratios(metrics: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Compute key financial ratios (%) and multiples from extracted metrics."""
    rev = metrics.get("Revenue")
    gp = metrics.get("Gross Profit")
    op_inc = metrics.get("Operating Income")
    net_inc = metrics.get("Net Income")
    assets = metrics.get("Total Assets")
    curr_assets = metrics.get("Current Assets")
    liab = metrics.get("Total Liabilities")
    curr_liab = metrics.get("Current Liabilities")
    equity = metrics.get("Shareholders Equity")

    ratios: Dict[str, Optional[float]] = {}

    ratios["Gross Margin (%)"] = round((gp / rev) * 100, 2) if (gp is not None and rev) else None
    ratios["Operating Margin (%)"] = round((op_inc / rev) * 100, 2) if (op_inc is not None and rev) else None
    ratios["Net Profit Margin (%)"] = round((net_inc / rev) * 100, 2) if (net_inc is not None and rev) else None
    ratios["Return on Equity - ROE (%)"] = round((net_inc / equity) * 100, 2) if (net_inc is not None and equity) else None
    ratios["Return on Assets - ROA (%)"] = round((net_inc / assets) * 100, 2) if (net_inc is not None and assets) else None
    ratios["Current Ratio (x)"] = round(curr_assets / curr_liab, 2) if (curr_assets is not None and curr_liab) else None
    ratios["Debt-to-Equity (x)"] = round(liab / equity, 2) if (liab is not None and equity) else None
    ratios["Debt-to-Assets (%)"] = round((liab / assets) * 100, 2) if (liab is not None and assets) else None

    return ratios
