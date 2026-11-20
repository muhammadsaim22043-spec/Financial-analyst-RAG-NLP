import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import requests

from config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    TOP_K_RESULTS,
)
from parser import DocumentChunk, ParsedReport

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class FinancialRAGEngine:
    """
    Hybrid Financial RAG Engine powered by HuggingFace SentenceTransformers + FAISS
    and local Ollama LLM inference.
    """

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ):
        self.embedding_model_name = embedding_model_name
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model

        self.embedder = None
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def _ensure_embedder(self):
        if self.embedder is None and SentenceTransformer is not None:
            self.embedder = SentenceTransformer(self.embedding_model_name)

    def check_ollama_status(self) -> Tuple[bool, List[str]]:
        """Check if local Ollama server is running and return available models."""
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                return True, models
        except Exception:
            pass
        return False, []

    def index_reports(self, reports: List[ParsedReport]) -> int:
        """Embed and index all chunks from one or more parsed financial reports."""
        self.chunks = []
        for rep in reports:
            self.chunks.extend(rep.chunks)

        if not self.chunks:
            self.index = None
            self.embeddings = None
            return 0

        self._ensure_embedder()
        texts = [c.text for c in self.chunks]

        if self.embedder is not None:
            vecs = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            vecs = vecs.astype("float32")
            # Normalize for cosine similarity
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
            self.embeddings = vecs

            if faiss is not None:
                dim = vecs.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(vecs)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Tuple[DocumentChunk, float]]:
        """
        Hybrid retrieval combining dense semantic similarity (HuggingFace + FAISS)
        with lexical keyword & table boosting for financial terms.
        """
        if not self.chunks:
            return []

        query_terms = [
            t.lower()
            for t in re.findall(r"[a-zA-Z0-9_]+", query)
            if len(t) > 2
        ]

        # Compute lexical overlap score
        lexical_scores = np.zeros(len(self.chunks), dtype="float32")
        for idx, ch in enumerate(self.chunks):
            lower_text = ch.text.lower()
            matches = sum(1 for term in query_terms if term in lower_text)
            if query_terms:
                lexical_scores[idx] = matches / len(query_terms)
            # Slightly boost structured tables for numerical / ratio queries
            if ch.chunk_type == "table":
                lexical_scores[idx] += 0.12

        # Compute dense vector similarity score
        dense_scores = np.zeros(len(self.chunks), dtype="float32")
        if self.embedder is not None and self.embeddings is not None:
            q_vec = self.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
            q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
            q_norm[q_norm == 0] = 1.0
            q_vec = q_vec / q_norm

            if self.index is not None:
                scores, indices = self.index.search(q_vec, len(self.chunks))
                for s, i in zip(scores[0], indices[0]):
                    if 0 <= i < len(self.chunks):
                        dense_scores[i] = float(s)
            else:
                dense_scores = np.dot(self.embeddings, q_vec[0])

        # Hybrid score: 70% semantic + 30% lexical/table boost
        if np.any(dense_scores):
            combined = 0.70 * dense_scores + 0.30 * lexical_scores
        else:
            combined = lexical_scores

        top_indices = np.argsort(combined)[::-1][:top_k]
        results = [(self.chunks[int(i)], float(combined[int(i)])) for i in top_indices]
        return results

    def _call_ollama(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Optional[str]:
        """Send a chat completion request to local Ollama."""
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        try:
            resp = requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("message", {}).get("content", "").strip()
        except Exception:
            pass
        return None

    def answer_question(
        self,
        question: str,
        reports: List[ParsedReport],
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = TOP_K_RESULTS,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant report chunks & tables, construct a grounded financial analyst prompt,
        and generate an insightful answer with citations.
        """
        retrieved = self.retrieve(question, top_k=top_k)

        context_blocks = []
        citations = []
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            context_blocks.append(f"--- Context [{rank}] ({chunk.source} | {chunk.page_or_sheet} | {chunk.chunk_type.upper()}) ---\n{chunk.text}")
            citations.append(
                {
                    "rank": rank,
                    "source": chunk.source,
                    "location": chunk.page_or_sheet,
                    "type": chunk.chunk_type,
                    "score": round(score, 3),
                    "snippet": chunk.text[:350] + ("..." if len(chunk.text) > 350 else ""),
                }
            )

        # Include extracted metrics summary so the LLM always has top-level numbers
        metrics_summary_lines = []
        for rep in reports:
            non_null_metrics = {k: v for k, v in rep.extracted_metrics.items() if v is not None}
            non_null_ratios = {k: v for k, v in rep.computed_ratios.items() if v is not None}
            if non_null_metrics or non_null_ratios:
                metrics_summary_lines.append(
                    f"Report `{rep.filename}` Key Metrics: {non_null_metrics} | Computed Ratios: {non_null_ratios}"
                )

        metrics_context = "\n".join(metrics_summary_lines)
        full_context = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a Senior Equity Research & Financial Analyst AI. "
            "Answer the user's question accurately using ONLY the provided financial report context and extracted metrics. "
            "Always follow these guidelines:\n"
            "1. Cite exact numbers, percentages, and table figures from the context.\n"
            "2. Reference the source and page/sheet in brackets, e.g., [Page 1] or [Sheet: Income Statement].\n"
            "3. Beyond stating raw numbers, provide brief financial insight (e.g., margin expansion/compression, YoY growth, liquidity health, or risk implications).\n"
            "4. If the answer cannot be determined from the provided report, state that clearly without guessing."
        )

        history_str = ""
        if chat_history:
            recent = chat_history[-4:]
            history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])

        user_prompt = (
            f"### Extracted High-Level Metrics & Ratios\n{metrics_context or 'None extracted automatically.'}\n\n"
            f"### Retrieved Document Chunks & Tables\n{full_context}\n\n"
            + (f"### Recent Conversation\n{history_str}\n\n" if history_str else "")
            + f"### User Question\n{question}\n\n"
            "Provide a clear, well-structured answer with direct citations and financial insight:"
        )

        llm_response = self._call_ollama(system_prompt, user_prompt, temperature=temperature)

        if not llm_response:
            # Rule-based analytical synthesis fallback if Ollama server is not running
            llm_response = self._fallback_answer(question, retrieved, reports)

        return {
            "answer": llm_response,
            "citations": citations,
        }

    def generate_executive_insights(self, reports: List[ParsedReport], temperature: float = 0.2) -> str:
        """Generate an automated Executive Summary & Financial Insight Report."""
        if not reports:
            return "No financial report loaded yet."

        summaries = []
        for rep in reports:
            metrics = {k: v for k, v in rep.extracted_metrics.items() if v is not None}
            ratios = {k: v for k, v in rep.computed_ratios.items() if v is not None}
            sample_chunks = "\n\n".join([c.text for c in rep.chunks[:6]])
            summaries.append(
                f"Document: {rep.filename}\n"
                f"Extracted Metrics: {metrics}\n"
                f"Computed Financial Ratios: {ratios}\n"
                f"Key Excerpts & Tables:\n{sample_chunks[:3500]}"
            )

        combined_input = "\n\n====================\n\n".join(summaries)
        system_prompt = (
            "You are a Chief Financial Officer (CFO) and Institutional Financial Analyst. "
            "Produce a concise, structured Executive Summary & Deep-Dive Insight Analysis based on the provided financial report data."
        )
        user_prompt = (
            f"{combined_input}\n\n"
            "Please generate an executive analysis covering:\n"
            "1. **Executive Headline & Overall Financial Health**\n"
            "2. **Profitability & Growth Analysis** (Revenue, Gross/Operating/Net Margins)\n"
            "3. **Balance Sheet, Liquidity & Leverage** (Assets, Liabilities, Current Ratio, Debt-to-Equity, ROE/ROA)\n"
            "4. **Key Risks, Watch Items & Strategic Takeaways**\n"
            "Use bullet points, bold key figures, and cite specific pages or tables."
        )

        llm_output = self._call_ollama(system_prompt, user_prompt, temperature=temperature)
        if llm_output:
            return llm_output

        return self._fallback_executive_summary(reports)

    def _fallback_executive_summary(self, reports: List[ParsedReport]) -> str:
        """Deterministic CFO-style insight analysis when Ollama is offline."""
        sections = [
            "> **Note:** *Local Ollama LLM was not detected at `"
            + self.ollama_base_url
            + "`. Below is the automated rule-based Financial Insight Analysis generated directly from the parsed statements and computed ratios.*"
        ]

        for rep in reports:
            m = rep.extracted_metrics
            r = rep.computed_ratios
            cs = getattr(rep, "currency_symbol", "$")
            ul = getattr(rep, "unit_label", "")
            unit_note = f" ({ul})" if ul else ""
            lines = [f"### 📊 Executive Insight Analysis: `{rep.filename}`{unit_note}"]

            # 1. Profitability
            lines.append("#### 1. Profitability & Operating Performance")
            rev = m.get("Revenue")
            gp = m.get("Gross Profit")
            op = m.get("Operating Income")
            ni = m.get("Net Income")
            gm = r.get("Gross Margin (%)")
            om = r.get("Operating Margin (%)")
            nm = r.get("Net Profit Margin (%)")

            if rev is not None:
                lines.append(f"- **Top-Line Revenue:** Reported at **{cs}{rev:,.2f}**{unit_note}.")
            if gp is not None and gm is not None:
                health = "strong pricing power & cost control" if gm >= 35 else ("moderate gross profitability" if gm >= 25 else "tight gross margin structure")
                lines.append(f"- **Gross Profit & Margin:** **{cs}{gp:,.2f}** (**{gm}%** gross margin), indicating {health}.")
            if op is not None and om is not None:
                lines.append(f"- **Operating Efficiency (EBIT):** **{cs}{op:,.2f}** (**{om}%** operating margin).")
            if ni is not None and nm is not None:
                ni_comment = "healthy bottom-line profitability" if nm >= 12 else ("positive net earnings" if nm > 0 else "net loss pressure")
                lines.append(f"- **Bottom-Line Net Income:** **{cs}{ni:,.2f}** (**{nm}%** net profit margin), reflecting {ni_comment}.")

            # 2. Liquidity & Solvency
            lines.append("\n#### 2. Balance Sheet, Liquidity & Leverage")
            cr = r.get("Current Ratio (x)")
            de = r.get("Debt-to-Equity (x)")
            roe = r.get("Return on Equity - ROE (%)")
            roa = r.get("Return on Assets - ROA (%)")

            if cr is not None:
                cr_status = "strong short-term liquidity buffer (>1.5x)" if cr >= 1.5 else ("adequate liquidity (1.0x–1.5x)" if cr >= 1.0 else "tight short-term working capital (<1.0x)")
                lines.append(f"- **Current Ratio:** **{cr}x** — indicates {cr_status}.")
            if de is not None:
                de_status = "conservative leverage (<1.0x)" if de < 1.0 else ("moderate financial leverage (1.0x–2.0x)" if de <= 2.0 else "elevated debt/payables leverage (>2.0x)")
                lines.append(f"- **Debt-to-Equity Ratio:** **{de}x** — reflects {de_status}.")
            if roe is not None or roa is not None:
                lines.append(
                    f"- **Capital Efficiency:** Return on Equity (ROE) of **{roe if roe is not None else 'N/A'}%** and Return on Assets (ROA) of **{roa if roa is not None else 'N/A'}%**."
                )

            # 3. Cash flow
            ocf = m.get("Operating Cash Flow")
            fcf = m.get("Free Cash Flow")
            if ocf is not None or fcf is not None:
                lines.append("\n#### 3. Cash Flow Generation")
                if ocf is not None:
                    lines.append(f"- **Operating Cash Flow:** **{cs}{ocf:,.2f}**{unit_note}")
                if fcf is not None:
                    lines.append(f"- **Free Cash Flow:** **{cs}{fcf:,.2f}**{unit_note}")

            sections.append("\n".join(lines))

        return "\n\n---\n\n".join(sections)

    def _fallback_answer(
        self,
        question: str,
        retrieved: List[Tuple[DocumentChunk, float]],
        reports: List[ParsedReport],
    ) -> str:
        """Synthesize an answer from retrieved chunks and extracted ratios when Ollama is unreachable."""
        lines = [
            "*(Local Ollama server is currently offline — showing synthesized insights from Hybrid Vector & Table Retrieval)*\n"
        ]

        # Check if any metric/ratio matches the query
        q_lower = question.lower()
        matched_items = []
        for rep in reports:
            for k, v in {**rep.extracted_metrics, **rep.computed_ratios}.items():
                if v is not None and any(word in q_lower for word in k.lower().split()):
                    matched_items.append(f"- **{k}** (`{rep.filename}`): **{v:,.2f}**")

        if matched_items:
            lines.append("#### 📌 Direct Metric & Ratio Matches")
            lines.extend(matched_items)
            lines.append("")

        if retrieved:
            lines.append("#### 📄 Top Grounded Excerpts & Structured Tables")
            for rank, (chunk, score) in enumerate(retrieved[:3], start=1):
                lines.append(
                    f"**[{rank}] {chunk.source} — {chunk.page_or_sheet}** *(Relevance: {score:.2f})*\n\n{chunk.text}\n"
                )
        else:
            lines.append("No matching sections were found in the uploaded report.")

        return "\n".join(lines)
