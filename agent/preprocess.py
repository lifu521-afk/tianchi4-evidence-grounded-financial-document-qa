from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
import re
from pathlib import Path
from typing import Iterable


from .io_utils import load_json, read_text_guess, write_json, write_jsonl


SUPPORTED_EXTENSIONS = {".pdf", ".PDF", ".txt", ".html", ".htm"}


@dataclass(frozen=True)
class Question:
    qid: str
    domain: str
    split: str
    question: str
    options: dict[str, str]
    answer_format: str
    qtype: str
    doc_ids: list[str]


@dataclass(frozen=True)
class RawDocument:
    doc_id: str
    domain: str
    path: Path
    title: str


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip += 1
        if tag in {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "td", "th"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.skip:
            self.skip -= 1
        if tag in {"p", "div", "tr", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip:
            text = data.strip()
            if text:
                self.parts.append(text)

    def text(self) -> str:
        return normalize_text(" ".join(self.parts))


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_cell(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\n", " ")).strip()


TABLE_SETTING_CANDIDATES: tuple[dict[str, object], ...] = (
    {
        "vertical_strategy": "lines",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 5,
    },
    {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "text_tolerance": 3,
    },
)


def meaningful_row(row) -> list[str]:
    cells = [normalize_cell(cell) for cell in row or []]
    return [cell for cell in cells if cell]


def table_score(tables) -> int:
    score = 0
    for table in tables or []:
        for row in table or []:
            cells = meaningful_row(row)
            if len(cells) >= 2:
                score += len(cells)
    return score


def page_has_table_hints(text: str) -> bool:
    hinted_lines = 0
    for line in str(text or "").splitlines():
        clean = normalize_cell(line)
        if not clean:
            continue
        numeric_hits = len(re.findall(r"\d+(?:\.\d+)?%?|亿元|万元|元|AAA|AA\+?", clean, flags=re.I))
        layout_columns = len(re.split(r"\s{2,}", line.strip()))
        if numeric_hits >= 2 or layout_columns >= 3:
            hinted_lines += 1
        if hinted_lines >= 2:
            return True
    return False


def extract_tables_best_effort(page, page_text: str = "") -> list:
    if not page_has_table_hints(page_text):
        return []
    try:
        best_tables = page.extract_tables() or []
    except Exception:
        best_tables = []
    best_score = table_score(best_tables)
    if best_score >= 6:
        return best_tables
    for settings in TABLE_SETTING_CANDIDATES:
        try:
            tables = page.extract_tables(table_settings=settings)
        except Exception:
            continue
        score = table_score(tables)
        if score > best_score:
            best_tables = tables or []
            best_score = score
    return best_tables


def table_rows_to_text(table_index: int, table) -> str:
    rows: list[list[str]] = []
    seen: set[str] = set()
    for row in table or []:
        cells = meaningful_row(row)
        if len(cells) < 2:
            continue
        line_key = "\u241f".join(cells)
        if line_key in seen:
            continue
        seen.add(line_key)
        rows.append(cells)
    if len(rows) < 2:
        return ""

    two_column_rows = sum(1 for row in rows if len(row) == 2)
    key_value_table = len(rows[0]) == 2 and two_column_rows >= max(2, round(len(rows) * 0.75))
    header = rows[0] if not key_value_table and 2 <= len(rows[0]) <= 12 else []
    lines = [" | ".join(header)] if header else []
    for cells in rows[1:] if header else rows:
        row_text = " | ".join(cells)
        if header and len(cells) == len(header) and cells != header:
            pairs = [f"{name}:{value}" for name, value in zip(header, cells) if name and value and name != value]
            pair_text = "；".join(pairs)
            if pair_text and len(pair_text) <= 700:
                row_text = f"{row_text} || {pair_text}"
        lines.append(row_text)
    return f"[TABLE {table_index} rows={len(rows)}]\n" + "\n".join(lines)


def marginal_line_key(line: str) -> str:
    clean = normalize_cell(line)
    clean = re.sub(r"\d+", "#", clean)
    return re.sub(r"\s+", "", clean)


def strip_repeated_marginal_lines(pages: list[dict], scan_lines: int = 4) -> list[dict]:
    if len(pages) < 3:
        return pages

    counts: Counter[str] = Counter()
    page_lines: list[list[str]] = []
    for page in pages:
        lines = [line.rstrip() for line in str(page.get("text", "")).splitlines()]
        page_lines.append(lines)
        marginal = [line for line in (*lines[:scan_lines], *lines[-scan_lines:]) if normalize_cell(line)]
        counts.update({marginal_line_key(line) for line in marginal if len(marginal_line_key(line)) >= 2})

    threshold = max(3, min(len(pages), round(len(pages) * 0.35)))
    repeated = {key for key, count in counts.items() if count >= threshold}
    if not repeated:
        return pages

    cleaned_pages: list[dict] = []
    for page, lines in zip(pages, page_lines):
        kept: list[str] = []
        last_index = len(lines) - 1
        for idx, line in enumerate(lines):
            key = marginal_line_key(line)
            in_margin = idx < scan_lines or idx > last_index - scan_lines
            if in_margin and key in repeated and len(normalize_cell(line)) <= 90:
                continue
            kept.append(line)
        cleaned = dict(page)
        cleaned["text"] = "\n".join(kept)
        cleaned_pages.append(cleaned)
    return cleaned_pages


def load_questions(questions_dir: Path) -> list[Question]:
    questions: list[Question] = []
    for path in sorted(questions_dir.glob("*.json")):
        for row in load_json(path):
            questions.append(
                Question(
                    qid=row["qid"],
                    domain=row["domain"],
                    split=row.get("split", "A"),
                    question=row["question"],
                    options=dict(row.get("options", {})),
                    answer_format=row["answer_format"],
                    qtype=row.get("type", ""),
                    doc_ids=list(row.get("doc_ids", [])),
                )
            )
    return questions


def discover_raw_documents(raw_dir: Path) -> dict[str, RawDocument]:
    docs: dict[str, RawDocument] = {}
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        try:
            domain = path.relative_to(raw_dir).parts[0]
        except IndexError:
            continue
        doc_id = path.stem
        if doc_id in docs:
            existing = docs[doc_id]
            # Prefer directly readable text/html over PDF when duplicate stems exist.
            rank = {".txt": 0, ".html": 1, ".htm": 1, ".pdf": 2, ".PDF": 2}
            if rank.get(path.suffix, 9) >= rank.get(existing.path.suffix, 9):
                continue
        docs[doc_id] = RawDocument(
            doc_id=doc_id,
            domain=domain,
            path=path,
            title=path.stem,
        )
    return docs


def question_doc_ids(questions: Iterable[Question]) -> set[str]:
    ids: set[str] = set()
    for question in questions:
        ids.update(question.doc_ids)
    return ids


def extract_html(path: Path) -> list[dict]:
    parser = VisibleTextParser()
    parser.feed(read_text_guess(path))
    text = parser.text()
    return [{"page": None, "text": text}] if text else []


def extract_txt(path: Path) -> list[dict]:
    text = normalize_text(read_text_guess(path))
    return [{"page": None, "text": text}] if text else []


def extract_page_tables(page, page_text: str = "") -> str:
    blocks: list[str] = []
    for table_index, table in enumerate(extract_tables_best_effort(page, page_text), start=1):
        table_text = table_rows_to_text(table_index, table)
        if table_text:
            blocks.append(table_text)
    return "\n\n".join(blocks)


def extract_pdf(path: Path) -> list[dict]:
    try:
        import pdfplumber
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pdfplumber is required for PDF preprocessing. Install it with "
            "`python -m pip install pdfplumber pypdf`, or run the project with "
            "the Codex bundled Python that already has pdfplumber."
        ) from exc

    raw_pages: list[dict] = []
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=3) or ""
            table_text = extract_page_tables(page, text)
            raw_pages.append({"page": idx, "text": text, "table_text": table_text})

    pages: list[dict] = []
    for raw_page in strip_repeated_marginal_lines(raw_pages):
        text = normalize_text("\n\n".join(part for part in (raw_page.get("text", ""), raw_page.get("table_text", "")) if part))
        if text:
            pages.append({"page": raw_page["page"], "text": text})
    return pages


def extract_document(raw_doc: RawDocument) -> list[dict]:
    suffix = raw_doc.path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(raw_doc.path)
    if suffix in {".html", ".htm"}:
        return extract_html(raw_doc.path)
    if suffix == ".txt":
        return extract_txt(raw_doc.path)
    return []


def split_text(text: str, chunk_chars: int = 1800, overlap: int = 250) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= chunk_chars:
            current = (current + "\n\n" + paragraph).strip()
            continue
        if current:
            chunks.extend(split_long_segment(current, chunk_chars, overlap))
        current = paragraph
    if current:
        chunks.extend(split_long_segment(current, chunk_chars, overlap))
    return chunks


def split_text_with_spans(
    text: str,
    chunk_chars: int = 1800,
    overlap: int = 250,
) -> list[dict]:
    """Return chunks with best-effort page character offsets.

    Offsets are additive metadata for source auditing. Existing callers can keep
    using split_text(), while new preprocessing runs persist stable page spans.
    """
    parts = split_text(text, chunk_chars=chunk_chars, overlap=overlap)
    rows: list[dict] = []
    cursor = 0
    for part in parts:
        search_start = max(0, cursor - overlap)
        start = text.find(part, search_start)
        if start < 0:
            start = text.find(part)
        if start < 0:
            start = cursor
        end = min(len(text), start + len(part))
        rows.append(
            {
                "text": part,
                "page_char_start": start,
                "page_char_end": end,
                "block_type": "table_or_mixed" if "[TABLE" in part else "text",
            }
        )
        cursor = max(cursor, end)
    return rows


def split_long_segment(text: str, chunk_chars: int, overlap: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        cut = max(text.rfind("\n", start, end), text.rfind("。", start, end), text.rfind("；", start, end))
        if cut > start + chunk_chars // 2:
            end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_processed_data(
    questions_dir: Path,
    raw_dir: Path,
    processed_dir: Path,
    only_question_docs: bool = True,
    limit_docs: int | None = None,
    progress: bool = True,
) -> dict[str, int]:
    questions = load_questions(questions_dir)
    raw_docs = discover_raw_documents(raw_dir)
    needed = question_doc_ids(questions) if only_question_docs else set(raw_docs)
    missing = sorted(doc_id for doc_id in needed if doc_id not in raw_docs)
    selected_docs = [raw_docs[doc_id] for doc_id in sorted(needed) if doc_id in raw_docs]
    if limit_docs is not None:
        selected_docs = selected_docs[:limit_docs]

    documents: list[dict] = []
    chunks: list[dict] = []
    for doc_index, raw_doc in enumerate(selected_docs, start=1):
        if progress:
            print(f"[{doc_index}/{len(selected_docs)}] extract {raw_doc.doc_id} ({raw_doc.path.suffix})", flush=True)
        pages = extract_document(raw_doc)
        char_count = sum(len(page["text"]) for page in pages)
        documents.append(
            {
                "doc_id": raw_doc.doc_id,
                "domain": raw_doc.domain,
                "title": raw_doc.title,
                "path": str(raw_doc.path),
                "pages": len(pages),
                "chars": char_count,
                "extension": raw_doc.path.suffix,
            }
        )
        chunk_no = 0
        for page in pages:
            for part_info in split_text_with_spans(page["text"]):
                chunk_no += 1
                chunks.append(
                    {
                        "chunk_id": f"{raw_doc.doc_id}#c{chunk_no:04d}",
                        "doc_id": raw_doc.doc_id,
                        "domain": raw_doc.domain,
                        "title": raw_doc.title,
                        "source_path": str(raw_doc.path),
                        "page": page["page"],
                        "page_char_start": part_info["page_char_start"],
                        "page_char_end": part_info["page_char_end"],
                        "block_type": part_info["block_type"],
                        "text": part_info["text"],
                    }
                )

    processed_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        processed_dir / "questions.jsonl",
        [
            {
                "qid": q.qid,
                "domain": q.domain,
                "split": q.split,
                "question": q.question,
                "options": q.options,
                "answer_format": q.answer_format,
                "type": q.qtype,
                "doc_ids": q.doc_ids,
            }
            for q in questions
        ],
    )
    write_jsonl(processed_dir / "documents.jsonl", documents)
    write_jsonl(processed_dir / "chunks.jsonl", chunks)
    write_json(
        processed_dir / "preprocess_report.json",
        {
            "questions": len(questions),
            "raw_documents": len(raw_docs),
            "selected_documents": len(selected_docs),
            "chunks": len(chunks),
            "missing_doc_ids": missing,
        },
    )
    return {
        "questions": len(questions),
        "raw_documents": len(raw_docs),
        "selected_documents": len(selected_docs),
        "chunks": len(chunks),
        "missing": len(missing),
    }

