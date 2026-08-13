from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .io_utils import load_json, read_text_guess, write_json, write_jsonl
from .preprocess import (
    RawDocument,
    SUPPORTED_EXTENSIONS,
    extract_document,
    split_text_with_spans,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_DIR = PROJECT_ROOT / "upload_b" / "question_b"
DEFAULT_SUBMIT_CSV = PROJECT_ROOT / "upload_b" / "submit.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "public_dataset_a" / "public_dataset_upload" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "processed_data_b"

_CACHE_VERSION = 1
_ANSWER_COLUMNS = tuple(f"answer_{index}" for index in range(1, 5))


@dataclass(frozen=True)
class BQuestion:
    qid: str
    domain: str
    split: str
    question: str
    type: str
    options: dict[str, str]
    answer_slots: int = 1
    answer_template: tuple[str, ...] = ()

    @property
    def qtype(self) -> str:
        """Compatibility alias for code that uses the A-list question name."""
        return self.type

    @property
    def template_answers(self) -> tuple[str, ...]:
        return self.answer_template

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "domain": self.domain,
            "split": self.split,
            "question": self.question,
            "type": self.type,
            "options": dict(self.options),
            "answer_slots": self.answer_slots,
            "answer_template": list(self.answer_template),
        }


@dataclass(frozen=True)
class BRawDocument:
    doc_id: str
    domain: str
    path: Path
    title: str
    relative_path: str

    def as_preprocess_document(self) -> RawDocument:
        return RawDocument(
            doc_id=self.doc_id,
            domain=self.domain,
            path=self.path,
            title=self.title,
        )


def _question_rows(path: Path) -> list[dict[str, Any]]:
    text = read_text_guess(path).lstrip("\ufeff")
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Question JSONL row must be an object: {path}:{line_number}"
                )
            rows.append(row)
        return rows

    payload = json.loads(text)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        rows = payload["questions"]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        raise ValueError(f"Unsupported question JSON payload in {path}")

    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Question file must contain JSON objects: {path}")
    return rows


def _load_submit_templates(
    submit_csv: Path,
) -> tuple[list[str], dict[str, tuple[int, tuple[str, ...]]]]:
    order: list[str] = []
    templates: dict[str, tuple[int, tuple[str, ...]]] = {}
    with submit_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [column for column in ("qid", *_ANSWER_COLUMNS) if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(
                f"Submission template is missing columns {missing_columns}: {submit_csv}"
            )

        for row in reader:
            qid = str(row.get("qid", "")).strip()
            if not qid or qid == "summary":
                continue
            if qid in templates:
                raise ValueError(f"Duplicate qid in submission template: {qid}")
            values = tuple(str(row.get(column, "") or "").strip() for column in _ANSWER_COLUMNS)
            used_indexes = [index for index, value in enumerate(values, start=1) if value]
            slots = max(used_indexes, default=1)
            order.append(qid)
            templates[qid] = (slots, values[:slots])
    return order, templates


def load_b_questions(
    questions_dir: Path = DEFAULT_QUESTIONS_DIR,
    submit_csv: Path | None = None,
) -> list[BQuestion]:
    """Load B-list JSON and JSONL questions with submission slot metadata.

    When ``submit_csv`` is omitted, ``questions_dir.parent / "submit.csv"`` is
    used if present. Questions are returned in submission-template order.
    """
    questions_dir = Path(questions_dir)
    if not questions_dir.is_dir():
        raise FileNotFoundError(f"B-list question directory not found: {questions_dir}")

    rows_by_qid: dict[str, dict[str, Any]] = {}
    source_by_qid: dict[str, Path] = {}
    question_paths = sorted(
        (
            path
            for path in questions_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        ),
        key=lambda path: path.name.casefold(),
    )
    if not question_paths:
        raise FileNotFoundError(f"No JSON or JSONL question files found in: {questions_dir}")

    for path in question_paths:
        for row in _question_rows(path):
            qid = str(row.get("qid", "")).strip()
            if not qid:
                raise ValueError(f"Question without qid in {path}")
            if qid in rows_by_qid:
                raise ValueError(
                    f"Duplicate question qid {qid!r} in {source_by_qid[qid]} and {path}"
                )
            rows_by_qid[qid] = row
            source_by_qid[qid] = path

    if submit_csv is None:
        candidate = questions_dir.parent / "submit.csv"
        submit_csv = candidate if candidate.is_file() else None
    elif submit_csv is not None:
        submit_csv = Path(submit_csv)

    template_order: list[str] = []
    templates: dict[str, tuple[int, tuple[str, ...]]] = {}
    if submit_csv is not None:
        if not submit_csv.is_file():
            raise FileNotFoundError(f"B-list submission template not found: {submit_csv}")
        template_order, templates = _load_submit_templates(submit_csv)
        missing_from_questions = [qid for qid in template_order if qid not in rows_by_qid]
        missing_from_template = [qid for qid in rows_by_qid if qid not in templates]
        if missing_from_questions or missing_from_template:
            raise ValueError(
                "Question/template qid mismatch: "
                f"missing_from_questions={missing_from_questions}, "
                f"missing_from_template={missing_from_template}"
            )

    ordered_qids = template_order or list(rows_by_qid)
    questions: list[BQuestion] = []
    for qid in ordered_qids:
        row = rows_by_qid[qid]
        domain = str(row.get("domain", "")).strip()
        question_text = str(row.get("question", "")).strip()
        qtype = str(row.get("type", "")).strip()
        options = row.get("options", {})
        if not domain or not question_text or not qtype:
            raise ValueError(
                f"Question {qid!r} must contain non-empty domain, question, and type"
            )
        if not isinstance(options, dict):
            raise ValueError(f"Question {qid!r} options must be an object")

        slots, answer_template = templates.get(qid, (1, ()))
        questions.append(
            BQuestion(
                qid=qid,
                domain=domain,
                split=str(row.get("split", "B") or "B"),
                question=question_text,
                type=qtype,
                options={str(key): str(value) for key, value in options.items()},
                answer_slots=slots,
                answer_template=answer_template,
            )
        )
    return questions


def _document_id(relative_path: Path) -> str:
    return f"b::{relative_path.as_posix()}"


def discover_b_raw_documents(raw_dir: Path = DEFAULT_RAW_DIR) -> list[BRawDocument]:
    """Discover every supported raw file without A-list stem deduplication."""
    raw_dir = Path(raw_dir).resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw document directory not found: {raw_dir}")

    documents: list[BRawDocument] = []
    seen_ids: set[str] = set()
    paths = sorted(
        (
            path
            for path in raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {extension.lower() for extension in SUPPORTED_EXTENSIONS}
        ),
        key=lambda path: path.relative_to(raw_dir).as_posix().casefold(),
    )
    for path in paths:
        relative_path = path.relative_to(raw_dir)
        parts = relative_path.parts
        domain = parts[0] if len(parts) > 1 else path.parent.name
        doc_id = _document_id(relative_path)
        identity = doc_id.casefold()
        if identity in seen_ids:
            raise ValueError(f"Non-unique B-list document id for path: {relative_path}")
        seen_ids.add(identity)
        documents.append(
            BRawDocument(
                doc_id=doc_id,
                domain=domain,
                path=path.resolve(),
                title=path.stem,
                relative_path=relative_path.as_posix(),
            )
        )
    return documents


def _cache_path(cache_dir: Path, doc_id: str) -> Path:
    digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _source_fingerprint(raw_doc: BRawDocument) -> dict[str, Any]:
    stat = raw_doc.path.stat()
    return {
        "cache_version": _CACHE_VERSION,
        "doc_id": raw_doc.doc_id,
        "relative_path": raw_doc.relative_path,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _valid_cached_pages(payload: Any, fingerprint: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
        return None
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return None
    normalized: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("text"), str):
            return None
        page_number = page.get("page")
        if page_number is not None and not isinstance(page_number, int):
            return None
        normalized.append({"page": page_number, "text": page["text"]})
    return normalized


def _extract_with_cache(
    raw_doc: BRawDocument,
    cache_dir: Path,
    use_cache: bool,
) -> tuple[list[dict[str, Any]], bool]:
    fingerprint = _source_fingerprint(raw_doc)
    cache_path = _cache_path(cache_dir, raw_doc.doc_id)
    if use_cache and cache_path.is_file():
        try:
            pages = _valid_cached_pages(load_json(cache_path), fingerprint)
        except (OSError, ValueError, TypeError):
            pages = None
        if pages is not None:
            return pages, True

    pages = extract_document(raw_doc.as_preprocess_document())
    normalized_pages = [
        {"page": page.get("page"), "text": str(page.get("text", ""))}
        for page in pages
        if str(page.get("text", "")).strip()
    ]
    if use_cache:
        write_json(
            cache_path,
            {
                "fingerprint": fingerprint,
                "pages": normalized_pages,
            },
        )
    return normalized_pages, False


def _count_by(items: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(items).items()))


def build_b_processed_data(
    questions_dir: Path = DEFAULT_QUESTIONS_DIR,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    submit_csv: Path | None = None,
    *,
    limit_docs: int | None = None,
    progress: bool = True,
    use_cache: bool = True,
    chunk_chars: int = 1800,
    overlap: int = 250,
) -> dict[str, Any]:
    """Build the independent B-list question, document, and chunk artifacts."""
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("overlap must be non-negative and smaller than chunk_chars")
    if limit_docs is not None and limit_docs < 0:
        raise ValueError("limit_docs must be non-negative")

    questions = load_b_questions(Path(questions_dir), submit_csv=submit_csv)
    raw_documents = discover_b_raw_documents(Path(raw_dir))
    selected_documents = raw_documents[:limit_docs] if limit_docs is not None else raw_documents

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = processed_dir / ".extract_cache"
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    total_pages = 0
    total_chars = 0

    for doc_index, raw_doc in enumerate(selected_documents, start=1):
        if progress:
            print(
                f"[{doc_index}/{len(selected_documents)}] extract "
                f"{raw_doc.relative_path}",
                flush=True,
            )
        pages, cache_hit = _extract_with_cache(raw_doc, cache_dir, use_cache)
        cache_hits += int(cache_hit)
        cache_misses += int(not cache_hit)
        char_count = sum(len(page["text"]) for page in pages)
        total_pages += len(pages)
        total_chars += char_count

        documents.append(
            {
                "doc_id": raw_doc.doc_id,
                "domain": raw_doc.domain,
                "title": raw_doc.title,
                "source_path": str(raw_doc.path),
                "relative_path": raw_doc.relative_path,
                "extension": raw_doc.path.suffix.lower(),
                "pages": len(pages),
                "chars": char_count,
            }
        )

        chunk_number = 0
        for page in pages:
            for part in split_text_with_spans(
                page["text"],
                chunk_chars=chunk_chars,
                overlap=overlap,
            ):
                chunk_number += 1
                chunks.append(
                    {
                        "chunk_id": f"{raw_doc.doc_id}#c{chunk_number:05d}",
                        "doc_id": raw_doc.doc_id,
                        "domain": raw_doc.domain,
                        "title": raw_doc.title,
                        "source_path": str(raw_doc.path),
                        "relative_path": raw_doc.relative_path,
                        "page": page["page"],
                        "page_char_start": part["page_char_start"],
                        "page_char_end": part["page_char_end"],
                        "block_type": part["block_type"],
                        "text": part["text"],
                    }
                )

    write_jsonl(processed_dir / "questions.jsonl", (question.to_dict() for question in questions))
    write_jsonl(processed_dir / "documents.jsonl", documents)
    write_jsonl(processed_dir / "chunks.jsonl", chunks)

    report: dict[str, Any] = {
        "questions": len(questions),
        "raw_documents": len(raw_documents),
        "selected_documents": len(selected_documents),
        "documents": len(documents),
        "pages": total_pages,
        "chars": total_chars,
        "chunks": len(chunks),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "domains": _count_by(document.domain for document in selected_documents),
        "extensions": _count_by(document.path.suffix.lower() for document in selected_documents),
        "questions_path": str((processed_dir / "questions.jsonl").resolve()),
        "documents_path": str((processed_dir / "documents.jsonl").resolve()),
        "chunks_path": str((processed_dir / "chunks.jsonl").resolve()),
    }
    write_json(processed_dir / "preprocess_report.json", report)
    return report


__all__ = [
    "BQuestion",
    "BRawDocument",
    "DEFAULT_PROCESSED_DIR",
    "DEFAULT_QUESTIONS_DIR",
    "DEFAULT_RAW_DIR",
    "DEFAULT_SUBMIT_CSV",
    "build_b_processed_data",
    "discover_b_raw_documents",
    "load_b_questions",
]
