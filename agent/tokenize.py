from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter


WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_./%-]*|\d+(?:\.\d+)?%?")


def normalize_for_search(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    text = normalize_for_search(text)
    tokens: list[str] = WORD_RE.findall(text)
    chinese = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    tokens.extend(chinese)
    tokens.extend("".join(pair) for pair in zip(chinese, chinese[1:]))
    return tokens


def term_counts(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def bm25_score(query_terms: Counter[str], doc_terms: Counter[str], doc_len: int, avg_len: float, idf: dict[str, float]) -> float:
    if not query_terms or not doc_terms:
        return 0.0
    k1 = 1.4
    b = 0.75
    denom_len = k1 * (1 - b + b * doc_len / max(avg_len, 1.0))
    score = 0.0
    for term, qf in query_terms.items():
        tf = doc_terms.get(term, 0)
        if not tf:
            continue
        score += idf.get(term, 0.0) * (tf * (k1 + 1)) / (tf + denom_len) * math.sqrt(qf)
    return score

