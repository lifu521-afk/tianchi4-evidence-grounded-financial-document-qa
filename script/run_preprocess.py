from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import default_paths
from agent.preprocess import build_processed_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Tianchi financial long-document QA data.")
    parser.add_argument("--root", default=str(ROOT), help="Project root directory.")
    parser.add_argument("--all-docs", action="store_true", help="Process every raw document, not only docs referenced by A questions.")
    parser.add_argument("--limit-docs", type=int, default=None, help="Debug limit for number of documents to process.")
    parser.add_argument("--quiet", action="store_true", help="Disable per-document progress output.")
    args = parser.parse_args()

    paths = default_paths(args.root)
    report = build_processed_data(
        questions_dir=paths.questions_dir,
        raw_dir=paths.raw_dir,
        processed_dir=paths.processed_dir,
        only_question_docs=not args.all_docs,
        limit_docs=args.limit_docs,
        progress=not args.quiet,
    )
    print(report)


if __name__ == "__main__":
    main()
