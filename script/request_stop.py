from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or clear the run_answer.py stop file.")
    parser.add_argument("--root", default=str(ROOT), help="Project root directory.")
    parser.add_argument("--stop-file", default="run_cache/STOP")
    parser.add_argument("--clear", action="store_true", help="Remove the stop file so --resume can continue.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    stop_file = resolve_path(root, args.stop_file)
    if args.clear:
        if stop_file.exists():
            stop_file.unlink()
            print(f"cleared: {stop_file}")
        else:
            print(f"already clear: {stop_file}")
        return

    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text("stop requested\n", encoding="utf-8")
    print(f"stop requested: {stop_file}")
    print("run_answer.py will stop before starting the next unfinished question.")


if __name__ == "__main__":
    main()
