"""Worker entry point executed inside an LSF allocation."""
import argparse
import json
from .runner import run_file_task

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    # Accept old submitted commands across upgrades, but never open this path.
    parser.add_argument("--database", help=argparse.SUPPRESS)
    args = parser.parse_args()
    with open(args.task_file, encoding="utf-8") as handle:
        task = json.load(handle)
    result = run_file_task(task)
    if result["status"] == "failed":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
