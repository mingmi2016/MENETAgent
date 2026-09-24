"""Worker entry point executed inside an LSF allocation."""
import argparse
import json
from .runner import run_task

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--database", required=True)
    args = parser.parse_args()
    with open(args.task_file, encoding="utf-8") as handle:
        task = json.load(handle)
    run_task(task, args.database)

if __name__ == "__main__":
    main()
