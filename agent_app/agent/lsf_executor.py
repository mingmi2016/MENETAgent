"""LSF submission and lifecycle helpers for MENET tasks."""
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

class LSFError(RuntimeError):
    pass

class LSFExecutor:
    def __init__(self, service_root: Path, database_path: str):
        self.service_root = service_root.resolve()
        self.database_path = str(Path(database_path).resolve())
        self.queue = os.environ.get("MENET_LSF_QUEUE", "gpu")
        self.host = os.environ.get("MENET_LSF_HOST", "gpu01")
        self.python = os.environ.get("MENET_LSF_PYTHON", os.environ.get("MENET_PYTHON", str(self.service_root.parent / "envs" / "torch_gpu" / "bin" / "python")))
        self.walltime = os.environ.get("MENET_LSF_WALLTIME", "24:00")

    def submit(self, task: Dict[str, Any]) -> str:
        output_dir = Path(task["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        task_file = output_dir / "task.json"
        task_file.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        stdout = output_dir / "lsf.stdout"
        stderr = output_dir / "lsf.stderr"
        job_name = f"menet_{task['task_id']}"
        core_root = self.service_root.parent / "MENET"
        command = (
            f"cd {self.service_root}; export MENET_CORE_ROOT={core_root}; "
            f"export PYTHONPATH={self.service_root}:{core_root}; "
            f"exec {self.python} -m agent.lsf_worker --task-file {task_file} --database {self.database_path}"
        )
        result = subprocess.run(
            ["bsub", "-q", self.queue, "-m", self.host, "-gpu", "num=1", "-W", self.walltime,
             "-J", job_name, "-oo", str(stdout), "-eo", str(stderr), command],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise LSFError((result.stderr or result.stdout).strip() or "bsub 提交失败")
        match = re.search(r"Job\s+<(\d+)>\s+is submitted", result.stdout)
        if not match:
            raise LSFError(f"无法解析 bsub 返回值: {result.stdout.strip()}")
        job_id = match.group(1)
        (output_dir / ".lsf_job_id").write_text(job_id, encoding="utf-8")
        return job_id

    def job_id(self, output_dir: str) -> Optional[str]:
        try:
            value = (Path(output_dir) / ".lsf_job_id").read_text(encoding="utf-8").strip()
            return value or None
        except OSError:
            return None

    def status(self, job_id: str) -> Optional[str]:
        result = subprocess.run(["bjobs", "-noheader", "-o", "stat", job_id], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return "DONE" if "not found" in (result.stderr or "").lower() else None
        value = result.stdout.strip().split()
        return value[0] if value else None

    def cancel(self, job_id: str) -> None:
        result = subprocess.run(["bkill", job_id], capture_output=True, text=True, check=False)
        if result.returncode != 0 and "not found" not in (result.stderr or "").lower():
            raise LSFError((result.stderr or result.stdout).strip() or f"bkill {job_id} 失败")
