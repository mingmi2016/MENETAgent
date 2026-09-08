"""Small persistent task store for the first Agent service version."""

import json
import sqlite3
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_USER_ID = "user_local"


class TaskStore:
    def __init__(self, database_path: str = "runs/agent.db"):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'user_local',
                    status TEXT NOT NULL,
                    task_json TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'user_local',
                    name TEXT NOT NULL DEFAULT '',
                    species TEXT NOT NULL DEFAULT '',
                    trait TEXT NOT NULL,
                    dataset_dir TEXT NOT NULL,
                    genotype_filename TEXT NOT NULL,
                    phenotype_filename TEXT NOT NULL,
                    genotype_size INTEGER NOT NULL,
                    phenotype_size INTEGER NOT NULL,
                    genotype_sha256 TEXT NOT NULL,
                    phenotype_sha256 TEXT NOT NULL,
                    source_url TEXT NOT NULL DEFAULT '',
                    license TEXT NOT NULL DEFAULT '',
                    is_demo INTEGER NOT NULL DEFAULT 0,
                    sample_count INTEGER,
                    snp_count INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'user_local',
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS trained_models (
                    model_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    trait TEXT NOT NULL,
                    dataset_id TEXT NOT NULL DEFAULT '',
                    dataset_dir TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    sample_count INTEGER,
                    snp_count INTEGER,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            self._ensure_column(connection, "tasks", "user_id", "TEXT NOT NULL DEFAULT 'user_local'")
            self._ensure_column(connection, "tasks", "started_at", "TEXT")
            self._ensure_column(connection, "tasks", "finished_at", "TEXT")
            self._ensure_column(connection, "datasets", "user_id", "TEXT NOT NULL DEFAULT 'user_local'")
            self._ensure_column(connection, "datasets", "name", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(connection, "datasets", "species", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(connection, "datasets", "source_url", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(connection, "datasets", "license", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(connection, "datasets", "is_demo", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(connection, "datasets", "sample_count", "INTEGER")
            self._ensure_column(connection, "datasets", "snp_count", "INTEGER")
            self._ensure_column(connection, "conversations", "user_id", "TEXT NOT NULL DEFAULT 'user_local'")
            now = self._now()
            connection.execute(
                "INSERT OR IGNORE INTO users(user_id,display_name,created_at,updated_at) VALUES(?,?,?,?)",
                (DEFAULT_USER_ID, "本地用户", now, now),
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_tasks_user_created ON tasks(user_id,created_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_datasets_user_created ON datasets(user_id,created_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id,updated_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_created ON messages(conversation_id,created_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_models_user_created ON trained_models(user_id,created_at)")

    def create_user(self, display_name: str) -> Dict[str, Any]:
        display_name = display_name.strip()
        if not display_name:
            raise ValueError("用户名称不能为空")
        user_id = f"user_{uuid4().hex[:12]}"
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO users(user_id,display_name,created_at,updated_at) VALUES(?,?,?,?)",
                (user_id, display_name, now, now),
            )
        return self.get_user(user_id) or {"user_id": user_id, "display_name": display_name}

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]

    def create(self, task: Dict[str, Any]) -> None:
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO tasks(task_id,user_id,status,task_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (task["task_id"], task.get("owner_user_id", DEFAULT_USER_ID), "created", json.dumps(task, ensure_ascii=False), now, now),
            )

    def update(self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None) -> None:
        now = self._now()
        started_at = now if status == "running" else None
        finished_at = now if status in {"completed", "failed", "cancelled"} else None
        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET status=?, result_json=?, updated_at=?, "
                "started_at=COALESCE(started_at,?), finished_at=COALESCE(?,finished_at) WHERE task_id=?",
                (status, json.dumps(result, ensure_ascii=False) if result is not None else None,
                 now, started_at, finished_at, task_id),
            )

    def get(self, task_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            query = "SELECT task_id,user_id,status,task_json,result_json,created_at,updated_at,started_at,finished_at FROM tasks WHERE task_id=?"
            params: tuple = (task_id,)
            if user_id:
                query += " AND user_id=?"
                params += (user_id,)
            row = connection.execute(query, params).fetchone()
        if row is None:
            return None
        return {
            "task_id": row[0], "user_id": row[1], "status": row[2], "task": json.loads(row[3]),
            "result": json.loads(row[4]) if row[4] else None,
            "created_at": row[5], "updated_at": row[6], "started_at": row[7], "finished_at": row[8],
        }

    def list_tasks(self, user_id: str = DEFAULT_USER_ID, limit: int = 50) -> list:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT task_id,user_id,status,task_json,result_json,created_at,updated_at,started_at,finished_at "
                "FROM tasks WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [{
            "task_id": row[0], "user_id": row[1], "status": row[2], "task": json.loads(row[3]),
            "result": json.loads(row[4]) if row[4] else None,
            "created_at": row[5], "updated_at": row[6], "started_at": row[7], "finished_at": row[8],
        } for row in rows]

    def register_model(self, task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        output_dir = Path(task["output_dir"])
        if not (output_dir / "menet_model.pt").is_file():
            raise ValueError("训练任务没有生成 MENET 模型文件")
        metrics = {}
        metrics_path = output_dir / "metrics.json"
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metrics = {}
        validation = next(
            (step.get("data", {}) for step in result.get("steps", []) if step.get("name") == "validate_dataset"),
            {},
        )
        model_id = f"model_{task['task_id'].removeprefix('task_')}"
        now = self._now()
        default_name = f"{task['trait']} · {now[:10]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO trained_models(model_id,task_id,user_id,name,trait,dataset_id,dataset_dir,output_dir,"
                "metrics_json,sample_count,snp_count,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(task_id) DO UPDATE SET metrics_json=excluded.metrics_json,sample_count=excluded.sample_count,"
                "snp_count=excluded.snp_count,updated_at=excluded.updated_at",
                (
                    model_id, task["task_id"], task.get("owner_user_id", DEFAULT_USER_ID), default_name,
                    task["trait"], task.get("metadata", {}).get("dataset_id", ""), task["dataset_dir"],
                    task["output_dir"], json.dumps(metrics, ensure_ascii=False),
                    validation.get("matched_sample_count"), validation.get("snp_count"), "active", now, now,
                ),
            )
        return self.get_model(model_id, task.get("owner_user_id", DEFAULT_USER_ID)) or {"model_id": model_id}

    def get_model(self, model_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM trained_models WHERE model_id=?"
        params: tuple = (model_id,)
        if user_id:
            query += " AND user_id=?"
            params += (user_id,)
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return self._model_record(row) if row else None

    def list_models(self, user_id: str, include_archived: bool = False, limit: int = 100) -> list:
        query = "SELECT * FROM trained_models WHERE user_id=?"
        params: tuple = (user_id,)
        if not include_archived:
            query += " AND status='active'"
        query += " ORDER BY created_at DESC LIMIT ?"
        params += (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._model_record(row) for row in rows]

    def rename_model(self, model_id: str, user_id: str, name: str) -> Optional[Dict[str, Any]]:
        name = name.strip()
        if not name:
            raise ValueError("模型名称不能为空")
        with self._connect() as connection:
            connection.execute(
                "UPDATE trained_models SET name=?,updated_at=? WHERE model_id=? AND user_id=?",
                (name, self._now(), model_id, user_id),
            )
        return self.get_model(model_id, user_id)

    def archive_model(self, model_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            connection.execute(
                "UPDATE trained_models SET status='archived',updated_at=? WHERE model_id=? AND user_id=?",
                (self._now(), model_id, user_id),
            )
        return self.get_model(model_id, user_id)

    def backfill_models(self) -> int:
        registered = 0
        for user in self.list_users():
            for record in self.list_tasks(user["user_id"], limit=1000):
                task = record["task"]
                if record["status"] != "completed" or task.get("intent") not in {"train_model", "generate_report"}:
                    continue
                if not (Path(task.get("output_dir", "")) / "menet_model.pt").is_file():
                    continue
                model_id = f"model_{task['task_id'].removeprefix('task_')}"
                if self.get_model(model_id) is None:
                    self.register_model(task, record.get("result") or {})
                    registered += 1
        return registered

    @staticmethod
    def _model_record(row: sqlite3.Row) -> Dict[str, Any]:
        value = dict(row)
        value["metrics"] = json.loads(value.pop("metrics_json") or "{}")
        return value

    def add_message(self, conversation_id: str, role: str, content: str) -> str:
        message_id = f"msg_{uuid4().hex[:12]}"
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages(message_id,conversation_id,role,content,created_at) VALUES(?,?,?,?,?)",
                (message_id, conversation_id, role, content, now),
            )
            connection.execute(
                "UPDATE conversations SET updated_at=? WHERE conversation_id=?",
                (now, conversation_id),
            )
        return message_id

    def history(self, conversation_id: str, limit: int = 12) -> list:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role,content,created_at FROM messages WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def create_dataset(self, dataset: Dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO datasets(dataset_id,user_id,name,species,trait,dataset_dir,genotype_filename,phenotype_filename,"
                "genotype_size,phenotype_size,genotype_sha256,phenotype_sha256,source_url,license,is_demo,sample_count,snp_count,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    dataset["dataset_id"], dataset.get("user_id", DEFAULT_USER_ID), dataset.get("name", ""),
                    dataset.get("species", ""), dataset["trait"], dataset["dataset_dir"],
                    dataset["genotype_filename"], dataset["phenotype_filename"],
                    dataset["genotype_size"], dataset["phenotype_size"],
                    dataset["genotype_sha256"], dataset["phenotype_sha256"], dataset.get("source_url", ""),
                    dataset.get("license", ""), int(bool(dataset.get("is_demo", False))),
                    dataset.get("sample_count"), dataset.get("snp_count"), self._now(),
                ),
            )

    def update_dataset_stats(self, dataset_dir: str, sample_count: int, snp_count: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE datasets SET sample_count=?,snp_count=? WHERE dataset_dir=?",
                (sample_count, snp_count, dataset_dir),
            )

    def update_dataset_metadata(self, dataset: Dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE datasets SET name=?,species=?,trait=?,dataset_dir=?,genotype_filename=?,phenotype_filename=?,"
                "genotype_size=?,phenotype_size=?,genotype_sha256=?,phenotype_sha256=?,source_url=?,license=?,"
                "is_demo=?,sample_count=?,snp_count=? WHERE dataset_id=?",
                (
                    dataset.get("name", ""), dataset.get("species", ""), dataset["trait"], dataset["dataset_dir"],
                    dataset["genotype_filename"], dataset["phenotype_filename"], dataset["genotype_size"],
                    dataset["phenotype_size"], dataset["genotype_sha256"], dataset["phenotype_sha256"],
                    dataset.get("source_url", ""), dataset.get("license", ""), int(bool(dataset.get("is_demo", False))),
                    dataset.get("sample_count"), dataset.get("snp_count"), dataset["dataset_id"],
                ),
            )

    def rebase_paths(self, replacements: Dict[str, str]) -> int:
        changed = 0
        with self._connect() as connection:
            for old, new in replacements.items():
                for table, column in (
                    ("datasets", "dataset_dir"),
                    ("tasks", "task_json"),
                    ("tasks", "result_json"),
                    ("conversations", "state_json"),
                ):
                    cursor = connection.execute(
                        f"UPDATE {table} SET {column}=replace({column},?,?) "
                        f"WHERE {column} IS NOT NULL AND instr({column},?)>0",
                        (old, new, old),
                    )
                    changed += cursor.rowcount
        return changed

    def get_dataset(self, dataset_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            query = "SELECT * FROM datasets WHERE dataset_id=?"
            params: tuple = (dataset_id,)
            if user_id:
                query += " AND (user_id=? OR is_demo=1)"
                params += (user_id,)
            row = connection.execute(query, params).fetchone()
        return dict(row) if row else None

    def list_datasets(self, user_id: str = DEFAULT_USER_ID, limit: int = 50) -> list:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM datasets WHERE user_id=? OR is_demo=1 ORDER BY is_demo DESC,created_at DESC LIMIT ?", (user_id, limit)
            ).fetchall()
        return [dict(row) for row in rows]

    def ensure_conversation(self, conversation_id: Optional[str] = None, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        if self.get_user(user_id) is None:
            raise ValueError("用户不存在")
        conversation_id = conversation_id or f"conv_{uuid4().hex[:12]}"
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO conversations(conversation_id,user_id,state_json,created_at,updated_at) VALUES(?,?,?,?,?)",
                (conversation_id, user_id, "{}", now, now),
            )
        conversation = self.get_conversation(conversation_id)
        if conversation and conversation["user_id"] != user_id:
            raise PermissionError("会话不属于当前用户")
        return conversation or {"conversation_id": conversation_id, "user_id": user_id, "state": {}}

    def get_conversation(self, conversation_id: str, message_limit: int = 50, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            query = "SELECT conversation_id,user_id,state_json,created_at,updated_at FROM conversations WHERE conversation_id=?"
            params: tuple = (conversation_id,)
            if user_id:
                query += " AND user_id=?"
                params += (user_id,)
            row = connection.execute(query, params).fetchone()
        if row is None:
            return None
        return {
            "conversation_id": row[0], "user_id": row[1], "state": json.loads(row[2]),
            "created_at": row[3], "updated_at": row[4],
            "messages": self.history(conversation_id, message_limit),
        }

    def list_conversations(self, user_id: str = DEFAULT_USER_ID, limit: int = 30) -> list:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.conversation_id,c.user_id,c.created_at,c.updated_at,
                    (SELECT content FROM messages first_message
                     WHERE first_message.conversation_id=c.conversation_id AND first_message.role='user'
                     ORDER BY first_message.created_at ASC LIMIT 1) AS title,
                    (SELECT content FROM messages last_message
                     WHERE last_message.conversation_id=c.conversation_id
                     ORDER BY last_message.created_at DESC LIMIT 1) AS last_message
                FROM conversations c
                WHERE c.user_id=?
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [{
            "conversation_id": row[0],
            "user_id": row[1],
            "created_at": row[2],
            "updated_at": row[3],
            "title": (row[4] or "新对话")[:60],
            "last_message": row[5] or "",
        } for row in rows]

    def update_conversation(self, conversation_id: str, changes: Dict[str, Any], user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        conversation = self.ensure_conversation(conversation_id, user_id)
        state = {**conversation.get("state", {}), **changes}
        with self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET state_json=?,updated_at=? WHERE conversation_id=?",
                (json.dumps(state, ensure_ascii=False), self._now(), conversation_id),
            )
        return state

    def recover_interrupted(self) -> int:
        """Mark jobs interrupted by a service restart as failed."""
        result = {
            "status": "failed",
            "errors": ["服务在任务执行期间重启，任务未自动恢复；请重新提交"],
        }
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET status='failed', result_json=?, updated_at=? "
                "WHERE status IN ('running','validating','preparing','training_encoder',"
                "'building_relatedness','training_menet','evaluating','explaining')",
                (json.dumps(result, ensure_ascii=False), self._now()),
            )
        return cursor.rowcount

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
