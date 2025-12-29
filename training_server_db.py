"""SQLite持久化层 - 提供任务和资源的数据库存储"""
import json
import sqlite3
import threading
from typing import Dict, Tuple, Optional
from training_server_models import TaskInfo, ResourceInfo


class Database:
    """数据库管理器 - 线程安全的SQLite操作"""

    def __init__(self, db_path: str = "training_server.db"):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialize(self) -> None:
        """初始化数据库表，自动迁移旧表结构"""
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()

            # 创建新表（如果不存在）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT,
                    message TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resources (
                    task_id TEXT PRIMARY KEY,
                    video_path TEXT,
                    dataset_path TEXT,
                    output_path TEXT
                )
            """)

            # 迁移：为旧表添加缺失的列
            cursor.execute("PRAGMA table_info(tasks)")
            existing_columns = {row[1] for row in cursor.fetchall()}

            if 'metadata' not in existing_columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN metadata TEXT")

            conn.commit()

    def load_all(self) -> Tuple[Dict[str, TaskInfo], Dict[str, ResourceInfo]]:
        """加载所有任务和资源"""
        conn = self._get_connection()
        tasks = {}
        resources = {}

        with self._lock:
            cursor = conn.cursor()

            for row in cursor.execute("SELECT * FROM tasks"):
                task = self._row_to_task_info(row)
                tasks[task.task_id] = task

            for row in cursor.execute("SELECT * FROM resources"):
                resource = self._row_to_resource_info(row)
                resources[resource.task_id] = resource

        return tasks, resources

    def save_task(self, task: TaskInfo) -> None:
        """保存或更新任务"""
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tasks
                (task_id, status, message, created_at, updated_at, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_id,
                task.status.value,
                task.message,
                task.created_at,
                task.updated_at,
                task.error_message,
                json.dumps(task.metadata) if task.metadata else None
            ))
            conn.commit()

    def save_resource(self, resource: ResourceInfo) -> None:
        """保存或更新资源"""
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO resources (task_id, video_path, dataset_path, output_path)
                VALUES (?, ?, ?, ?)
            """, (resource.task_id, resource.video_path, resource.dataset_path, resource.output_path))
            conn.commit()

    def delete_task(self, task_id: str) -> None:
        """删除任务和资源"""
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            cursor.execute("DELETE FROM resources WHERE task_id = ?", (task_id,))
            conn.commit()

    @staticmethod
    def _row_to_task_info(row) -> TaskInfo:
        """数据库行转换为TaskInfo"""
        return TaskInfo(
            task_id=row["task_id"],
            status=row["status"],
            message=row["message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error_message=row["error_message"],
            metadata=json.loads(row["metadata"]) if row.get("metadata") else None
        )

    @staticmethod
    def _row_to_resource_info(row) -> ResourceInfo:
        """数据库行转换为ResourceInfo"""
        return ResourceInfo(
            task_id=row["task_id"],
            video_path=row["video_path"],
            dataset_path=row["dataset_path"],
            output_path=row["output_path"]
        )


# 全局数据库实例
db = Database()
