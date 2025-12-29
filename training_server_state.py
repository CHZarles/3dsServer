from typing import Dict, Optional
import threading
from datetime import datetime
from training_server_models import TaskInfo, ResourceInfo, TaskStatus


class StateManager:
    """状态管理器 - 负责任务和资源的线程安全管理"""

    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._resources: Dict[str, ResourceInfo] = {}
        self._lock = threading.Lock()

    def create_task(self, task_id: str, message: str = "Task created") -> TaskInfo:
        """创建新任务"""
        with self._lock:
            task = TaskInfo(
                task_id=task_id,
                status=TaskStatus.PENDING,
                message=message,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            self._tasks[task_id] = task
            return task

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        with self._lock:
            return self._tasks.get(task_id)

    def update_task_status(self, task_id: str, status: TaskStatus,
                          message: str, error: Optional[str] = None) -> bool:
        """更新任务状态"""
        with self._lock:
            if task_id not in self._tasks:
                return False

            self._tasks[task_id] = self._tasks[task_id].copy(
                update={
                    'status': status,
                    'message': message,
                    'updated_at': datetime.now().isoformat(),
                    'error_message': error
                }
            )
            return True

    def list_tasks(self) -> list[str]:
        """列出所有任务ID"""
        with self._lock:
            return list(self._tasks.keys())

    def get_all_tasks(self) -> Dict[str, TaskInfo]:
        """获取所有任务"""
        with self._lock:
            return self._tasks.copy()

    def add_resource(self, task_id: str, resource: ResourceInfo) -> bool:
        """添加资源信息"""
        with self._lock:
            self._resources[task_id] = resource
            return True

    def get_resource(self, task_id: str) -> Optional[ResourceInfo]:
        """获取资源信息"""
        with self._lock:
            return self._resources.get(task_id)

    def update_resource(self, task_id: str, **kwargs) -> bool:
        """更新资源信息"""
        with self._lock:
            if task_id not in self._resources:
                return False

            resource_dict = self._resources[task_id].dict()
            resource_dict.update(kwargs)
            self._resources[task_id] = ResourceInfo(**resource_dict)
            return True

    def remove_task(self, task_id: str) -> bool:
        """删除任务和资源"""
        with self._lock:
            removed = task_id in self._tasks
            self._tasks.pop(task_id, None)
            self._resources.pop(task_id, None)
            return removed


# 全局状态管理器实例
state_manager = StateManager()