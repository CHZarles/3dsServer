"""工具函数 - 日志、命令执行、任务清理"""
import os
import shutil
import subprocess
import logging
import queue
import re
import sys
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Optional

from training_server_state import state_manager
from training_server_models import TaskStatus


# ==================== 日志系统 ====================

_log_queue: queue.Queue = queue.Queue()
_queue_listener: Optional[QueueListener] = None


def init_logging(log_file: str = 'training_server.log') -> None:
    """初始化中央化日志系统"""
    global _queue_listener
    if _queue_listener is not None:
        return

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.StreamHandler):
            try:
                root_logger.removeHandler(handler)
            except Exception:
                pass

    if not any(isinstance(h, QueueHandler) for h in root_logger.handlers):
        root_logger.addHandler(QueueHandler(_log_queue))

    _queue_listener = QueueListener(_log_queue, stream_handler, file_handler)
    _queue_listener.start()


def get_task_logger(task_id: str) -> logging.Logger:
    """获取任务专用日志器"""
    log_dir = Path('logs') / task_id
    log_dir.mkdir(parents=True, exist_ok=True)

    logger_name = f"task.{task_id}"
    task_logger = logging.getLogger(logger_name)
    task_logger.setLevel(logging.INFO)
    task_logger.propagate = False

    if not task_logger.handlers:
        handler = logging.FileHandler(log_dir / 'training_server.log', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        task_logger.addHandler(handler)

    return task_logger


# ==================== 命令执行 ====================

def execute_command(cmd: list[str], timeout: int = 36000, task_id: Optional[str] = None) -> bool:
    """执行命令并输出日志"""
    task_logger = get_task_logger(task_id) if task_id else logging.getLogger(__name__)
    task_logger.info(f"Executing: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )

        for line in proc.stdout or []:
            line = line.rstrip('\r\n')
            if not line.strip():
                continue

            if re.search(r"ERROR|WARN|EXCEPT|FATAL|Exception", line, re.I):
                task_logger.error(line)
            else:
                task_logger.info(line)

        proc.wait(timeout=timeout)
        success = proc.returncode == 0

        if success:
            task_logger.info("Command executed successfully")
        else:
            task_logger.error(f"Command failed with return code {proc.returncode}")

        return success

    except subprocess.TimeoutExpired:
        if proc:
            proc.kill()
        task_logger.error(f"Command timed out after {timeout} seconds")
        return False
    except Exception as e:
        task_logger.error(f"Command execution failed: {e}")
        return False


# ==================== 任务状态更新 ====================

def update_task_status(task_id: str, status: TaskStatus, message: str,
                      error: Optional[str] = None) -> bool:
    """更新任务状态并持久化"""
    if not state_manager.update_task_status(task_id, status, message, error):
        return False

    task_logger = logging.getLogger(__name__)
    task_logger.info(f"Task {task_id}: {status.value} - {message}")
    if error:
        task_logger.error(f"Task {task_id} error: {error}")

    # 持久化到数据库
    try:
        from training_server_db import db
        task = state_manager.get_task(task_id)
        if task:
            db.save_task(task)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to persist task {task_id}: {e}")

    return True


# ==================== 任务清理 ====================

def cleanup_task(task_id: str) -> None:
    """清理任务相关文件和状态"""
    logger = logging.getLogger(__name__)

    try:
        resource = state_manager.get_resource(task_id)
        if not resource:
            logger.warning(f"No resource found for task {task_id}")
            return

        # 清理视频文件
        if resource.video_path and Path(resource.video_path).exists():
            try:
                os.unlink(resource.video_path)
                logger.info(f"Removed video: {resource.video_path}")
            except Exception as e:
                logger.warning(f"Failed to remove video: {e}")

        # 清理数据集目录
        if resource.dataset_path and Path(resource.dataset_path).exists():
            try:
                shutil.rmtree(resource.dataset_path)
                logger.info(f"Removed dataset: {resource.dataset_path}")
            except Exception as e:
                logger.warning(f"Failed to remove dataset: {e}")

        # 清理输出目录
        if resource.output_path and Path(resource.output_path).exists():
            try:
                shutil.rmtree(resource.output_path)
                logger.info(f"Removed output: {resource.output_path}")
            except Exception as e:
                logger.warning(f"Failed to remove output: {e}")

        # 从状态管理器移除
        state_manager.remove_task(task_id)
        logger.info(f"Cleanup completed for task {task_id}")

        # 从数据库删除
        try:
            from training_server_db import db
            db.delete_task(task_id)
        except Exception as e:
            logger.warning(f"Failed to delete task from DB: {e}")

    except Exception as e:
        logger.error(f"Cleanup failed for task {task_id}: {e}")


# ==================== 文件工具 ====================

def tail_file(path: Path, lines: int = 200) -> str:
    """高效读取文件末尾N行"""
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            to_read = min(lines * 200, file_size)
            f.seek(max(0, file_size - to_read))
            data = f.read().decode(errors='replace')
            all_lines = data.splitlines()
            return '\n'.join(all_lines[-lines:] if len(all_lines) > lines else all_lines)
    except Exception:
        return ""
