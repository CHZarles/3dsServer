"""任务执行 - 预处理和训练任务的具体实现"""
import threading
import logging
from pathlib import Path
from typing import Optional

from training_server_state import state_manager
from training_server_models import TaskStatus, PreprocessArgs, TrainArgs
from training_server_utils import update_task_status, execute_command, cleanup_task


logger = logging.getLogger(__name__)


class TaskPipeline:
    """任务执行流水线"""

    def __init__(self, config: Optional[TrainArgs] = None):
        self.config = config or TrainArgs()

    def run_preprocessing(self, task_id: str, video_path: str, output_dir: str,
                        args: PreprocessArgs) -> bool:
        """运行预处理任务"""
        try:
            update_task_status(task_id, TaskStatus.PREPROCESSING, "Starting preprocessing")

            # 构建预处理命令
            cmd = [
                "python", "run_pipeline.py",
                "--video_path", video_path,
                "--output_dir", output_dir,
                "--max_frames", str(args.max_frames),
                "--min_frames", str(args.min_frames)
            ]

            # 执行预处理
            if not execute_command(
                cmd,
                timeout=1800,
                task_id=task_id
            ):
                raise Exception("Frame extraction failed")

            # 更新状态和资源
            update_task_status(task_id, TaskStatus.PREPROCESSING_COMPLETED,
                             "Preprocessing completed")

            dataset_path = str(Path(output_dir) / "images_undistorted")
            resource = state_manager.get_resource(task_id)
            if resource:
                resource.dataset_path = dataset_path
                from training_server_db import db
                db.save_resource(resource)
                logger.info(f"Updated dataset path for {task_id}: {dataset_path}")

            return True

        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            logger.error(error_msg)
            update_task_status(task_id, TaskStatus.FAILED, "Preprocessing failed", error_msg)
            return False

    def run_training(self, task_id: str, dataset_path: str, args: TrainArgs) -> bool:
        """运行训练任务"""
        try:
            update_task_status(task_id, TaskStatus.TRAINING, "Starting training")

            # 准备输出路径
            output_path = args.output
            final_output_path = str(Path(output_path) / task_id)

            # 构建训练命令
            cmd = [
                "python", "main.py",
                "fit",
                "--data.path", dataset_path,
                "--name", task_id,
                "--output", output_path
            ]

            # 执行训练
            if not execute_command(
                cmd,
                timeout=18000,
                task_id=task_id
            ):
                raise Exception("Training failed")

            # 更新状态和资源
            update_task_status(task_id, TaskStatus.TRAINING_COMPLETED,
                             "Training completed")

            resource = state_manager.get_resource(task_id)
            if resource:
                resource.output_path = final_output_path
                from training_server_db import db
                db.save_resource(resource)
                logger.info(f"Updated output path for {task_id}: {final_output_path}")

            return True

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(error_msg)
            update_task_status(task_id, TaskStatus.FAILED, "Training failed", error_msg)
            return False


# ==================== 任务执行器 ====================

class TaskExecutor:
    """任务执行器 - 负责任务的调度和执行"""

    def __init__(self):
        self.pipeline = TaskPipeline()

    def execute_preprocessing(self, task_id: str, video_path: str,
                            output_dir: str, args: PreprocessArgs) -> None:
        """执行预处理任务（在单独线程中）"""
        def _run():
            success = self.pipeline.run_preprocessing(task_id, video_path, output_dir, args)
            if not success:
                logger.error(f"Preprocessing failed for task {task_id}")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def execute_training(self, task_id: str, dataset_path: str, args: TrainArgs) -> None:
        """执行训练任务（在单独线程中）"""
        def _run():
            success = self.pipeline.run_training(task_id, dataset_path, args)
            if not success:
                logger.error(f"Training failed for task {task_id}")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def run_full_pipeline(self, task_id: str, video_path: str,
                         output_dir: str, args: PreprocessArgs,
                         train_args: Optional[TrainArgs] = None) -> None:
        """运行完整的预处理+训练流水线"""
        train_args = train_args or TrainArgs()

        def _full_run():
            # 执行预处理
            if not self.pipeline.run_preprocessing(task_id, video_path, output_dir, args):
                return

            # 预处理完成后，如果配置了则自动开始训练
            if train_args.start_training:
                resource = state_manager.get_resource(task_id)
                if resource and resource.dataset_path:
                    self.execute_training(task_id, resource.dataset_path, train_args)

        thread = threading.Thread(target=_full_run, daemon=True)
        thread.start()


# 全局任务执行器
task_executor = TaskExecutor()


# ==================== 兼容性函数（旧代码接口） ====================

def preprocessing_task(task_id: str, video_path: str, output_dir: str, args: PreprocessArgs):
    """预处理任务（兼容性接口）"""
    task_executor.execute_preprocessing(task_id, video_path, output_dir, args)


def training_task(task_id: str, dataset_path: str, args: TrainArgs):
    """训练任务（兼容性接口）"""
    task_executor.execute_training(task_id, dataset_path, args)