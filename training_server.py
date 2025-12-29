"""Gaussian Splatting Training Server - 高斯泼溅训练服务器"""
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse

from training_server_models import TaskStatus, PreprocessArgs, TrainArgs, ServerConfig
from training_server_state import state_manager
from training_server_db import db
from training_server_tasks import task_executor
from training_server_utils import init_logging, tail_file, cleanup_task


# 配置
config = ServerConfig()

# 确保必要的目录存在
os.makedirs(config.temp_dir, exist_ok=True)
os.makedirs(config.logs_dir, exist_ok=True)


# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    # 启动时初始化
    init_logging()
    db.initialize()
    recover_tasks()

    # 启动清理线程
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()

    yield

    # 关闭时清理（如果有需要）


# 创建FastAPI应用，使用lifespan
app = FastAPI(
    title="Gaussian Splatting Training Server",
    version="2.0.0",
    lifespan=lifespan
)

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")


def recover_tasks():
    """恢复任务：从数据库加载并恢复进行中的任务"""
    try:
        loaded_tasks, loaded_resources = db.load_all()

        # 恢复到状态管理器
        for task_id, task in loaded_tasks.items():
            state_manager.add_resource(task_id, loaded_resources.get(task_id))

            # 如果是进行中的任务，重新启动
            if task.status == TaskStatus.PREPROCESSING:
                recover_preprocessing(task_id)
            elif task.status == TaskStatus.TRAINING:
                recover_training(task_id)

        print(f"Recovered {len(loaded_tasks)} tasks")
    except Exception as e:
        print(f"Failed to recover tasks: {e}")


def recover_preprocessing(task_id: str):
    """恢复预处理任务"""
    resource = state_manager.get_resource(task_id)
    if resource and resource.video_path:
        # 清理可能存在的数据集
        if resource.dataset_path and Path(resource.dataset_path).exists():
            shutil.rmtree(resource.dataset_path)
            resource.dataset_path = None
            db.save_resource(resource)

        # 重新开始预处理
        task_executor.execute_preprocessing(
            task_id,
            resource.video_path,
            str(Path(config.temp_dir) / task_id),
            PreprocessArgs()
        )


def recover_training(task_id: str):
    """恢复训练任务"""
    resource = state_manager.get_resource(task_id)
    if resource and resource.dataset_path and Path(resource.dataset_path).exists():
        # 清理可能存在的输出
        if resource.output_path and Path(resource.output_path).exists():
            shutil.rmtree(resource.output_path)
            resource.output_path = None
            db.save_resource(resource)

        # 重新开始训练
        task_executor.execute_training(
            task_id,
            resource.dataset_path,
            TrainArgs(output=str(Path(config.temp_dir) / 'checkpoints'))
        )


def cleanup_loop(interval: int = 3600, ttl: int = 43200):
    """清理循环：定期清理旧的完成任务"""
    import time
    from datetime import datetime

    while True:
        time.sleep(interval)
        try:
            now = datetime.now()
            tasks = state_manager.get_all_tasks()

            for task_id, task in tasks.items():
                if task.status in [TaskStatus.FAILED, TaskStatus.TRAINING_COMPLETED]:
                    try:
                        updated_time = datetime.fromisoformat(task.updated_at)
                        if (now - updated_time).total_seconds() > ttl:
                            cleanup_thread = threading.Thread(
                                target=cleanup_task,
                                args=(task_id,),
                                daemon=True
                            )
                            cleanup_thread.start()
                    except Exception as e:
                        print(f"Cleanup check failed for {task_id}: {e}")

        except Exception as e:
            print(f"Cleanup loop error: {e}")


# ==================== API端点 ====================

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    max_frames: int = 150,
    min_frames: int = 30,
    start_training: bool = True
):
    """上传视频并启动训练流水线"""
    # 验证文件格式
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    if Path(file.filename).suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    # 生成任务ID
    task_id = str(uuid.uuid4())

    # 创建目录结构
    task_dir = Path(config.temp_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(config.temp_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 保存上传的视频
    video_path = task_dir / file.filename
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 创建任务和资源记录
    state_manager.create_task(task_id, "Video uploaded")

    from training_server_models import ResourceInfo
    resource = ResourceInfo(
        task_id=task_id,
        video_path=str(video_path)
    )
    state_manager.add_resource(task_id, resource)

    # 持久化到数据库
    task = state_manager.get_task(task_id)
    db.save_task(task)
    db.save_resource(resource)

    # 启动预处理和训练流水线
    preprocess_args = PreprocessArgs(max_frames=max_frames, min_frames=min_frames)
    train_args = TrainArgs(
        output=str(checkpoint_dir),
        start_training=start_training
    )

    task_executor.run_full_pipeline(
        task_id,
        str(video_path),
        str(task_dir),
        preprocess_args,
        train_args
    )

    return {
        "task_id": task_id,
        "message": "Video uploaded and pipeline started",
        "status": "pending",
        "auto_training": start_training
    }


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    task = state_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/download/{task_id}")
async def download_result(task_id: str):
    """下载训练结果（ZIP格式）"""
    task = state_manager.get_task(task_id)
    if not task or task.status != TaskStatus.TRAINING_COMPLETED:
        raise HTTPException(status_code=400, detail="Training not completed")

    resource = state_manager.get_resource(task_id)
    if not resource or not resource.output_path:
        raise HTTPException(status_code=500, detail="Output path not available")

    output_path = Path(resource.output_path)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output not found")

    # 创建临时ZIP文件
    temp_dir = tempfile.mkdtemp(prefix=f"zip_{task_id}_")
    zip_path = Path(temp_dir) / f"{task_id}_output.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), 'zip', root_dir=output_path)

    return FileResponse(
        str(zip_path),
        media_type='application/zip',
        filename=f"{task_id}_output.zip"
    )


@app.get("/tasks")
async def list_tasks():
    """列出所有任务ID"""
    return {"tasks": state_manager.list_tasks()}


@app.get("/logs/{task_id}")
async def get_task_logs(task_id: str, lines: int = 100):
    """获取任务日志"""
    log_path = Path(config.logs_dir) / task_id / 'training_server.log'
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    log_content = tail_file(log_path, lines)
    return PlainTextResponse(content=log_content, media_type='text/plain')


@app.get("/")
async def root():
    """提供Web UI"""
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    import sys

    print("Starting Training Server...")
    print(f"Python: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Config: {config}")

    uvicorn.run(
        "training_server:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        reload=False
    )