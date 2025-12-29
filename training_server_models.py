from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    PREPROCESSING_COMPLETED = "preprocessing_completed"
    TRAINING = "training"
    TRAINING_COMPLETED = "training_completed"
    FAILED = "failed"
    DELETED = "deleted"


class TaskInfo(BaseModel):
    """任务信息模型"""
    task_id: str
    status: TaskStatus
    message: str
    created_at: str
    updated_at: str
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ResourceInfo(BaseModel):
    """资源信息模型"""
    task_id: str
    video_path: Optional[str] = None
    dataset_path: Optional[str] = None
    output_path: Optional[str] = None


class PreprocessArgs(BaseModel):
    """预处理参数"""
    max_frames: int = 150
    min_frames: int = 30


class TrainArgs(BaseModel):
    """训练参数"""
    output: str = "outputs"
    start_training: bool = True
    # 可扩展的训练参数
    iterations: Optional[int] = None
    resolution: Optional[int] = None


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    temp_dir: str = "temp_workspace"
    logs_dir: str = "logs"
    db_path: str = "training_server.db"
    # 超时配置（秒）
    preprocess_timeout: int = 1800  # 30分钟
    training_timeout: int = 18000  # 5小时
    # 清理配置
    cleanup_interval: int = 3600  # 1小时
    cleanup_ttl: int = 43200  # 12小时
