# Gaussian Splatting Training Server 

一个基于 FastAPI 的视频转3D模型训练服务器



## 🏗️ 系统架构

### 组件结构
```
training_server.py (主服务器)
├── training_server_models.py (数据模型)
├── training_server_db.py (SQLite持久化)
├── training_server_state.py (状态管理)
├── training_server_tasks.py (任务执行)
└── training_server_utils.py (工具函数)
└── static/index.html (Web界面)
```

### 业务流程

**视频上传 → 预处理 → 训练 → 下载结果**

#### 📤 上传阶段
- 用户上传视频文件
- 生成唯一任务ID
- 保存视频到：`temp_workspace/{task_id}/`
- 训练输出到：`temp_workspace/checkpoints/`

#### 🔧 预处理阶段
- 使用 `run_pipeline.py` 处理视频
- 提取帧图像到：`{task_dir}/images_undistorted/`
- 状态更新为 `PREPROCESSING_COMPLETED`

#### 🧠 训练阶段
- 使用 `main.py` 进行训练
- 训练输出到：`temp_workspace/checkpoints/{task_id}/`
- 状态更新为 `TRAINING_COMPLETED`

### 状态流转
```
PENDING → PREPROCESSING → PREPROCESSING_COMPLETED / FAILED
                      ↓
                   TRAINING → TRAINING_COMPLETED / FAILED
```

## 📁 目录结构

```
temp_workspace/
├── {task_id}/
│   └── {video_file} (上传的视频)
└── checkpoints/
    └── {task_id}/ (训练输出)
        └── [模型文件]
logs/
└── {task_id}/
    └── training_server.log (任务日志)
```

## ⚙️ 配置参数

```python
ServerConfig:
- host: "0.0.0.0"
- port: 8000
- temp_dir: "temp_workspace"
- logs_dir: "logs"
- db_path: "training_server.db"
- preprocess_timeout: 1800秒 (30分钟)
- training_timeout: 18000秒 (5小时)
- cleanup_ttl: 43200秒 (12小时)
```

## 🔌 API 接口

### 核心接口
- `POST /upload` - 上传视频并启动训练
- `GET /status/{task_id}` - 查询任务状态
- `GET /download/{task_id}` - 下载训练结果(ZIP)
- `GET /tasks` - 列出所有任务
- `GET /logs/{task_id}` - 查看任务日志
- `GET /` - 提供Web界面

### 响应示例

#### 上传响应
```json
{
  "task_id": "uuid-string",
  "message": "Video uploaded and pipeline started",
  "status": "pending",
  "auto_training": true
}
```

#### 任务状态
```json
{
  "task_id": "uuid-string",
  "status": "training",
  "message": "Starting training",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:00",
  "error_message": null,
  "metadata": null
}
```

## 🌐 Web界面

访问 `http://localhost:8000` 查看Web界面，包含：
- 视频上传表单（配置帧数参数）
- 任务列表显示
- 实时状态查询
- 任务日志查看
- 结果下载功能

## 🚀 快速启动

```bash
# 安装依赖
pip install fastapi uvicorn

# 启动服务器
python training_server.py
```
## 📝 注意事项

1. clone gaussian-splatting-lightning, 然后将本项目内容拷贝到gaussian-splatting-lightning运行

