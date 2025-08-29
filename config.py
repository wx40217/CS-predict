"""
配置文件 - CS:GO/CS2 比赛预测模型
适配硬件：AMD 9800X3D + RTX 4080S + 48GB RAM
"""
import os
from typing import Dict, List

# API配置 - 使用hltv-async-api
REQUEST_DELAY = 1.5  # 请求间隔(秒)，避免被API限制
MAX_RETRIES = 3
MAX_CONCURRENT_REQUESTS = 5  # 最大并发请求数

# 数据配置
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = "models"
LOGS_DIR = "logs"

# 地图池配置
CS2_MAP_POOL = [
    "mirage", "inferno", "dust2", "nuke", "overpass", "vertigo", "ancient"
]

# 特征配置
PLAYER_FEATURES = [
    "rating_2_0",      # HLTV 2.0评分
    "kd_ratio",        # K/D比率
    "adr",             # 平均伤害
    "kpr",             # 每回合击杀数
    "spr",             # 每回合生存数
    "headshot_pct",    # 爆头率
    "age",             # 年龄
    "experience_months", # 职业经验(月)
    "recent_form"      # 近期状态
]

TEAM_FEATURES = [
    "world_ranking",   # 世界排名
    "recent_matches_winrate", # 近期胜率
    "map_winrates",    # 各地图胜率
    "avg_player_rating", # 平均选手评分
    "team_chemistry",  # 队伍默契度(通过共同比赛场次计算)
]

# 模型配置 - 针对RTX 4080S优化
MODEL_CONFIG = {
    "device": "cuda",
    "batch_size": 256,  # 适中批次大小，留有余量
    "max_epochs": 200,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "dropout_rate": 0.3,
    "hidden_dims": [512, 256, 128, 64],  # 多层网络
    "num_workers": 8,  # 数据加载线程数，充分利用9800X3D
    "pin_memory": True,
    "mixed_precision": True,  # 使用混合精度训练，提升效率
}

# 训练配置
TRAINING_CONFIG = {
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 8,
    "min_lr": 1e-6,
    "gradient_clip_norm": 1.0,
}

# 年龄影响配置
AGE_IMPACT_CONFIG = {
    "peak_age": 22,      # 选手巅峰年龄
    "decline_start": 26, # 开始衰退年龄
    "experience_weight": 0.3, # 经验权重
    "reaction_weight": 0.7,   # 反应速度权重
}

# 内存优化配置 - 为其他软件预留资源
MEMORY_CONFIG = {
    "max_memory_usage": 0.6,  # 最大使用60%系统内存
    "cache_size": "8GB",      # 数据缓存大小
    "model_checkpoint_freq": 10, # 每10轮保存检查点
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(LOGS_DIR, "model.log"),
}