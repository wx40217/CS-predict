# CS2比赛预测模型

基于HLTV API数据的CS:GO/CS2比赛预测系统，输入两支队伍的十名选手信息，输出各地图胜率和推荐地图选择。

## 🎯 功能特性

- **智能地图预测**: 为每张地图预测双方胜率
- **地图推荐系统**: 基于队伍实力分析推荐最优地图选择
- **年龄影响建模**: 考虑选手年龄对表现的影响
- **深度学习架构**: 使用注意力机制和集成学习
- **硬件优化**: 针对RTX 4080S + 9800X3D优化

## 🔧 硬件要求

- **CPU**: AMD Ryzen 9 9800X3D (或同等性能)
- **GPU**: NVIDIA RTX 4080 Super (12GB VRAM)
- **内存**: 48GB RAM (模型最多使用60%)
- **存储**: 至少10GB可用空间

## 📦 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 创建数据目录

```bash
mkdir -p data/raw data/processed models logs
```

### 3. 验证CUDA环境

```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU数量: {torch.cuda.device_count()}")
```

## 🚀 快速开始

### 1. 数据收集

```python
from data_collector import HLTVDataCollector

# 初始化数据收集器
collector = HLTVDataCollector()

# 收集队伍数据 (需要替换为实际队伍ID)
team_ids = [4608, 5995, 6665, 7020]  # G2, FaZe, NAVI, Astralis等
teams_df = collector.collect_team_data(team_ids)

# 收集选手数据
player_ids = [7998, 11893, 8183]  # s1mple, ZywOo, NiKo等
players_df = collector.collect_player_data(player_ids)
```

### 2. 数据预处理

```python
from data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()

# 预处理数据
players_processed = preprocessor.preprocess_player_data(players_df)
teams_processed = preprocessor.preprocess_team_data(teams_df)

# 保存预处理后的数据
preprocessor.save_preprocessed_data(players_processed, teams_processed)
```

### 3. 模型训练

```python
from trainer import CS2ModelTrainer

# 创建训练器
trainer = CS2ModelTrainer(use_ensemble=False)

# 开始训练
results = trainer.train(epochs=100)

print(f"训练完成！测试准确率: {results['test_accuracy']:.4f}")
```

### 4. 模型预测

```python
from predictor import CS2MatchPredictor_Inference, PlayerInfo, TeamInfo

# 创建预测器
predictor = CS2MatchPredictor_Inference()

# 定义队伍信息
team1_players = [
    PlayerInfo("s1mple", 26, 1.35, 1.28, 85.2),
    PlayerInfo("ZywOo", 23, 1.32, 1.25, 83.8),
    # ... 更多选手
]

team1 = TeamInfo(
    name="Team Liquid",
    players=team1_players,
    world_ranking=3,
    recent_winrate=0.72
)

# 进行预测
prediction = predictor.predict_match(team1, team2)
predictor.print_prediction_report(prediction)
```

## 📊 模型架构

### 核心组件

1. **队伍特征编码器**: 提取队伍层面特征
2. **选手特征聚合器**: 聚合五名选手的个人特征
3. **注意力机制**: 关注重要特征维度
4. **地图专用网络**: 为每张地图训练专门的预测网络
5. **胜率预测器**: 综合预测比赛胜率

### 特征体系

#### 选手特征
- HLTV 2.0评分 (rating_2_0)
- K/D比率 (kd_ratio)
- 平均伤害 (adr)
- 每回合击杀数 (kpr)
- 爆头率 (headshot_pct)
- 年龄及年龄影响系数
- 职业经验月数
- 近期状态评估

#### 队伍特征
- 世界排名及排名分数
- 近期比赛胜率
- 各地图胜率
- 平均选手评分
- 队伍默契度
- 地图优势方差

### 年龄影响建模

```python
def calculate_age_impact(age):
    """
    年龄影响系数计算:
    - 22岁为巅峰期 (系数1.0)
    - 22岁以下随年龄增长提升
    - 26岁后开始衰退，但经验可部分补偿
    """
    if age <= 22:
        return 0.8 + 0.2 * (age / 22)
    elif age <= 26:
        return 1.0
    else:
        decline = max(0.3, 1.0 - 0.05 * (age - 26))
        experience_bonus = min(0.2, 0.02 * (age - 26))
        return decline + experience_bonus
```

## ⚙️ 配置说明

### 模型配置 (config.py)

```python
MODEL_CONFIG = {
    "device": "cuda",
    "batch_size": 256,          # 适合RTX 4080S
    "max_epochs": 200,
    "learning_rate": 0.001,
    "dropout_rate": 0.3,
    "hidden_dims": [512, 256, 128, 64],
    "num_workers": 8,           # 充分利用9800X3D
    "mixed_precision": True,    # 提升训练效率
}
```

### 内存优化配置

```python
MEMORY_CONFIG = {
    "max_memory_usage": 0.6,    # 最多使用60%系统内存
    "cache_size": "8GB",        # 数据缓存大小
    "model_checkpoint_freq": 10, # 每10轮保存检查点
}
```

## 📈 训练监控

### 训练曲线可视化

训练过程会自动生成以下图表：
- 训练和验证损失曲线
- 验证准确率曲线  
- 学习率变化曲线

### 早停机制

- 验证损失连续15轮不下降时自动停止
- 学习率连续8轮不改善时自动衰减
- 自动保存最佳模型权重

### 日志记录

```
2024-01-15 10:30:15 - INFO - Epoch 25/200
2024-01-15 10:30:45 - INFO - 训练损失: 0.3245, 验证损失: 0.3891, 验证准确率: 0.7234
```

## 🎮 使用示例

### 完整预测流程

```python
# 1. 创建选手信息
navi_players = [
    PlayerInfo("s1mple", 26, 1.35, 1.28, 85.2, 0.85, 52.1, 60),
    PlayerInfo("electroNic", 26, 1.15, 1.08, 73.4, 0.69, 45.8, 72),
    PlayerInfo("Perfecto", 25, 1.05, 0.98, 68.9, 0.65, 44.1, 48),
    PlayerInfo("b1t", 21, 1.12, 1.05, 71.2, 0.67, 46.7, 18),
    PlayerInfo("Boombl4", 27, 0.95, 0.89, 65.3, 0.58, 42.5, 60)
]

# 2. 创建队伍信息
navi = TeamInfo(
    name="NAVI",
    players=navi_players,
    world_ranking=5,
    recent_winrate=0.65,
    map_winrates={
        "mirage": 0.72, "inferno": 0.68, "dust2": 0.71,
        "nuke": 0.58, "overpass": 0.65, "vertigo": 0.62, 
        "ancient": 0.69
    }
)

# 3. 进行预测
prediction = predictor.predict_match(liquid, navi)

# 4. 查看结果
predictor.print_prediction_report(prediction)
```

### 预测报告示例

```
============================================================
CS2比赛预测报告
============================================================
Team Liquid vs NAVI
============================================================

📊 各地图胜率预测 (Team Liquid获胜概率):
----------------------------------------
mirage     │ ████████████████░░░░ │  78.3%
dust2      │ ███████████████░░░░░ │  73.1%
ancient    │ ██████████████░░░░░░ │  69.7%
overpass   │ ████████████░░░░░░░░ │  62.4%
vertigo    │ ██████████░░░░░░░░░░ │  51.8%
inferno    │ ████████░░░░░░░░░░░░ │  42.6%
nuke       │ ██████░░░░░░░░░░░░░░ │  31.2%

🗺️  推荐地图选择:
------------------------------
Team Liquid优势地图: mirage, dust2, ancient
NAVI优势地图: nuke, inferno

⚔️  整体实力对比 (Team Liquid优势程度):
----------------------------------------
ranking_advantage    :  65.2% (中)
recent_form_advantage :  72.1% (强)
player_skill_advantage:  68.9% (中)
experience_advantage  :  58.3% (中)
overall_advantage     :  66.1% (中)

🎯 预测总结:
--------------------
Team Liquid 明显优势
```

## 🔧 高级功能

### 集成模型

```python
# 使用集成模型提高预测准确性
trainer = CS2ModelTrainer(use_ensemble=True, num_ensemble_models=5)
results = trainer.train()
```

### 超参数优化

```python
import optuna

def objective(trial):
    # 定义超参数搜索空间
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # 训练模型并返回验证准确率
    # ...
    
# 开始优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### 模型解释性

```python
# 特征重要性分析
def analyze_feature_importance(model, test_data):
    # 使用SHAP或其他方法分析特征重要性
    pass

# 预测置信度
def get_prediction_confidence(prediction):
    # 计算预测的置信度区间
    pass
```

## 🚨 注意事项

### API使用限制

1. **请求频率**: 建议每次请求间隔1秒以上
2. **数据更新**: HLTV数据更新频率不定，建议定期重新收集
3. **API稳定性**: 第三方API可能不稳定，建议实现重试机制

### 模型限制

1. **数据依赖**: 预测准确性高度依赖数据质量
2. **时效性**: 选手状态变化快，需要定期重训练
3. **样本偏差**: 训练数据可能存在时间或地区偏差

### 性能优化

1. **内存管理**: 大批量预测时注意内存使用
2. **GPU利用**: 确保CUDA正确配置
3. **并行处理**: 利用多核CPU进行数据预处理

## 📚 扩展开发

### 添加新特征

```python
# 在config.py中添加新特征
PLAYER_FEATURES.extend([
    'clutch_success_rate',  # 残局成功率
    'first_kill_rate',      # 首杀率
    'utility_damage'        # 道具伤害
])

# 在data_preprocessor.py中实现特征计算
def calculate_clutch_rate(player_data):
    # 实现残局成功率计算
    pass
```

### 支持新地图

```python
# 更新地图池
CS2_MAP_POOL.extend(['new_map_name'])

# 重新训练模型以支持新地图
trainer = CS2ModelTrainer()
trainer.train()
```

### 实时预测API

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
predictor = CS2MatchPredictor_Inference()

@app.route('/predict', methods=['POST'])
def predict_match():
    data = request.json
    # 解析队伍信息
    # 进行预测
    # 返回结果
    return jsonify(prediction_result)
```

## 🤝 贡献指南

1. Fork项目仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🆘 常见问题

### Q: 模型训练时间过长怎么办？
A: 可以减少训练轮数、增加批次大小或使用更少的集成模型数量。

### Q: 预测准确率不理想？
A: 检查数据质量、增加训练数据量、调整模型超参数或使用集成模型。

### Q: GPU内存不足？
A: 减少批次大小、关闭混合精度训练或使用梯度累积。

### Q: API请求被限制？
A: 增加请求间隔时间、使用代理或考虑缓存策略。

---

**开发者**: CS2预测模型团队  
**联系方式**: [your-email@example.com]  
**项目地址**: [https://github.com/your-username/cs2-match-predictor]