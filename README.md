# CS比赛选图与胜率预测模型

基于HLTV数据的CS比赛选图和胜率预测AI模型，能够根据两支队伍的十名队员信息，预测比赛选图策略和双方胜率。

## 功能特性

- **选图预测**: 基于队员历史表现和地图偏好，预测最优选图策略
- **胜率预测**: 综合考虑队员实力、年龄、历史战绩等因素，预测比赛胜率
- **实时数据**: 集成HLTV API，获取最新比赛和选手数据
- **年龄因素**: 考虑选手年龄对竞技状态的影响

## 系统要求

- **硬件配置**: 推荐配置（适合您的9800X3D + 4080S + 48G内存）
  - CPU: AMD Ryzen 7 9800X3D 或更高
  - GPU: NVIDIA RTX 4080S 或更高（支持CUDA）
  - 内存: 32GB+ RAM
  - 存储: 50GB+ 可用空间

- **软件环境**:
  - Python 3.9+
  - CUDA 12.0+ (用于GPU加速)
  - PyTorch 2.0+

## 安装方法

```bash
# 克隆项目
git clone <repository-url>
cd cs-prediction-model

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 使用方法

### 1. 数据收集与预处理

```bash
# 收集HLTV数据
python data_collection.py --start_date 2023-01-01 --end_date 2024-01-01

# 预处理数据
python data_preprocessing.py --input_dir ./raw_data --output_dir ./processed_data
```

### 2. 模型训练

```bash
# 训练选图预测模型
python train_map_model.py --config configs/map_model.yaml --gpu 0

# 训练胜率预测模型
python train_winrate_model.py --config configs/winrate_model.yaml --gpu 0
```

### 3. 模型推理

```python
from models.predictor import CSPredictor

# 初始化预测器
predictor = CSPredictor(
    map_model_path="./checkpoints/map_model.pth",
    winrate_model_path="./checkpoints/winrate_model.pth"
)

# 预测比赛结果
team1_players = ["ZywOo", "shox", "apEX", "misutaaa", "Kyojin"]
team2_players = ["s1mple", "electronic", "Boombl4", "Perfecto", "flamie"]

result = predictor.predict(team1_players, team2_players)
print(f"推荐选图: {result['recommended_maps']}")
print(f"胜率预测: Team1 {result['winrate_team1']:.2%}, Team2 {result['winrate_team2']:.2%}")
```

### 4. 批量预测

```bash
# 批量预测多场比赛
python batch_predict.py --input_file matches.csv --output_file predictions.csv
```

## 模型架构

### 选图预测模型
- **输入**: 10名队员的历史地图表现、地图偏好、团队配合度
- **架构**: Transformer + 多头注意力机制
- **输出**: 7张地图的选图概率分布

### 胜率预测模型
- **输入**: 队员个人数据、团队数据、历史对战记录
- **架构**: 深度神经网络 + 特征融合
- **输出**: 双方胜率概率

## 性能优化

针对您的硬件配置进行了以下优化：

- **GPU加速**: 使用PyTorch CUDA后端，充分利用RTX 4080S
- **内存优化**: 批处理大小适配48G内存
- **多进程**: 数据预处理使用多进程加速
- **模型量化**: 支持INT8量化，减少内存占用

## 配置文件

主要配置文件位于 `configs/` 目录：

- `map_model.yaml`: 选图模型训练配置
- `winrate_model.yaml`: 胜率模型训练配置
- `data_config.yaml`: 数据处理配置
- `inference_config.yaml`: 推理配置

## 数据格式

### 输入数据格式
```json
{
  "team1": {
    "players": [
      {
        "nickname": "ZywOo",
        "age": 21,
        "rating": 1.33,
        "maps_played": 1066,
        "kd": 1.38
      }
    ]
  },
  "team2": { ... }
}
```

### 输出数据格式
```json
{
  "recommended_maps": ["de_dust2", "de_mirage", "de_inferno"],
  "winrate_team1": 0.65,
  "winrate_team2": 0.35,
  "confidence": 0.82
}
```

## 监控与评估

- **训练监控**: 使用TensorBoard监控训练过程
- **模型评估**: 提供详细的评估指标和可视化
- **性能基准**: 与HLTV官方数据对比验证

## 注意事项

1. 首次运行需要下载大量历史数据，请确保网络稳定
2. 模型训练时间取决于数据量和硬件配置
3. 建议定期更新模型以适应最新的比赛数据
4. 年龄因素权重可根据实际需求调整

## 许可证

MIT License