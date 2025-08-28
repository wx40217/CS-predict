## HLTV Map Pick & Winrate Model (Scaffold)

本项目提供一个完整的训练/推理脚手架：给定两支队伍的5名队员（共10人），预测一局比赛中最可能被选择的地图（pick），并给出双方在各张地图上的胜率。你将在本地完成训练与推理。

### 特性
- 使用 `https://hltv-api.vercel.app/` 提供的数据，带简单缓存
- 预处理包含年龄特征（age 与 age^2），数值统计缺失自动填0
- 模型结构：玩家编码器（PlayerEncoder）→ 队伍聚合（TeamAggregator）→ 多头输出（地图选择 softmax + 各图胜率 sigmoid）
- 训练：PyTorch + AMP 混合精度，适配 9800X3D + RTX 4080 Super
- 断点保存与简单日志

### 硬件假设
- CPU: 9800X3D
- GPU: RTX 4080 Super (16GB)
- 内存: ~48GB

默认参数对上述配置较为友好，可在 `config/config.yaml` 中调整。

### 环境安装
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 数据抓取（缓存到 data/raw）
```bash
python -m src.data.fetch_hltv --out_dir data/raw --max_matches 5000
```
说明：HLTV API社区镜像字段可能变化，抓取器做了多种字段名兼容与重试；如抓取很少，请检查网络或适当调大 `--max_matches`。

### 预处理（生成 data/processed/samples.parquet）
```bash
python -m src.data.preprocess \
  --raw_dir data/raw \
  --out_dir data/processed \
  --min_date 2022-01-01
```
输出样本包含：
- 10名选手拼接后的特征向量（含年龄/年龄平方），长度固定
- 地图标签（7选1：ancient, anubis, inferno, mirage, nuke, overpass, vertigo）
- 胜率监督：仅在实际打出的地图处有监督（其余为-1遮蔽）

### 训练
```bash
python -m src.train --config config/config.yaml \
  train.batch_size=256 train.num_workers=12 train.precision=amp \
  train.max_epochs=30 paths.processed_dir=data/processed
```
- 最佳模型保存在 `checkpoints/best.pt`

### 推理
准备10名选手的名单，示例见 `examples/ten_players.json`：
```json
{
  "team_a": ["s1mple", "rain", "broky", "ropz", "karrigan"],
  "team_b": ["ZywOo", "apEX", "flameZ", "Spinx", "mezii"]
}
```
运行：
```bash
python -m src.infer \
  --ckpt checkpoints/best.pt \
  --players_file examples/ten_players.json \
  --raw_dir data/raw
```
输出包含：
- predicted_pick：预测的选图
- pick_probabilities：每张地图被选中的概率
- win_probabilities：每张地图A/B双方的胜率

### 模型结构说明（仅设计与框架，训练由你本地完成）
- PlayerEncoder（多层感知机 + LayerNorm + GELU + Dropout），对单人特征编码
- TeamAggregator：对每队5人取均值后线性投影，得到队A、队B表征
- 融合：concat(A,B) → MLP → 得到对局上下文向量
- 两个输出头：
  - pick_head：softmax输出7图选择分布
  - win_head：对每张图输出A/B胜率（sigmoid）
- 损失：`CE(pick)` + `0.5 * BCE(win on played map)`（未打出的图以掩码忽略）

### 年龄影响的考虑
- 在特征中显式加入 `age` 与 `age^2`（截断到 [15,40]），让模型可学习年龄与表现的潜在非线性关系
- 你可继续加入最近三个月 rating、角色、地图专长等特征以提升效果

### 目录结构
- `src/data/fetch_hltv.py`：抓取并缓存 `matches.jsonl`、`players.jsonl`、`teams.jsonl`
- `src/data/preprocess.py`：生成 `data/processed/samples.parquet`
- `src/data/dataset.py`：加载数据集并切分
- `src/model.py`：模型定义
- `src/train.py`：训练脚本（AMP/Checkpoint）
- `src/infer.py`：推理脚本（10名选手 → 选图与胜率）
- `config/config.yaml`：默认配置

### 备注
- API字段不稳定时，请在 `src/data/preprocess.py` 内调整字段映射。
- 若想进一步提升性能：可引入注意力聚合、更丰富的对手/地图交互特征、Elo/MMR等先验。

### 许可证
MIT