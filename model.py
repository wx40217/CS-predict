"""
深度学习模型架构 - CS:GO/CS2 比赛预测模型
针对RTX 4080S + 9800X3D优化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from config import MODEL_CONFIG, CS2_MAP_POOL

class MatchDataset(Dataset):
    """比赛数据集"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray = None, 
                 map_labels: np.ndarray = None):
        """
        Args:
            features: 比赛特征 [N, feature_dim]
            labels: 比赛结果标签 [N] (0: team2获胜, 1: team1获胜)
            map_labels: 地图标签 [N] (地图索引)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.map_labels = torch.LongTensor(map_labels) if map_labels is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        sample = {'features': self.features[idx]}
        
        if self.labels is not None:
            sample['label'] = self.labels[idx]
        
        if self.map_labels is not None:
            sample['map_label'] = self.map_labels[idx]
            
        return sample

class AttentionLayer(nn.Module):
    """注意力机制层，用于关注重要特征"""
    
    def __init__(self, input_dim: int, attention_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, x):
        # x: [batch_size, input_dim]
        attention_weights = F.softmax(self.attention(x), dim=1)
        return x * attention_weights

class TeamEncoder(nn.Module):
    """队伍特征编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.encoder(x)

class MapPredictor(nn.Module):
    """地图选择预测器"""
    
    def __init__(self, team_feature_dim: int, num_maps: int = len(CS2_MAP_POOL)):
        super().__init__()
        self.num_maps = num_maps
        
        # 为每张地图创建专门的网络
        self.map_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(team_feature_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(MODEL_CONFIG['dropout_rate']),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(MODEL_CONFIG['dropout_rate']),
                nn.Linear(128, 1)
            ) for _ in range(num_maps)
        ])
        
    def forward(self, team1_features, team2_features):
        # 合并两队特征
        combined_features = torch.cat([team1_features, team2_features], dim=1)
        
        # 为每张地图预测胜率
        map_predictions = []
        for map_net in self.map_networks:
            pred = torch.sigmoid(map_net(combined_features))
            map_predictions.append(pred)
        
        # [batch_size, num_maps]
        return torch.cat(map_predictions, dim=1)

class WinRatePredictor(nn.Module):
    """胜率预测器（给定地图）"""
    
    def __init__(self, team_feature_dim: int, map_embed_dim: int = 32):
        super().__init__()
        
        # 地图嵌入
        self.map_embedding = nn.Embedding(len(CS2_MAP_POOL), map_embed_dim)
        
        # 主预测网络
        input_dim = team_feature_dim * 2 + map_embed_dim
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, team1_features, team2_features, map_indices):
        # 获取地图嵌入
        map_embeds = self.map_embedding(map_indices)
        
        # 合并所有特征
        combined = torch.cat([team1_features, team2_features, map_embeds], dim=1)
        
        # 预测胜率 (team1获胜的概率)
        win_prob = torch.sigmoid(self.predictor(combined))
        return win_prob

class CS2MatchPredictor(nn.Module):
    """CS2比赛预测模型主体"""
    
    def __init__(self, input_feature_dim: int):
        super().__init__()
        
        # 假设输入特征被平分给两支队伍，加上对比特征
        team_feature_dim = (input_feature_dim - 2) // 2  # 减去2个对比特征
        
        # 队伍特征编码器
        self.team1_encoder = TeamEncoder(team_feature_dim)
        self.team2_encoder = TeamEncoder(team_feature_dim)
        
        # 注意力机制
        encoded_dim = 64  # TeamEncoder输出维度的一半
        self.attention1 = AttentionLayer(encoded_dim)
        self.attention2 = AttentionLayer(encoded_dim)
        
        # 地图选择预测器
        self.map_predictor = MapPredictor(encoded_dim)
        
        # 胜率预测器
        self.winrate_predictor = WinRatePredictor(encoded_dim)
        
        self.team_feature_dim = team_feature_dim
        self.encoded_dim = encoded_dim
        
    def forward(self, x, map_indices=None, predict_maps=False):
        batch_size = x.size(0)
        
        # 拆分输入特征
        team1_features = x[:, :self.team_feature_dim]
        team2_features = x[:, self.team_feature_dim:2*self.team_feature_dim]
        comparison_features = x[:, 2*self.team_feature_dim:]
        
        # 编码队伍特征
        team1_encoded = self.team1_encoder(team1_features)
        team2_encoded = self.team2_encoder(team2_features)
        
        # 应用注意力机制
        team1_attended = self.attention1(team1_encoded)
        team2_attended = self.attention2(team2_encoded)
        
        if predict_maps:
            # 预测地图选择偏好
            map_preferences = self.map_predictor(team1_attended, team2_attended)
            return map_preferences
        
        if map_indices is not None:
            # 预测指定地图的胜率
            win_probs = self.winrate_predictor(team1_attended, team2_attended, map_indices)
            return win_probs
        
        # 如果没有指定地图，返回所有地图的胜率
        all_map_probs = []
        for map_idx in range(len(CS2_MAP_POOL)):
            map_tensor = torch.full((batch_size,), map_idx, dtype=torch.long, device=x.device)
            prob = self.winrate_predictor(team1_attended, team2_attended, map_tensor)
            all_map_probs.append(prob)
        
        return torch.cat(all_map_probs, dim=1)  # [batch_size, num_maps]

class EnsembleModel(nn.Module):
    """集成模型，结合多个子模型提高预测准确性"""
    
    def __init__(self, input_feature_dim: int, num_models: int = 3):
        super().__init__()
        
        self.models = nn.ModuleList([
            CS2MatchPredictor(input_feature_dim) for _ in range(num_models)
        ])
        
        self.num_models = num_models
        
        # 集成权重学习
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, x, map_indices=None, predict_maps=False):
        predictions = []
        
        for model in self.models:
            pred = model(x, map_indices, predict_maps)
            predictions.append(pred)
        
        # 加权平均
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions))
        
        return ensemble_pred

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model: nn.Module, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=MODEL_CONFIG['learning_rate'],
            weight_decay=MODEL_CONFIG['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8, verbose=True
        )
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device).float()
            
            # 前向传播
            if 'map_label' in batch:
                map_indices = batch['map_label'].to(self.device)
                predictions = self.model(features, map_indices).squeeze()
            else:
                # 如果没有地图信息，预测所有地图的平均胜率
                all_map_preds = self.model(features)
                predictions = all_map_preds.mean(dim=1)
            
            # 计算损失
            loss = self.criterion(predictions, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                MODEL_CONFIG.get('gradient_clip_norm', 1.0)
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device).float()
                
                # 预测
                if 'map_label' in batch:
                    map_indices = batch['map_label'].to(self.device)
                    predictions = self.model(features, map_indices).squeeze()
                else:
                    all_map_preds = self.model(features)
                    predictions = all_map_preds.mean(dim=1)
                
                # 计算损失
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
                # 计算准确率
                predicted_labels = (predictions > 0.5).float()
                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy

if __name__ == "__main__":
    # 测试模型结构
    input_dim = 50  # 示例特征维度
    model = CS2MatchPredictor(input_dim)
    
    # 测试前向传播
    batch_size = 32
    test_input = torch.randn(batch_size, input_dim)
    
    # 测试地图预测
    map_prefs = model(test_input, predict_maps=True)
    print(f"地图偏好预测形状: {map_prefs.shape}")
    
    # 测试胜率预测
    map_indices = torch.randint(0, len(CS2_MAP_POOL), (batch_size,))
    win_probs = model(test_input, map_indices)
    print(f"胜率预测形状: {win_probs.shape}")
    
    print("模型结构测试完成")