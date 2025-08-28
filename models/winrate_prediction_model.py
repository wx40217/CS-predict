"""
胜率预测模型 - 基于深度神经网络
综合考虑队员实力、年龄、历史战绩等因素预测比赛胜率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class AgeFeatureExtractor(nn.Module):
    """年龄特征提取器 - 重点考虑年龄对竞技状态的影响"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 年龄分组嵌入
        age_weights = config['age_weights']
        self.age_groups = age_weights['age_groups']
        
        # 年龄特征提取
        self.age_embedding = nn.Embedding(5, 32)  # 5个年龄组
        
        # 年龄优势计算
        self.age_advantage = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # 经验差距计算
        self.experience_gap = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # 年轻因子计算
        self.youth_factor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # 老将因子计算
        self.veteran_factor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
    def get_age_group(self, ages: torch.Tensor) -> torch.Tensor:
        """将年龄映射到年龄组"""
        age_groups = torch.zeros_like(ages, dtype=torch.long)
        
        # 年轻选手 [16, 22]
        age_groups[(ages >= 16) & (ages <= 22)] = 0
        # 黄金年龄 [23, 27]
        age_groups[(ages >= 23) & (ages <= 27)] = 1
        # 老将 [28, 32]
        age_groups[(ages >= 28) & (ages <= 32)] = 2
        # 传奇选手 [33, 40]
        age_groups[(ages >= 33) & (ages <= 40)] = 3
        # 超龄选手 [40+]
        age_groups[ages > 40] = 4
        
        return age_groups
        
    def forward(self, ages: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取年龄相关特征"""
        batch_size = ages.shape[0]
        
        # 获取年龄组
        age_group_indices = self.get_age_group(ages)
        age_embeddings = self.age_embedding(age_group_indices)
        
        # 计算年龄优势
        age_advantage = self.age_advantage(age_embeddings)
        
        # 计算经验差距
        experience_gap = self.experience_gap(age_embeddings)
        
        # 计算年轻因子
        youth_factor = self.youth_factor(age_embeddings)
        
        # 计算老将因子
        veteran_factor = self.veteran_factor(age_embeddings)
        
        # 全局年龄特征
        global_age_features = torch.cat([
            age_advantage.mean(dim=1),
            experience_gap.mean(dim=1),
            youth_factor.mean(dim=1),
            veteran_factor.mean(dim=1)
        ], dim=1)
        
        return {
            'age_advantage': age_advantage,
            'experience_gap': experience_gap,
            'youth_factor': youth_factor,
            'veteran_factor': veteran_factor,
            'global_age_features': global_age_features
        }


class PlayerFeatureExtractor(nn.Module):
    """选手特征提取器"""
    
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 个人特征提取
        self.player_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 特征融合
        self.feature_fusion = nn.Linear(hidden_dim // 2, hidden_dim)
        
    def forward(self, player_features: torch.Tensor) -> torch.Tensor:
        """提取选手特征"""
        batch_size, num_players, _ = player_features.shape
        
        # 重塑为 [batch_size * num_players, feature_dim]
        x = player_features.view(-1, self.feature_dim)
        
        # 特征提取
        x = self.player_extractor(x)
        x = self.feature_fusion(x)
        
        # 重塑回 [batch_size, num_players, hidden_dim]
        x = x.view(batch_size, num_players, -1)
        
        return x


class TeamFeatureExtractor(nn.Module):
    """团队特征提取器"""
    
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 团队特征提取
        self.team_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 团队化学计算
        self.chemistry_calculator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, team_features: torch.Tensor) -> torch.Tensor:
        """提取团队特征"""
        x = self.team_extractor(team_features)
        chemistry = self.chemistry_calculator(x)
        
        return x, chemistry


class WinRatePredictionNN(nn.Module):
    """胜率预测深度神经网络"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 配置参数
        self.input_dim = config['architecture']['input_dim']
        self.hidden_dims = config['architecture']['hidden_dims']
        self.dropout = config['architecture']['dropout']
        self.activation = config['architecture']['activation']
        self.batch_norm = config['architecture']['batch_norm']
        
        # 特征维度
        player_features = config['features']['player_features']
        team_features = config['features']['team_features']
        matchup_features = config['features']['matchup_features']
        age_features = config['features']['age_features']
        
        self.player_feature_dim = len(player_features)
        self.team_feature_dim = len(team_features)
        self.matchup_feature_dim = len(matchup_features)
        self.age_feature_dim = len(age_features) * 8  # 每个年龄特征8维
        
        # 特征提取器
        self.player_extractor = PlayerFeatureExtractor(
            self.player_feature_dim, 
            self.hidden_dims[0]
        )
        
        self.team_extractor = TeamFeatureExtractor(
            self.team_feature_dim, 
            self.hidden_dims[0]
        )
        
        self.age_extractor = AgeFeatureExtractor(config)
        
        # 特征融合
        fusion_input_dim = (self.hidden_dims[0] * 2 +  # 选手和团队特征
                           self.age_feature_dim +       # 年龄特征
                           self.matchup_feature_dim)    # 对战特征
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 主网络
        layers = []
        input_dim = self.hidden_dims[0]
        
        for hidden_dim in self.hidden_dims[1:]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation())
            layers.append(nn.Dropout(self.dropout))
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.main_network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
        
    def _get_activation(self):
        """获取激活函数"""
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "gelu":
            return nn.GELU()
        elif self.activation == "swish":
            return nn.SiLU()
        else:
            return nn.ReLU()
            
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                
    def forward(self, player_features: torch.Tensor, team_features: torch.Tensor,
                matchup_features: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            player_features: [batch_size, 10, feature_dim] 选手特征
            team_features: [batch_size, team_feature_dim] 团队特征
            matchup_features: [batch_size, matchup_feature_dim] 对战特征
            ages: [batch_size, 10] 选手年龄
            
        Returns:
            胜率预测 [batch_size, 1]
        """
        batch_size = player_features.shape[0]
        
        # 提取选手特征
        player_emb = self.player_extractor(player_features)
        player_global = player_emb.mean(dim=1)  # 全局选手特征
        
        # 提取团队特征
        team_emb, chemistry = self.team_extractor(team_features)
        
        # 提取年龄特征
        age_features = self.age_extractor(ages)
        age_global = age_features['global_age_features']
        
        # 特征融合
        combined_features = torch.cat([
            player_global,
            team_emb.squeeze(1),
            age_global,
            matchup_features
        ], dim=1)
        
        # 特征融合
        fused_features = self.feature_fusion(combined_features)
        
        # 主网络预测
        winrate = self.main_network(fused_features)
        
        return winrate
    
    def predict_winrate(self, player_features: torch.Tensor, team_features: torch.Tensor,
                        matchup_features: torch.Tensor, ages: torch.Tensor) -> float:
        """
        预测胜率
        
        Returns:
            胜率概率 (0-1)
        """
        self.eval()
        with torch.no_grad():
            winrate = self.forward(player_features, team_features, 
                                 matchup_features, ages)
            return winrate.cpu().numpy()[0][0]


class WinRateLoss(nn.Module):
    """胜率预测损失函数"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 主要损失函数
        self.main_loss = nn.BCELoss()
        
        # 年龄相关损失（重点考虑年龄因素）
        self.age_loss = nn.MSELoss()
        
        # 损失权重
        self.main_weight = 0.8
        self.age_weight = 0.2
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                age_predictions: torch.Tensor = None, age_targets: torch.Tensor = None) -> torch.Tensor:
        """
        计算损失
        
        Args:
            predictions: 胜率预测
            targets: 真实胜率
            age_predictions: 年龄相关预测（可选）
            age_targets: 年龄相关目标（可选）
            
        Returns:
            总损失
        """
        # 主要损失
        main_loss = self.main_loss(predictions, targets)
        
        # 年龄相关损失
        age_loss = 0.0
        if age_predictions is not None and age_targets is not None:
            age_loss = self.age_loss(age_predictions, age_targets)
        
        # 总损失
        total_loss = self.main_weight * main_loss + self.age_weight * age_loss
        
        return total_loss


def create_winrate_model(config: Dict) -> WinRatePredictionNN:
    """创建胜率预测模型实例"""
    return WinRatePredictionNN(config)


def load_winrate_model(checkpoint_path: str, config: Dict) -> WinRatePredictionNN:
    """加载训练好的胜率预测模型"""
    model = create_winrate_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model