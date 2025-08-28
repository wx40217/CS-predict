"""
选图预测模型 - 基于Transformer架构
根据10名队员的历史表现预测最优选图策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class PlayerEmbedding(nn.Module):
    """选手特征嵌入模块"""
    
    def __init__(self, feature_dim: int, embed_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        
        # 选手特征嵌入
        self.player_embedding = nn.Linear(feature_dim, embed_dim)
        
        # 年龄特殊处理（考虑年龄对竞技状态的影响）
        self.age_embedding = nn.Embedding(50, embed_dim // 4)  # 16-65岁
        
        # 经验年限嵌入
        self.experience_embedding = nn.Embedding(20, embed_dim // 4)  # 0-19年
        
        # 特征融合
        self.feature_fusion = nn.Linear(embed_dim + embed_dim // 2, embed_dim)
        
        # 批归一化
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        
    def forward(self, player_features: torch.Tensor, ages: torch.Tensor, 
                experience: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = player_features.shape
        
        # 基础特征嵌入
        x = self.player_embedding(player_features)
        
        # 年龄嵌入（考虑年龄对竞技状态的影响）
        age_indices = torch.clamp(ages - 16, 0, 49).long()
        age_emb = self.age_embedding(age_indices)
        
        # 经验嵌入
        exp_indices = torch.clamp(experience, 0, 19).long()
        exp_emb = self.experience_embedding(exp_indices)
        
        # 特征融合
        combined = torch.cat([x, age_emb, exp_emb], dim=-1)
        x = self.feature_fusion(combined)
        
        # 批归一化
        x = x.view(-1, self.embed_dim)
        x = self.batch_norm(x)
        x = x.view(batch_size, seq_len, self.embed_dim)
        
        return x


class MapPredictionTransformer(nn.Module):
    """选图预测Transformer模型"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 配置参数
        self.input_dim = config['architecture']['input_dim']
        self.hidden_dim = config['architecture']['hidden_dim']
        self.num_layers = config['architecture']['num_layers']
        self.num_heads = config['architecture']['num_heads']
        self.dropout = config['architecture']['dropout']
        self.max_seq_length = config['architecture']['max_seq_length']
        self.num_maps = config['maps']['num_maps']
        
        # 选手特征嵌入
        self.player_embedding = PlayerEmbedding(
            feature_dim=len(config['features']['player_stats']),
            embed_dim=self.input_dim
        )
        
        # 团队特征嵌入
        self.team_embedding = nn.Linear(
            len(config['features']['team_stats']), 
            self.input_dim
        )
        
        # 地图特征嵌入
        self.map_embedding = nn.Linear(
            len(config['features']['map_stats']), 
            self.input_dim
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.input_dim, self.max_seq_length)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 选图预测头
        self.map_predictor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_maps)
        )
        
        # 选图策略预测头
        self.strategy_predictor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 3)  # ban, pick, decider
        )
        
        # 初始化权重
        self._init_weights()
        
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
                map_features: torch.Tensor, ages: torch.Tensor, 
                experience: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            player_features: [batch_size, 10, feature_dim] 选手特征
            team_features: [batch_size, team_feature_dim] 团队特征
            map_features: [batch_size, map_feature_dim] 地图特征
            ages: [batch_size, 10] 选手年龄
            experience: [batch_size, 10] 选手经验年限
            
        Returns:
            包含选图概率和策略的字典
        """
        batch_size = player_features.shape[0]
        
        # 选手特征嵌入
        player_emb = self.player_embedding(player_features, ages, experience)
        
        # 团队特征嵌入
        team_emb = self.team_embedding(team_features).unsqueeze(1)
        
        # 地图特征嵌入
        map_emb = self.map_embedding(map_features).unsqueeze(1)
        
        # 特征拼接
        combined_features = torch.cat([player_emb, team_emb, map_emb], dim=1)
        
        # 位置编码
        combined_features = combined_features.transpose(0, 1)
        combined_features = self.pos_encoding(combined_features)
        combined_features = combined_features.transpose(0, 1)
        
        # Transformer编码
        encoded_features = self.transformer(combined_features)
        
        # 全局池化
        global_features = self.global_pool(encoded_features.transpose(1, 2))
        global_features = global_features.squeeze(-1)
        
        # 选图概率预测
        map_probs = self.map_predictor(global_features)
        map_probs = F.softmax(map_probs, dim=-1)
        
        # 选图策略预测
        strategy_probs = self.strategy_predictor(global_features)
        strategy_probs = F.softmax(strategy_probs, dim=-1)
        
        return {
            'map_probabilities': map_probs,
            'strategy_probabilities': strategy_probs,
            'encoded_features': encoded_features
        }
    
    def predict_maps(self, player_features: torch.Tensor, team_features: torch.Tensor,
                     map_features: torch.Tensor, ages: torch.Tensor, 
                     experience: torch.Tensor) -> Tuple[List[str], List[float]]:
        """
        预测选图策略
        
        Returns:
            recommended_maps: 推荐地图列表
            map_scores: 地图得分列表
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(player_features, team_features, map_features, 
                                 ages, experience)
            
            map_probs = outputs['map_probabilities'].cpu().numpy()
            strategy_probs = outputs['strategy_probabilities'].cpu().numpy()
            
            # 获取概率最高的地图
            map_indices = np.argsort(map_probs[0])[::-1]
            map_names = ['de_dust2', 'de_mirage', 'de_inferno', 'de_overpass', 
                        'de_nuke', 'de_vertigo', 'de_ancient']
            
            recommended_maps = [map_names[i] for i in map_indices]
            map_scores = [map_probs[0][i] for i in map_indices]
            
            return recommended_maps, map_scores


class MapPredictionLoss(nn.Module):
    """选图预测损失函数"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 地图预测损失
        self.map_loss = nn.CrossEntropyLoss()
        
        # 策略预测损失
        self.strategy_loss = nn.CrossEntropyLoss()
        
        # 特征一致性损失
        self.consistency_loss = nn.MSELoss()
        
        # 损失权重
        self.map_weight = 0.6
        self.strategy_weight = 0.3
        self.consistency_weight = 0.1
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算损失
        
        Args:
            predictions: 模型预测结果
            targets: 真实标签
            
        Returns:
            总损失
        """
        # 地图预测损失
        map_loss = self.map_loss(
            predictions['map_probabilities'], 
            targets['map_labels']
        )
        
        # 策略预测损失
        strategy_loss = self.strategy_loss(
            predictions['strategy_probabilities'], 
            targets['strategy_labels']
        )
        
        # 特征一致性损失（如果提供了）
        consistency_loss = 0.0
        if 'target_features' in targets:
            consistency_loss = self.consistency_loss(
                predictions['encoded_features'], 
                targets['target_features']
            )
        
        # 总损失
        total_loss = (self.map_weight * map_loss + 
                     self.strategy_weight * strategy_loss + 
                     self.consistency_weight * consistency_loss)
        
        return total_loss


def create_map_model(config: Dict) -> MapPredictionTransformer:
    """创建选图预测模型实例"""
    return MapPredictionTransformer(config)


def load_map_model(checkpoint_path: str, config: Dict) -> MapPredictionTransformer:
    """加载训练好的选图预测模型"""
    model = create_map_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model