"""
模型训练和推理框架
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import os
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from model import CS2MatchPredictor, EnsembleModel, MatchDataset, ModelTrainer
from data_preprocessor import DataPreprocessor
from config import (
    MODEL_CONFIG, TRAINING_CONFIG, MODEL_DIR, LOGS_DIR, 
    PROCESSED_DATA_DIR, CS2_MAP_POOL
)

class CS2ModelTrainer:
    """CS2比赛预测模型训练器"""
    
    def __init__(self, use_ensemble: bool = False, num_ensemble_models: int = 3):
        # 创建必要目录
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(LOGS_DIR, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 设备配置
        self.device = MODEL_CONFIG['device'] if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"使用设备: {self.device}")
        
        # 模型配置
        self.use_ensemble = use_ensemble
        self.num_ensemble_models = num_ensemble_models
        
        # 数据预处理器
        self.preprocessor = DataPreprocessor()
        
        # 训练历史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """加载和准备训练数据"""
        self.logger.info("加载训练数据...")
        
        # 这里需要根据实际数据结构调整
        # 示例：从预处理的数据创建训练集
        
        # 加载预处理后的选手和队伍数据
        try:
            players_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'players_processed.csv'))
            teams_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'teams_processed.csv'))
            
            # 这里需要实现从历史比赛数据生成训练样本的逻辑
            # 简化版本：生成模拟数据用于演示
            features, labels, map_labels = self._create_training_samples(players_df, teams_df)
            
        except FileNotFoundError:
            self.logger.warning("未找到预处理数据，生成模拟数据用于演示")
            features, labels, map_labels = self._generate_mock_data()
        
        # 创建数据集
        dataset = MatchDataset(features, labels, map_labels)
        
        # 划分训练/验证/测试集
        train_size = int(TRAINING_CONFIG['train_split'] * len(dataset))
        val_size = int(TRAINING_CONFIG['val_split'] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=MODEL_CONFIG['batch_size'],
            shuffle=True,
            num_workers=MODEL_CONFIG['num_workers'],
            pin_memory=MODEL_CONFIG['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=MODEL_CONFIG['batch_size'],
            shuffle=False,
            num_workers=MODEL_CONFIG['num_workers'],
            pin_memory=MODEL_CONFIG['pin_memory']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=MODEL_CONFIG['batch_size'],
            shuffle=False,
            num_workers=MODEL_CONFIG['num_workers'],
            pin_memory=MODEL_CONFIG['pin_memory']
        )
        
        self.logger.info(f"数据加载完成 - 训练集: {len(train_dataset)}, "
                        f"验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def _create_training_samples(self, players_df: pd.DataFrame, 
                               teams_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从实际数据创建训练样本"""
        # 这里需要实现具体的样本生成逻辑
        # 基于历史比赛数据，为每场比赛生成特征向量和标签
        
        # 简化实现：基于队伍数据生成配对样本
        features_list = []
        labels_list = []
        map_labels_list = []
        
        # 生成队伍间的配对
        team_ids = teams_df['team_id'].values
        
        for i, team1_id in enumerate(team_ids):
            for j, team2_id in enumerate(team_ids):
                if i >= j:  # 避免重复配对
                    continue
                
                team1_data = teams_df[teams_df['team_id'] == team1_id].iloc[0].to_dict()
                team2_data = teams_df[teams_df['team_id'] == team2_id].iloc[0].to_dict()
                
                # 获取队伍选手（简化：使用平均数据）
                team1_players = self._get_team_players(team1_id, players_df)
                team2_players = self._get_team_players(team2_id, players_df)
                
                # 为每张地图生成样本
                for map_idx, map_name in enumerate(CS2_MAP_POOL):
                    # 生成特征向量
                    feature_vector = self.preprocessor.create_match_features(
                        team1_data, team2_data, team1_players, team2_players, map_name
                    )
                    
                    # 基于队伍实力生成标签（简化逻辑）
                    team1_strength = team1_data.get('ranking_score', 0.5)
                    team2_strength = team2_data.get('ranking_score', 0.5)
                    
                    # 添加随机性
                    win_prob = team1_strength / (team1_strength + team2_strength)
                    win_prob += np.random.normal(0, 0.1)  # 添加噪声
                    label = 1 if win_prob > 0.5 else 0
                    
                    features_list.append(feature_vector)
                    labels_list.append(label)
                    map_labels_list.append(map_idx)
        
        return (np.array(features_list), 
                np.array(labels_list), 
                np.array(map_labels_list))
    
    def _get_team_players(self, team_id: int, players_df: pd.DataFrame) -> List[Dict]:
        """获取队伍选手数据"""
        team_players = players_df[players_df['team_id'] == team_id]
        if len(team_players) == 0:
            # 如果没有选手数据，返回默认值
            return [{'adjusted_rating': 1.0, 'kd_ratio': 1.0, 'adr': 70.0, 
                    'age': 23.0, 'recent_form': 0.5}] * 5
        
        return team_players.to_dict('records')[:5]  # 最多5名选手
    
    def _generate_mock_data(self, num_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成模拟训练数据"""
        self.logger.info(f"生成 {num_samples} 个模拟训练样本")
        
        # 特征维度（需要与模型输入匹配）
        feature_dim = 50
        
        features = np.random.randn(num_samples, feature_dim).astype(np.float32)
        
        # 生成相关的标签
        # 基于特征的线性组合 + 噪声
        weights = np.random.randn(feature_dim)
        logits = np.dot(features, weights)
        probabilities = 1 / (1 + np.exp(-logits))  # sigmoid
        labels = (probabilities > 0.5).astype(int)
        
        # 生成地图标签
        map_labels = np.random.randint(0, len(CS2_MAP_POOL), num_samples)
        
        return features, labels, map_labels
    
    def create_model(self, input_dim: int) -> nn.Module:
        """创建模型"""
        if self.use_ensemble:
            model = EnsembleModel(input_dim, self.num_ensemble_models)
            self.logger.info(f"创建集成模型，包含 {self.num_ensemble_models} 个子模型")
        else:
            model = CS2MatchPredictor(input_dim)
            self.logger.info("创建单一预测模型")
        
        # 模型参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型参数总数: {total_params:,}")
        self.logger.info(f"可训练参数: {trainable_params:,}")
        
        return model
    
    def train(self, epochs: int = None) -> Dict:
        """训练模型"""
        epochs = epochs or MODEL_CONFIG['max_epochs']
        
        self.logger.info("开始训练模型...")
        
        # 加载数据
        train_loader, val_loader, test_loader = self.load_data()
        
        # 获取特征维度
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch['features'].shape[1]
        
        # 创建模型
        model = self.create_model(input_dim)
        trainer = ModelTrainer(model, self.device)
        
        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # 训练循环
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # 训练
            train_loss = trainer.train_epoch(train_loader)
            
            # 验证
            val_loss, val_accuracy = trainer.validate(val_loader)
            
            # 更新学习率
            trainer.scheduler.step(val_loss)
            current_lr = trainer.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['learning_rates'].append(current_lr)
            
            self.logger.info(f"训练损失: {train_loss:.4f}, "
                           f"验证损失: {val_loss:.4f}, "
                           f"验证准确率: {val_accuracy:.4f}, "
                           f"学习率: {current_lr:.6f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                
                # 保存最佳模型
                self.save_model(model, f"best_model_epoch_{epoch + 1}.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
                self.logger.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                break
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_model(model, f"checkpoint_epoch_{epoch + 1}.pth")
        
        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 最终测试
        test_loss, test_accuracy = trainer.validate(test_loader)
        self.logger.info(f"最终测试结果 - 损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}")
        
        # 保存最终模型
        self.save_model(model, "final_model.pth")
        
        # 保存训练历史
        self.save_training_history()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        return {
            'model': model,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'training_history': self.training_history
        }
    
    def save_model(self, model: nn.Module, filename: str):
        """保存模型"""
        model_path = os.path.join(MODEL_DIR, filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': MODEL_CONFIG,
            'training_config': TRAINING_CONFIG,
            'model_type': 'ensemble' if self.use_ensemble else 'single',
            'num_ensemble_models': self.num_ensemble_models if self.use_ensemble else 1,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        self.logger.info(f"模型已保存: {model_path}")
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(LOGS_DIR, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"训练历史已保存: {history_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(self.training_history['train_loss'], label='训练损失')
        plt.plot(self.training_history['val_loss'], label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.title('训练和验证损失')
        
        # 准确率曲线
        plt.subplot(1, 3, 2)
        plt.plot(self.training_history['val_accuracy'], label='验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
        plt.title('验证准确率')
        
        # 学习率曲线
        plt.subplot(1, 3, 3)
        plt.plot(self.training_history['learning_rates'], label='学习率')
        plt.xlabel('Epoch')
        plt.ylabel('学习率')
        plt.legend()
        plt.title('学习率变化')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(LOGS_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("训练曲线已保存")

if __name__ == "__main__":
    # 示例训练
    trainer = CS2ModelTrainer(use_ensemble=False)
    
    # 开始训练
    results = trainer.train(epochs=50)
    
    print(f"训练完成！最终测试准确率: {results['test_accuracy']:.4f}")