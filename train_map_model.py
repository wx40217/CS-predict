#!/usr/bin/env python3
"""
选图预测模型训练脚本
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 导入模型
from models.map_prediction_model import create_map_model, MapPredictionLoss

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MapModelTrainer:
    """选图预测模型训练器"""
    
    def __init__(self, config_path: str, gpu_id: int = 0):
        """初始化训练器"""
        self.config = self._load_config(config_path)
        self.device = self._setup_device(gpu_id)
        
        # 创建模型
        self.model = create_map_model(self.config).to(self.device)
        
        # 创建损失函数
        self.criterion = MapPredictionLoss(self.config).to(self.device)
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练
        self.scaler = GradScaler() if self.config['hardware']['mixed_precision'] else None
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # 创建检查点目录
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志目录
        self.log_dir = Path(self.config['logging']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"训练器初始化完成，设备: {self.device}")
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_device(self, gpu_id: int) -> torch.device:
        """设置设备"""
        if torch.cuda.is_available() and self.config['hardware']['device'] == 'cuda':
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(device)
            logger.info(f"使用GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device('cpu')
            logger.info("使用CPU")
        
        return device
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_config = self.config['training']['optimizer']
        
        if optimizer_config['type'] == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                betas=optimizer_config['betas'],
                eps=optimizer_config['eps']
            )
        else:
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建学习率调度器"""
        scheduler_config = self.config['training']['scheduler']
        
        if scheduler_config['type'] == 'CosineAnnealingWarmRestarts':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config['T_0'],
                T_mult=scheduler_config['T_mult']
            )
        elif scheduler_config['type'] == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config['mode'],
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                min_lr=scheduler_config['min_lr']
            )
        else:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """加载训练数据"""
        logger.info("开始加载训练数据")
        
        # 加载处理后的数据
        data_path = Path(self.config['data']['storage']['processed_data_dir'])
        map_dataset_path = data_path / "map_prediction_dataset.parquet"
        
        if not map_dataset_path.exists():
            raise FileNotFoundError(f"训练数据不存在: {map_dataset_path}")
        
        # 加载数据
        dataset = pd.read_parquet(map_dataset_path)
        logger.info(f"加载数据集: {len(dataset)} 样本")
        
        # 数据分割
        train_data, temp_data = train_test_split(
            dataset, 
            test_size=1 - self.config['training']['train_split'],
            random_state=42
        )
        
        val_size = self.config['training']['val_split'] / (self.config['training']['val_split'] + self.config['training']['test_split'])
        val_data, test_data = train_test_split(
            temp_data,
            test_size=1 - val_size,
            random_state=42
        )
        
        logger.info(f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")
        
        # 创建数据加载器
        train_loader = self._create_data_loader(train_data, is_training=True)
        val_loader = self._create_data_loader(val_data, is_training=False)
        test_loader = self._create_data_loader(test_data, is_training=False)
        
        return train_loader, val_loader, test_loader
    
    def _create_data_loader(self, data: pd.DataFrame, is_training: bool) -> DataLoader:
        """创建数据加载器"""
        # 这里需要根据实际的数据结构来创建张量
        # 暂时使用随机数据进行演示
        
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['hardware']['num_workers']
        pin_memory = self.config['hardware']['pin_memory']
        
        # 创建随机数据（实际使用时需要替换为真实数据）
        num_samples = len(data)
        feature_dim = self.config['model']['architecture']['input_dim']
        max_seq_length = self.config['model']['architecture']['max_seq_length']
        
        # 生成随机特征
        player_features = torch.randn(num_samples, max_seq_length, feature_dim)
        team_features = torch.randn(num_samples, len(self.config['model']['features']['team_stats']))
        map_features = torch.randn(num_samples, len(self.config['model']['features']['map_stats']))
        ages = torch.randint(16, 35, (num_samples, max_seq_length))
        experience = torch.randint(0, 20, (num_samples, max_seq_length))
        
        # 生成随机标签
        map_labels = torch.randint(0, self.config['model']['maps']['num_maps'], (num_samples,))
        strategy_labels = torch.randint(0, 3, (num_samples,))
        
        # 创建数据集
        dataset = TensorDataset(
            player_features, team_features, map_features, ages, experience,
            map_labels, strategy_labels
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (player_features, team_features, map_features, ages, experience, 
                        map_labels, strategy_labels) in enumerate(progress_bar):
            
            # 移动数据到设备
            player_features = player_features.to(self.device)
            team_features = team_features.to(self.device)
            map_features = map_features.to(self.device)
            ages = ages.to(self.device)
            experience = experience.to(self.device)
            map_labels = map_labels.to(self.device)
            strategy_labels = strategy_labels.to(self.device)
            
            # 前向传播
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(player_features, team_features, map_features, ages, experience)
                    loss = self.criterion(outputs, {
                        'map_labels': map_labels,
                        'strategy_labels': strategy_labels
                    })
            else:
                outputs = self.model(player_features, team_features, map_features, ages, experience)
                loss = self.criterion(outputs, {
                    'map_labels': map_labels,
                    'strategy_labels': strategy_labels
                })
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            
            # 计算准确率
            map_preds = torch.argmax(outputs['map_probabilities'], dim=1)
            total_correct += (map_preds == map_labels).sum().item()
            total_samples += map_labels.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{total_correct/total_samples:.4f}"
            })
            
            # 记录日志
            if batch_idx % self.config['logging']['log_freq'] == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for player_features, team_features, map_features, ages, experience, \
                map_labels, strategy_labels in val_loader:
                
                # 移动数据到设备
                player_features = player_features.to(self.device)
                team_features = team_features.to(self.device)
                map_features = map_features.to(self.device)
                ages = ages.to(self.device)
                experience = experience.to(self.device)
                map_labels = map_labels.to(self.device)
                strategy_labels = strategy_labels.to(self.device)
                
                # 前向传播
                outputs = self.model(player_features, team_features, map_features, ages, experience)
                loss = self.criterion(outputs, {
                    'map_labels': map_labels,
                    'strategy_labels': strategy_labels
                })
                
                # 统计
                total_loss += loss.item()
                
                # 计算准确率
                map_preds = torch.argmax(outputs['map_probabilities'], dim=1)
                total_correct += (map_preds == map_labels).sum().item()
                total_samples += map_labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_history': self.train_history
        }
        
        # 保存最新检查点
        checkpoint_path = self.checkpoint_dir / f"map_model_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.checkpoint_dir / "map_model_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点"""
        checkpoints = sorted(self.checkpoint_dir.glob("map_model_epoch_*.pth"))
        keep_last = self.config['checkpoint']['keep_last']
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
                logger.info(f"删除旧检查点: {checkpoint}")
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # 损失曲线
        ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(epochs, self.train_history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.train_history['val_acc'], 'r-', label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # 学习率曲线
        ax3.plot(epochs, [self.optimizer.param_groups[0]['lr']] * len(epochs), 'g-')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        
        # 损失分布
        ax4.hist(self.train_history['train_loss'], bins=20, alpha=0.7, label='Train Loss')
        ax4.hist(self.train_history['val_loss'], bins=20, alpha=0.7, label='Val Loss')
        ax4.set_title('Loss Distribution')
        ax4.set_xlabel('Loss')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练历史图表已保存: {self.log_dir / 'training_history.png'}")
    
    def train(self):
        """开始训练"""
        logger.info("开始训练选图预测模型")
        
        # 加载数据
        train_loader, val_loader, test_loader = self.load_data()
        
        # 训练参数
        num_epochs = self.config['training']['num_epochs']
        early_stopping_patience = self.config['training']['early_stopping']['patience']
        min_delta = self.config['training']['early_stopping']['min_delta']
        
        # 早停相关
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 训练循环
        for epoch in range(num_epochs):
            logger.info(f"开始训练 Epoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 更新学习率
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_acc'].append(val_acc)
            
            # 打印结果
            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 检查是否是最佳模型
            is_best = val_loss < best_val_loss - min_delta
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"发现更好的模型，验证损失: {val_loss:.4f}")
            else:
                patience_counter += 1
            
            # 保存检查点
            if (epoch + 1) % self.config['checkpoint']['save_freq'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # 早停检查
            if patience_counter >= early_stopping_patience:
                logger.info(f"早停触发，{early_stopping_patience} 个epoch没有改善")
                break
        
        # 训练完成
        logger.info("训练完成")
        
        # 绘制训练历史
        self.plot_training_history()
        
        # 最终验证
        final_val_loss, final_val_acc = self.validate(val_loader)
        logger.info(f"最终验证结果 - Loss: {final_val_loss:.4f}, Acc: {final_val_acc:.4f}")
        
        # 测试集评估
        test_loss, test_acc = self.validate(test_loader)
        logger.info(f"测试集结果 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练选图预测模型')
    parser.add_argument('--config', type=str, default='configs/map_model.yaml',
                       help='配置文件路径')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = MapModelTrainer(args.config, args.gpu)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()