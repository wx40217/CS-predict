"""
模型推理预测器 - 用于实际比赛预测
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import os
import joblib
from dataclasses import dataclass

from model import CS2MatchPredictor, EnsembleModel
from data_preprocessor import DataPreprocessor
from config import MODEL_DIR, PROCESSED_DATA_DIR, CS2_MAP_POOL

@dataclass
class PlayerInfo:
    """选手信息"""
    name: str
    age: int
    rating_2_0: float = 1.0
    kd_ratio: float = 1.0
    adr: float = 70.0
    kpr: float = 0.7
    headshot_pct: float = 50.0
    experience_months: int = 12
    maps_played: int = 100
    rounds_played: int = 2000

@dataclass
class TeamInfo:
    """队伍信息"""
    name: str
    players: List[PlayerInfo]
    world_ranking: int = 50
    recent_winrate: float = 0.5
    map_winrates: Dict[str, float] = None
    
    def __post_init__(self):
        if self.map_winrates is None:
            self.map_winrates = {map_name: 0.5 for map_name in CS2_MAP_POOL}

@dataclass
class MatchPrediction:
    """比赛预测结果"""
    team1_name: str
    team2_name: str
    map_predictions: Dict[str, float]  # 每张地图team1的胜率
    recommended_maps_team1: List[str]  # team1推荐选择的地图
    recommended_maps_team2: List[str]  # team2推荐选择的地图
    overall_strength_comparison: Dict[str, float]  # 整体实力对比

class CS2MatchPredictor_Inference:
    """CS2比赛预测器（推理版本）"""
    
    def __init__(self, model_path: str = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径，如果为None则使用最新的模型
        """
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 设备配置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 加载数据预处理器
        self.preprocessor = self._load_preprocessor()
        
        self.logger.info("预测器初始化完成")
    
    def _load_model(self, model_path: str = None) -> nn.Module:
        """加载训练好的模型"""
        if model_path is None:
            # 查找最新的模型文件
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
            if not model_files:
                raise FileNotFoundError(f"在 {MODEL_DIR} 中未找到模型文件")
            
            # 使用最新的模型（按修改时间排序）
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)
            model_path = os.path.join(MODEL_DIR, model_files[0])
        
        self.logger.info(f"加载模型: {model_path}")
        
        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 确定模型类型和创建模型
        model_type = checkpoint.get('model_type', 'single')
        
        if model_type == 'ensemble':
            num_models = checkpoint.get('num_ensemble_models', 3)
            # 需要知道输入维度，这里从状态字典推断
            sample_key = 'models.0.team1_encoder.encoder.0.weight'
            if sample_key in checkpoint['model_state_dict']:
                input_dim = checkpoint['model_state_dict'][sample_key].shape[1] * 2 + 2
            else:
                input_dim = 50  # 默认值
            model = EnsembleModel(input_dim, num_models)
        else:
            # 从状态字典推断输入维度
            sample_key = 'team1_encoder.encoder.0.weight'
            if sample_key in checkpoint['model_state_dict']:
                input_dim = checkpoint['model_state_dict'][sample_key].shape[1] * 2 + 2
            else:
                input_dim = 50  # 默认值
            model = CS2MatchPredictor(input_dim)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        self.logger.info(f"模型加载完成，类型: {model_type}")
        
        return model
    
    def _load_preprocessor(self) -> DataPreprocessor:
        """加载数据预处理器"""
        preprocessor = DataPreprocessor()
        
        # 加载预处理器状态
        try:
            preprocessor.player_scaler = joblib.load(
                os.path.join(PROCESSED_DATA_DIR, 'player_scaler.pkl')
            )
            preprocessor.team_scaler = joblib.load(
                os.path.join(PROCESSED_DATA_DIR, 'team_scaler.pkl')
            )
            preprocessor.imputers = joblib.load(
                os.path.join(PROCESSED_DATA_DIR, 'imputers.pkl')
            )
            self.logger.info("预处理器状态加载完成")
        except FileNotFoundError:
            self.logger.warning("未找到预处理器状态文件，使用默认配置")
        
        return preprocessor
    
    def _convert_player_to_dict(self, player: PlayerInfo) -> Dict:
        """将PlayerInfo转换为字典格式"""
        return {
            'name': player.name,
            'age': player.age,
            'rating_2_0': player.rating_2_0,
            'kd_ratio': player.kd_ratio,
            'adr': player.adr,
            'kpr': player.kpr,
            'headshot_pct': player.headshot_pct,
            'experience_months': player.experience_months,
            'maps_played': player.maps_played,
            'rounds_played': player.rounds_played,
            # 计算衍生特征
            'age_impact': self.preprocessor.calculate_age_impact(player.age),
            'adjusted_rating': player.rating_2_0 * self.preprocessor.calculate_age_impact(player.age),
            'recent_form': self.preprocessor.calculate_recent_form({
                'rating_2_0': player.rating_2_0
            }),
            'survival_rate': 1.0 - (1.0 / max(player.kd_ratio, 0.1))
        }
    
    def _convert_team_to_dict(self, team: TeamInfo) -> Dict:
        """将TeamInfo转换为字典格式"""
        # 计算队伍统计
        player_ratings = [p.rating_2_0 for p in team.players]
        
        team_dict = {
            'name': team.name,
            'world_ranking': team.world_ranking,
            'recent_winrate': team.recent_winrate,
            'ranking_score': 1.0 / max(team.world_ranking, 1),
            'avg_map_winrate': np.mean(list(team.map_winrates.values())),
            'map_advantage_variance': np.var(list(team.map_winrates.values())),
        }
        
        # 添加各地图胜率
        for map_name in CS2_MAP_POOL:
            team_dict[f'map_winrate_{map_name}'] = team.map_winrates.get(map_name, 0.5)
        
        return team_dict
    
    def predict_match(self, team1: TeamInfo, team2: TeamInfo) -> MatchPrediction:
        """
        预测比赛结果
        
        Args:
            team1: 队伍1信息
            team2: 队伍2信息
            
        Returns:
            比赛预测结果
        """
        self.logger.info(f"开始预测比赛: {team1.name} vs {team2.name}")
        
        # 转换为字典格式
        team1_dict = self._convert_team_to_dict(team1)
        team2_dict = self._convert_team_to_dict(team2)
        
        team1_players = [self._convert_player_to_dict(p) for p in team1.players]
        team2_players = [self._convert_player_to_dict(p) for p in team2.players]
        
        # 预测每张地图的胜率
        map_predictions = {}
        
        with torch.no_grad():
            for map_name in CS2_MAP_POOL:
                # 为指定地图创建特征向量
                features = self.preprocessor.create_match_features(
                    team1_dict, team2_dict, team1_players, team2_players, map_name
                )
                
                # 转换为张量
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                
                # 获取地图索引
                map_idx = CS2_MAP_POOL.index(map_name)
                map_indices = torch.LongTensor([map_idx]).to(self.device)
                
                # 预测
                win_prob = self.model(features_tensor, map_indices).squeeze().item()
                map_predictions[map_name] = win_prob
        
        # 分析推荐地图
        recommended_maps_team1, recommended_maps_team2 = self._analyze_map_recommendations(
            map_predictions
        )
        
        # 计算整体实力对比
        strength_comparison = self._calculate_strength_comparison(
            team1_dict, team2_dict, team1_players, team2_players
        )
        
        prediction = MatchPrediction(
            team1_name=team1.name,
            team2_name=team2.name,
            map_predictions=map_predictions,
            recommended_maps_team1=recommended_maps_team1,
            recommended_maps_team2=recommended_maps_team2,
            overall_strength_comparison=strength_comparison
        )
        
        self.logger.info("比赛预测完成")
        
        return prediction
    
    def _analyze_map_recommendations(self, map_predictions: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """分析地图推荐"""
        # 按胜率排序
        sorted_maps = sorted(map_predictions.items(), key=lambda x: x[1], reverse=True)
        
        # team1优势地图（胜率高）
        team1_maps = [map_name for map_name, win_rate in sorted_maps[:3] if win_rate > 0.5]
        
        # team2优势地图（team1胜率低）
        team2_maps = [map_name for map_name, win_rate in sorted_maps[-3:] if win_rate < 0.5]
        team2_maps.reverse()  # 按team2优势程度排序
        
        return team1_maps, team2_maps
    
    def _calculate_strength_comparison(self, team1_dict: Dict, team2_dict: Dict,
                                     team1_players: List[Dict], team2_players: List[Dict]) -> Dict[str, float]:
        """计算整体实力对比"""
        
        # 队伍层面对比
        ranking_advantage = team1_dict['ranking_score'] / (team1_dict['ranking_score'] + team2_dict['ranking_score'])
        recent_form_advantage = team1_dict['recent_winrate'] / (team1_dict['recent_winrate'] + team2_dict['recent_winrate'])
        
        # 选手层面对比
        team1_avg_rating = np.mean([p['adjusted_rating'] for p in team1_players])
        team2_avg_rating = np.mean([p['adjusted_rating'] for p in team2_players])
        player_advantage = team1_avg_rating / (team1_avg_rating + team2_avg_rating)
        
        # 经验对比
        team1_avg_exp = np.mean([p['experience_months'] for p in team1_players])
        team2_avg_exp = np.mean([p['experience_months'] for p in team2_players])
        experience_advantage = team1_avg_exp / (team1_avg_exp + team2_avg_exp)
        
        return {
            'ranking_advantage': ranking_advantage,
            'recent_form_advantage': recent_form_advantage,
            'player_skill_advantage': player_advantage,
            'experience_advantage': experience_advantage,
            'overall_advantage': (ranking_advantage + recent_form_advantage + player_advantage + experience_advantage) / 4
        }
    
    def print_prediction_report(self, prediction: MatchPrediction):
        """打印预测报告"""
        print(f"\n{'='*60}")
        print(f"CS2比赛预测报告")
        print(f"{'='*60}")
        print(f"{prediction.team1_name} vs {prediction.team2_name}")
        print(f"{'='*60}")
        
        print(f"\n📊 各地图胜率预测 ({prediction.team1_name}获胜概率):")
        print(f"{'-'*40}")
        for map_name, win_rate in sorted(prediction.map_predictions.items(), key=lambda x: x[1], reverse=True):
            win_pct = win_rate * 100
            bar_length = int(win_rate * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f"{map_name:10} │ {bar} │ {win_pct:5.1f}%")
        
        print(f"\n🗺️  推荐地图选择:")
        print(f"{'-'*30}")
        print(f"{prediction.team1_name}优势地图: {', '.join(prediction.recommended_maps_team1)}")
        print(f"{prediction.team2_name}优势地图: {', '.join(prediction.recommended_maps_team2)}")
        
        print(f"\n⚔️  整体实力对比 ({prediction.team1_name}优势程度):")
        print(f"{'-'*40}")
        comparison = prediction.overall_strength_comparison
        
        for key, value in comparison.items():
            pct = value * 100
            advantage_level = "强" if pct > 60 else "中" if pct > 40 else "弱"
            print(f"{key:20}: {pct:5.1f}% ({advantage_level})")
        
        print(f"\n🎯 预测总结:")
        print(f"{'-'*20}")
        overall_adv = comparison['overall_advantage']
        if overall_adv > 0.6:
            print(f"{prediction.team1_name} 明显优势")
        elif overall_adv > 0.4:
            print(f"双方实力接近，{prediction.team1_name} 略有优势")
        elif overall_adv < 0.4:
            print(f"{prediction.team2_name} 明显优势")
        else:
            print("双方实力非常接近")

# 示例使用函数
def create_sample_teams() -> Tuple[TeamInfo, TeamInfo]:
    """创建示例队伍用于测试"""
    
    # 队伍1选手
    team1_players = [
        PlayerInfo("s1mple", 26, 1.35, 1.28, 85.2, 0.85, 52.1, 60),
        PlayerInfo("ZywOo", 23, 1.32, 1.25, 83.8, 0.82, 48.9, 48),
        PlayerInfo("sh1ro", 24, 1.28, 1.22, 81.5, 0.79, 49.2, 36),
        PlayerInfo("NiKo", 27, 1.25, 1.18, 79.3, 0.76, 51.8, 84),
        PlayerInfo("device", 29, 1.22, 1.15, 77.8, 0.74, 53.2, 96)
    ]
    
    # 队伍2选手
    team2_players = [
        PlayerInfo("Ax1Le", 22, 1.18, 1.12, 75.6, 0.72, 47.3, 24),
        PlayerInfo("electroNic", 26, 1.15, 1.08, 73.4, 0.69, 45.8, 72),
        PlayerInfo("Perfecto", 25, 1.05, 0.98, 68.9, 0.65, 44.1, 48),
        PlayerInfo("b1t", 21, 1.12, 1.05, 71.2, 0.67, 46.7, 18),
        PlayerInfo("Boombl4", 27, 0.95, 0.89, 65.3, 0.58, 42.5, 60)
    ]
    
    team1 = TeamInfo(
        name="Team Liquid",
        players=team1_players,
        world_ranking=3,
        recent_winrate=0.72,
        map_winrates={
            "mirage": 0.75, "inferno": 0.68, "dust2": 0.71,
            "nuke": 0.58, "overpass": 0.65, "vertigo": 0.62, "ancient": 0.69
        }
    )
    
    team2 = TeamInfo(
        name="NAVI",
        players=team2_players,
        world_ranking=8,
        recent_winrate=0.58,
        map_winrates={
            "mirage": 0.62, "inferno": 0.71, "dust2": 0.55,
            "nuke": 0.68, "overpass": 0.59, "vertigo": 0.64, "ancient": 0.57
        }
    )
    
    return team1, team2

if __name__ == "__main__":
    # 示例使用
    try:
        # 创建预测器
        predictor = CS2MatchPredictor_Inference()
        
        # 创建示例队伍
        team1, team2 = create_sample_teams()
        
        # 进行预测
        prediction = predictor.predict_match(team1, team2)
        
        # 打印报告
        predictor.print_prediction_report(prediction)
        
    except Exception as e:
        print(f"预测失败: {e}")
        print("请确保已经训练并保存了模型")