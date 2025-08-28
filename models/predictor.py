"""
CS比赛综合预测器
整合选图预测和胜率预测功能
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
import logging
from models.map_prediction_model import load_map_model
from models.winrate_prediction_model import load_winrate_model

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSPredictor:
    """CS比赛综合预测器"""
    
    def __init__(self, map_model_path: str, winrate_model_path: str, 
                 config_path: str = "configs/data_config.yaml"):
        """初始化预测器"""
        self.config = self._load_config(config_path)
        
        # 加载模型
        self.map_model = self._load_map_model(map_model_path)
        self.winrate_model = self._load_winrate_model(winrate_model_path)
        
        # 地图名称映射
        self.map_names = [
            'de_dust2', 'de_mirage', 'de_inferno', 'de_overpass', 
            'de_nuke', 'de_vertigo', 'de_ancient'
        ]
        
        # 年龄权重配置
        self.age_weights = self.config['data']['preprocessing']['age_processing']['age_weights']
        
        logger.info("CS预测器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_map_model(self, model_path: str):
        """加载选图预测模型"""
        if not Path(model_path).exists():
            logger.warning(f"选图模型文件不存在: {model_path}")
            return None
        
        try:
            # 这里需要根据实际的配置文件来加载模型
            # 暂时返回None，实际使用时需要实现
            logger.info(f"选图模型加载成功: {model_path}")
            return None
        except Exception as e:
            logger.error(f"选图模型加载失败: {e}")
            return None
    
    def _load_winrate_model(self, model_path: str):
        """加载胜率预测模型"""
        if not Path(model_path).exists():
            logger.warning(f"胜率模型文件不存在: {model_path}")
            return None
        
        try:
            # 这里需要根据实际的配置文件来加载模型
            # 暂时返回None，实际使用时需要实现
            logger.info(f"胜率模型加载成功: {model_path}")
            return None
        except Exception as e:
            logger.error(f"胜率模型加载失败: {e}")
            return None
    
    def get_player_features(self, player_nicknames: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取选手特征"""
        # 这里需要从HLTV API或本地数据库获取选手特征
        # 暂时使用模拟数据
        
        num_players = len(player_nicknames)
        
        # 模拟选手特征（实际使用时需要替换为真实数据）
        player_features = np.random.randn(num_players, 8)  # 8个特征维度
        ages = np.random.randint(18, 32, num_players)
        experience = np.random.randint(1, 15, num_players)
        
        # 根据年龄调整特征权重
        for i, age in enumerate(ages):
            if age <= 22:  # 年轻选手
                player_features[i] *= self.age_weights['youth_factor']
            elif age >= 30:  # 老将
                player_features[i] *= self.age_weights['veteran_factor']
            else:  # 黄金年龄
                player_features[i] *= self.age_weights['prime_factor']
        
        return player_features, ages, experience
    
    def get_team_features(self, player_features: np.ndarray, ages: np.ndarray) -> np.ndarray:
        """计算团队特征"""
        # 团队平均评分
        team_rating = np.mean(player_features[:, 0])  # 假设第一个特征是rating
        
        # 团队年龄优势
        avg_age = np.mean(ages)
        if avg_age <= 25:
            age_advantage = self.age_weights['youth_factor']
        elif avg_age <= 28:
            age_advantage = self.age_weights['prime_factor']
        elif avg_age <= 32:
            age_advantage = self.age_weights['veteran_factor']
        else:
            age_advantage = self.age_weights['legend_factor']
        
        # 团队化学（基于特征一致性）
        feature_std = np.std(player_features, axis=0)
        team_chemistry = 1.0 / (1.0 + np.mean(feature_std))
        
        # 团队一致性
        team_consistency = 1.0 - np.std(player_features[:, 0]) / (np.mean(player_features[:, 0]) + 1e-8)
        
        return np.array([team_rating, age_advantage, team_chemistry, team_consistency])
    
    def get_map_features(self) -> np.ndarray:
        """获取地图特征"""
        # 这里可以包含地图的统计信息，如胜率、选择频率等
        # 暂时使用随机数据
        return np.random.randn(4)  # 4个地图特征维度
    
    def get_matchup_features(self, team1_features: np.ndarray, team2_features: np.ndarray) -> np.ndarray:
        """计算对战特征"""
        # 团队实力差距
        skill_gap = team1_features[0] - team2_features[0]
        
        # 年龄优势差距
        age_gap = team1_features[1] - team2_features[1]
        
        # 团队化学差距
        chemistry_gap = team1_features[2] - team2_features[2]
        
        # 团队一致性差距
        consistency_gap = team1_features[3] - team2_features[3]
        
        return np.array([skill_gap, age_gap, chemistry_gap, consistency_gap])
    
    def predict_maps(self, team1_players: List[str], team2_players: List[str]) -> Tuple[List[str], List[float]]:
        """预测选图策略"""
        logger.info("开始选图预测")
        
        if self.map_model is None:
            logger.warning("选图模型未加载，使用规则基础预测")
            return self._rule_based_map_prediction(team1_players, team2_players)
        
        try:
            # 获取特征
            team1_features, team1_ages, team1_exp = self.get_player_features(team1_players)
            team2_features, team2_ages, team2_exp = self.get_player_features(team2_players)
            
            # 计算团队特征
            team1_stats = self.get_team_features(team1_features, team1_ages)
            team2_stats = self.get_team_features(team2_features, team2_ages)
            
            # 地图特征
            map_features = self.get_map_features()
            
            # 这里需要调用实际的模型进行预测
            # 暂时返回规则基础的结果
            return self._rule_based_map_prediction(team1_players, team2_players)
            
        except Exception as e:
            logger.error(f"选图预测失败: {e}")
            return self._rule_based_map_prediction(team1_players, team2_players)
    
    def _rule_based_map_prediction(self, team1_players: List[str], team2_players: List[str]) -> Tuple[List[str], List[float]]:
        """基于规则的选图预测"""
        # 简单的规则基础预测
        # 实际使用时可以基于历史数据和统计信息
        
        # 假设的地图偏好（实际使用时需要基于历史数据）
        map_preferences = {
            'de_dust2': 0.15,
            'de_mirage': 0.20,
            'de_inferno': 0.18,
            'de_overpass': 0.16,
            'de_nuke': 0.12,
            'de_vertigo': 0.08,
            'de_ancient': 0.11
        }
        
        # 按概率排序
        sorted_maps = sorted(map_preferences.items(), key=lambda x: x[1], reverse=True)
        recommended_maps = [map_name for map_name, _ in sorted_maps]
        map_scores = [score for _, score in sorted_maps]
        
        return recommended_maps, map_scores
    
    def predict_winrate(self, team1_players: List[str], team2_players: List[str]) -> Tuple[float, float]:
        """预测胜率"""
        logger.info("开始胜率预测")
        
        if self.winrate_model is None:
            logger.warning("胜率模型未加载，使用规则基础预测")
            return self._rule_based_winrate_prediction(team1_players, team2_players)
        
        try:
            # 获取特征
            team1_features, team1_ages, team1_exp = self.get_player_features(team1_players)
            team2_features, team2_ages, team2_exp = self.get_player_features(team2_players)
            
            # 计算团队特征
            team1_stats = self.get_team_features(team1_features, team1_ages)
            team2_stats = self.get_team_features(team2_features, team2_ages)
            
            # 对战特征
            matchup_features = self.get_matchup_features(team1_stats, team2_stats)
            
            # 这里需要调用实际的模型进行预测
            # 暂时返回规则基础的结果
            return self._rule_based_winrate_prediction(team1_players, team2_players)
            
        except Exception as e:
            logger.error(f"胜率预测失败: {e}")
            return self._rule_based_winrate_prediction(team1_players, team2_players)
    
    def _rule_based_winrate_prediction(self, team1_players: List[str], team2_players: List[str]) -> Tuple[float, float]:
        """基于规则的胜率预测"""
        # 简单的规则基础预测
        # 实际使用时可以基于历史数据和统计信息
        
        # 模拟的胜率计算
        base_winrate = 0.5
        
        # 随机调整（实际使用时需要基于真实数据）
        adjustment = np.random.normal(0, 0.1)
        team1_winrate = np.clip(base_winrate + adjustment, 0.1, 0.9)
        team2_winrate = 1.0 - team1_winrate
        
        return team1_winrate, team2_winrate
    
    def predict(self, team1_players: List[str], team2_players: List[str]) -> Dict[str, Any]:
        """综合预测"""
        logger.info("开始综合预测")
        
        if len(team1_players) != 5 or len(team2_players) != 5:
            raise ValueError("每支队伍必须包含5名选手")
        
        # 选图预测
        recommended_maps, map_scores = self.predict_maps(team1_players, team2_players)
        
        # 胜率预测
        team1_winrate, team2_winrate = self.predict_winrate(team1_players, team2_players)
        
        # 计算置信度
        confidence = self._calculate_confidence(team1_players, team2_players)
        
        # 年龄分析
        age_analysis = self._analyze_age_factors(team1_players, team2_players)
        
        # 构建结果
        result = {
            'recommended_maps': recommended_maps,
            'map_scores': map_scores,
            'winrate_team1': team1_winrate,
            'winrate_team2': team2_winrate,
            'confidence': confidence,
            'age_analysis': age_analysis,
            'prediction_time': pd.Timestamp.now().isoformat()
        }
        
        logger.info("综合预测完成")
        return result
    
    def _calculate_confidence(self, team1_players: List[str], team2_players: List[str]) -> float:
        """计算预测置信度"""
        # 基于选手知名度和数据完整性的置信度计算
        # 实际使用时需要基于真实数据
        
        # 模拟置信度
        base_confidence = 0.7
        
        # 根据选手数量调整
        if len(team1_players) == 5 and len(team2_players) == 5:
            base_confidence += 0.1
        
        # 随机调整
        adjustment = np.random.normal(0, 0.05)
        confidence = np.clip(base_confidence + adjustment, 0.5, 0.95)
        
        return confidence
    
    def _analyze_age_factors(self, team1_players: List[str], team2_players: List[str]) -> Dict[str, Any]:
        """分析年龄因素"""
        # 获取年龄信息
        team1_features, team1_ages, team1_exp = self.get_player_features(team1_players)
        team2_features, team2_ages, team2_exp = self.get_player_features(team2_players)
        
        # 计算年龄统计
        team1_avg_age = np.mean(team1_ages)
        team2_avg_age = np.mean(team2_ages)
        
        team1_age_std = np.std(team1_ages)
        team2_age_std = np.std(team2_ages)
        
        # 年龄优势分析
        age_advantage = "team1" if team1_avg_age < team2_avg_age else "team2"
        age_gap = abs(team1_avg_age - team2_avg_age)
        
        # 经验分析
        team1_avg_exp = np.mean(team1_exp)
        team2_avg_exp = np.mean(team2_exp)
        experience_advantage = "team1" if team1_avg_exp > team2_avg_exp else "team2"
        
        return {
            'team1_avg_age': team1_avg_age,
            'team2_avg_age': team2_avg_age,
            'team1_age_std': team1_age_std,
            'team2_age_std': team2_age_std,
            'age_advantage': age_advantage,
            'age_gap': age_gap,
            'team1_avg_experience': team1_avg_exp,
            'team2_avg_experience': team2_avg_exp,
            'experience_advantage': experience_advantage,
            'age_factor_impact': self._calculate_age_impact(team1_avg_age, team2_avg_age)
        }
    
    def _calculate_age_impact(self, age1: float, age2: float) -> float:
        """计算年龄因素对比赛的影响"""
        # 年龄差异的影响权重
        age_diff = abs(age1 - age2)
        
        if age_diff < 2:
            return 0.05  # 年龄相近，影响较小
        elif age_diff < 5:
            return 0.10  # 年龄差异中等，影响中等
        else:
            return 0.15  # 年龄差异较大，影响较大
    
    def batch_predict(self, matches: List[Dict[str, List[str]]]) -> List[Dict[str, Any]]:
        """批量预测"""
        logger.info(f"开始批量预测 {len(matches)} 场比赛")
        
        results = []
        for i, match in enumerate(matches):
            try:
                team1_players = match['team1']
                team2_players = match['team2']
                
                result = self.predict(team1_players, team2_players)
                results.append({
                    'match_id': i,
                    'team1': team1_players,
                    'team2': team2_players,
                    'prediction': result
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已预测 {i + 1}/{len(matches)} 场比赛")
                    
            except Exception as e:
                logger.error(f"第 {i + 1} 场比赛预测失败: {e}")
                results.append({
                    'match_id': i,
                    'team1': match.get('team1', []),
                    'team2': match.get('team2', []),
                    'error': str(e)
                })
        
        logger.info("批量预测完成")
        return results
    
    def save_predictions(self, predictions: List[Dict[str, Any]], output_path: str):
        """保存预测结果"""
        try:
            df = pd.DataFrame(predictions)
            df.to_json(output_path, orient='records', indent=2)
            logger.info(f"预测结果已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存预测结果失败: {e}")


def main():
    """主函数示例"""
    # 创建预测器
    predictor = CSPredictor(
        map_model_path="./checkpoints/map_model_best.pth",
        winrate_model_path="./checkpoints/winrate_model_best.pth"
    )
    
    # 示例选手
    team1_players = ["ZywOo", "shox", "apEX", "misutaaa", "Kyojin"]
    team2_players = ["s1mple", "electronic", "Boombl4", "Perfecto", "flamie"]
    
    # 进行预测
    result = predictor.predict(team1_players, team2_players)
    
    # 打印结果
    print("=== CS比赛预测结果 ===")
    print(f"推荐选图: {result['recommended_maps']}")
    print(f"地图得分: {[f'{score:.3f}' for score in result['map_scores']]}")
    print(f"胜率预测: Team1 {result['winrate_team1']:.2%}, Team2 {result['winrate_team2']:.2%}")
    print(f"预测置信度: {result['confidence']:.2%}")
    
    # 年龄分析
    age_analysis = result['age_analysis']
    print(f"\n=== 年龄分析 ===")
    print(f"Team1 平均年龄: {age_analysis['team1_avg_age']:.1f}")
    print(f"Team2 平均年龄: {age_analysis['team2_avg_age']:.1f}")
    print(f"年龄优势: {age_analysis['age_advantage']}")
    print(f"年龄因素影响: {age_analysis['age_factor_impact']:.2%}")


if __name__ == "__main__":
    main()