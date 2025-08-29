"""
数据预处理模块 - 清洗和预处理HLTV数据
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, CS2_MAP_POOL,
    PLAYER_FEATURES, TEAM_FEATURES, AGE_IMPACT_CONFIG
)

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 初始化预处理器
        self.player_scaler = StandardScaler()
        self.team_scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
    
    def calculate_age_impact(self, age: float) -> float:
        """计算年龄对表现的影响系数"""
        if age <= 0:
            return 1.0
        
        peak_age = AGE_IMPACT_CONFIG['peak_age']
        decline_start = AGE_IMPACT_CONFIG['decline_start']
        
        if age <= peak_age:
            # 年轻选手，随年龄增长表现提升
            return 0.8 + 0.2 * (age / peak_age)
        elif age <= decline_start:
            # 巅峰期，保持高水平
            return 1.0
        else:
            # 开始衰退，但经验可以部分弥补
            decline_factor = max(0.3, 1.0 - 0.05 * (age - decline_start))
            experience_bonus = min(0.2, 0.02 * (age - decline_start))
            return decline_factor + experience_bonus
    
    def calculate_recent_form(self, player_data: Dict, lookback_days: int = 90) -> float:
        """计算选手近期状态"""
        # 这里需要根据实际数据结构调整
        # 简化版本：基于最近的评分变化
        rating = player_data.get('rating_2_0', 1.0)
        
        # 基于评分的简单状态计算
        if rating >= 1.2:
            return 1.0  # 优秀状态
        elif rating >= 1.0:
            return 0.8  # 良好状态
        elif rating >= 0.9:
            return 0.6  # 一般状态
        else:
            return 0.4  # 较差状态
    
    def calculate_team_chemistry(self, team_players: List[Dict]) -> float:
        """计算队伍默契度"""
        if len(team_players) < 5:
            return 0.5
        
        # 基于选手共同比赛时间计算默契度
        avg_experience = np.mean([p.get('experience_months', 0) for p in team_players])
        
        # 简化计算：经验越丰富，默契度越高
        chemistry = min(1.0, avg_experience / 24.0)  # 2年经验为满分
        return max(0.3, chemistry)
    
    def preprocess_player_data(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """预处理选手数据"""
        df = players_df.copy()
        
        self.logger.info("开始预处理选手数据")
        
        # 处理缺失值
        numeric_columns = ['rating_2_0', 'kd_ratio', 'adr', 'kpr', 'headshot_pct', 'age']
        for col in numeric_columns:
            if col in df.columns:
                # 使用中位数填充缺失值
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='median')
                    df[col] = self.imputers[col].fit_transform(df[[col]]).flatten()
                else:
                    df[col] = self.imputers[col].transform(df[[col]]).flatten()
        
        # 计算年龄影响系数
        df['age_impact'] = df['age'].apply(self.calculate_age_impact)
        
        # 调整评分基于年龄影响
        df['adjusted_rating'] = df['rating_2_0'] * df['age_impact']
        
        # 计算近期状态
        df['recent_form'] = df.apply(
            lambda row: self.calculate_recent_form(row.to_dict()), axis=1
        )
        
        # 计算每回合生存率
        df['survival_rate'] = 1.0 - (1.0 / np.maximum(df['kd_ratio'], 0.1))
        df['survival_rate'] = np.clip(df['survival_rate'], 0, 1)
        
        # 标准化数值特征
        feature_cols = [col for col in PLAYER_FEATURES if col in df.columns]
        feature_cols.extend(['age_impact', 'adjusted_rating', 'survival_rate'])
        
        df[feature_cols] = self.player_scaler.fit_transform(df[feature_cols])
        
        self.logger.info(f"选手数据预处理完成，处理了 {len(df)} 条记录")
        
        return df
    
    def preprocess_team_data(self, teams_df: pd.DataFrame) -> pd.DataFrame:
        """预处理队伍数据"""
        df = teams_df.copy()
        
        self.logger.info("开始预处理队伍数据")
        
        # 处理世界排名（排名越低数值越大，需要转换）
        df['ranking_score'] = 1.0 / np.maximum(df['world_ranking'], 1)
        
        # 计算平均地图胜率
        map_cols = [f'map_winrate_{map_name}' for map_name in CS2_MAP_POOL]
        existing_map_cols = [col for col in map_cols if col in df.columns]
        
        if existing_map_cols:
            df['avg_map_winrate'] = df[existing_map_cols].mean(axis=1)
        else:
            df['avg_map_winrate'] = 0.5
        
        # 计算地图优势度（某些地图胜率明显高于平均水平）
        if existing_map_cols:
            df['map_advantage_variance'] = df[existing_map_cols].var(axis=1)
        else:
            df['map_advantage_variance'] = 0.0
        
        # 标准化数值特征
        numeric_features = ['ranking_score', 'recent_winrate', 'avg_map_winrate', 
                           'map_advantage_variance'] + existing_map_cols
        
        df[numeric_features] = self.team_scaler.fit_transform(df[numeric_features])
        
        self.logger.info(f"队伍数据预处理完成，处理了 {len(df)} 条记录")
        
        return df
    
    def create_match_features(self, team1_data: Dict, team2_data: Dict, 
                            team1_players: List[Dict], team2_players: List[Dict],
                            map_name: str = None) -> np.ndarray:
        """为单场比赛创建特征向量"""
        
        features = []
        
        # 队伍1特征
        features.extend([
            team1_data.get('ranking_score', 0),
            team1_data.get('recent_winrate', 0),
            team1_data.get('avg_map_winrate', 0),
            team1_data.get('map_advantage_variance', 0)
        ])
        
        # 队伍1在指定地图的胜率
        if map_name and map_name in CS2_MAP_POOL:
            map_winrate_key = f'map_winrate_{map_name}'
            features.append(team1_data.get(map_winrate_key, 0))
        else:
            features.append(team1_data.get('avg_map_winrate', 0))
        
        # 队伍1选手特征聚合
        team1_player_features = self._aggregate_player_features(team1_players)
        features.extend(team1_player_features)
        
        # 队伍2特征（结构相同）
        features.extend([
            team2_data.get('ranking_score', 0),
            team2_data.get('recent_winrate', 0),
            team2_data.get('avg_map_winrate', 0),
            team2_data.get('map_advantage_variance', 0)
        ])
        
        if map_name and map_name in CS2_MAP_POOL:
            map_winrate_key = f'map_winrate_{map_name}'
            features.append(team2_data.get(map_winrate_key, 0))
        else:
            features.append(team2_data.get('avg_map_winrate', 0))
        
        team2_player_features = self._aggregate_player_features(team2_players)
        features.extend(team2_player_features)
        
        # 对比特征
        ranking_diff = team1_data.get('ranking_score', 0) - team2_data.get('ranking_score', 0)
        winrate_diff = team1_data.get('recent_winrate', 0) - team2_data.get('recent_winrate', 0)
        
        features.extend([ranking_diff, winrate_diff])
        
        return np.array(features, dtype=np.float32)
    
    def _aggregate_player_features(self, players: List[Dict]) -> List[float]:
        """聚合选手特征"""
        if not players:
            return [0.0] * 10  # 返回默认特征
        
        # 提取数值特征
        ratings = [p.get('adjusted_rating', 1.0) for p in players]
        kd_ratios = [p.get('kd_ratio', 1.0) for p in players]
        adrs = [p.get('adr', 70.0) for p in players]
        ages = [p.get('age', 23.0) for p in players]
        recent_forms = [p.get('recent_form', 0.5) for p in players]
        
        # 聚合统计
        aggregated = [
            np.mean(ratings),      # 平均评分
            np.max(ratings),       # 最高评分
            np.std(ratings),       # 评分标准差
            np.mean(kd_ratios),    # 平均K/D
            np.mean(adrs),         # 平均ADR
            np.mean(ages),         # 平均年龄
            np.mean(recent_forms), # 平均状态
            self.calculate_team_chemistry(players),  # 队伍默契度
            len([r for r in ratings if r > 1.1]),   # 明星选手数量
            np.min(ratings)        # 最弱选手评分
        ]
        
        return aggregated
    
    def save_preprocessed_data(self, players_df: pd.DataFrame, teams_df: pd.DataFrame):
        """保存预处理后的数据"""
        players_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'players_processed.csv'), index=False)
        teams_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'teams_processed.csv'), index=False)
        
        # 保存预处理器
        import joblib
        joblib.dump(self.player_scaler, os.path.join(PROCESSED_DATA_DIR, 'player_scaler.pkl'))
        joblib.dump(self.team_scaler, os.path.join(PROCESSED_DATA_DIR, 'team_scaler.pkl'))
        joblib.dump(self.imputers, os.path.join(PROCESSED_DATA_DIR, 'imputers.pkl'))
        
        self.logger.info("预处理数据和预处理器已保存")

if __name__ == "__main__":
    # 示例使用
    preprocessor = DataPreprocessor()
    
    # 加载原始数据
    try:
        players_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'players_raw.csv'))
        teams_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'teams_raw.csv'))
        
        # 预处理
        players_processed = preprocessor.preprocess_player_data(players_df)
        teams_processed = preprocessor.preprocess_team_data(teams_df)
        
        # 保存
        preprocessor.save_preprocessed_data(players_processed, teams_processed)
        
        print("数据预处理完成")
        
    except FileNotFoundError as e:
        print(f"未找到原始数据文件: {e}")
        print("请先运行 data_collector.py 收集数据")