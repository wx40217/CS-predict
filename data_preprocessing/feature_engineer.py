"""
特征工程模块
处理原始HLTV数据，生成模型训练所需的特征
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSFeatureEngineer:
    """CS特征工程师"""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """初始化特征工程师"""
        self.config = self._load_config(config_path)
        self.raw_data_dir = Path(self.config['data']['storage']['raw_data_dir'])
        self.processed_data_dir = Path(self.config['data']['storage']['processed_data_dir'])
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 特征配置
        self.features = self.config['data']['preprocessing']['features']
        self.age_processing = self.config['data']['preprocessing']['age_processing']
        
        # 数据清洗配置
        self.cleaning = self.config['data']['preprocessing']['cleaning']
        self.normalization = self.config['data']['preprocessing']['normalization']
        
        # 初始化特征处理器
        self._init_processors()
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _init_processors(self):
        """初始化特征处理器"""
        # 缺失值填充器
        self.imputer = SimpleImputer(strategy='mean')
        
        # 特征标准化器
        if self.normalization['method'] == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization['method'] == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        # 年龄分组标签
        self.age_labels = self.age_processing['age_labels']
        self.age_bins = self.age_processing['age_bins']
        
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """加载原始数据"""
        logger.info("开始加载原始数据")
        
        data = {}
        
        # 加载比赛数据
        matches_dir = self.raw_data_dir / "matches"
        if matches_dir.exists():
            matches_files = list(matches_dir.glob("*.json"))
            if matches_files:
                with open(matches_files[0], 'r', encoding='utf-8') as f:
                    matches_data = json.load(f)
                data['matches'] = pd.DataFrame(matches_data)
                logger.info(f"加载比赛数据: {len(data['matches'])} 场")
        
        # 加载选手数据
        players_dir = self.raw_data_dir / "players"
        if players_dir.exists():
            players_files = list(players_dir.glob("*.json"))
            if players_files:
                with open(players_files[0], 'r', encoding='utf-8') as f:
                    players_data = json.load(f)
                data['players'] = pd.DataFrame(players_data)
                logger.info(f"加载选手数据: {len(data['players'])} 名")
        
        # 加载队伍数据
        teams_dir = self.raw_data_dir / "teams"
        if teams_dir.exists():
            teams_files = list(teams_dir.glob("*.json"))
            if teams_files:
                with open(teams_files[0], 'r', encoding='utf-8') as f:
                    teams_data = json.load(f)
                data['teams'] = pd.DataFrame(teams_data)
                logger.info(f"加载队伍数据: {len(data['teams'])} 支")
        
        # 加载比赛结果数据
        results_dir = self.raw_data_dir / "results"
        if results_dir.exists():
            results_files = list(results_dir.glob("*.json"))
            if results_files:
                with open(results_files[0], 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                data['results'] = pd.DataFrame(results_data)
                logger.info(f"加载比赛结果: {len(data['results'])} 条")
        
        return data
    
    def clean_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """数据清洗"""
        logger.info("开始数据清洗")
        
        cleaned_data = {}
        
        for data_type, df in data.items():
            if df is None or df.empty:
                continue
                
            logger.info(f"清洗 {data_type} 数据")
            
            # 删除重复行
            if self.cleaning['remove_duplicates']:
                df = df.drop_duplicates()
                logger.info(f"删除重复行后: {len(df)} 行")
            
            # 处理缺失值
            if self.cleaning['fill_missing']:
                df = self._handle_missing_values(df, data_type)
            
            # 异常值处理
            if self.cleaning['outlier_removal']:
                df = self._remove_outliers(df, data_type)
            
            cleaned_data[data_type] = df
        
        return cleaned_data
    
    def _handle_missing_values(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """处理缺失值"""
        # 数值列使用均值填充
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if not numeric_columns.empty:
            df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        
        # 分类列使用众数填充
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_value)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """移除异常值"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['id', 'match_id', 'player_id', 'team_id']:
                continue  # 跳过ID列
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 标记异常值
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                logger.info(f"{data_type} - {col}: 发现 {outlier_count} 个异常值")
                # 将异常值替换为边界值
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def engineer_player_features(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """工程化选手特征"""
        logger.info("开始工程化选手特征")
        
        if players_df is None or players_df.empty:
            return pd.DataFrame()
        
        # 复制数据
        df = players_df.copy()
        
        # 基础特征
        player_features = []
        
        # 数值特征
        numeric_features = ['rating', 'impact', 'dpr', 'adr', 'kast', 'kpr', 'headshots']
        for feature in numeric_features:
            if feature in df.columns:
                player_features.append(feature)
        
        # 年龄特征
        if 'age' in df.columns:
            player_features.append('age')
            # 计算经验年限（假设16岁开始职业）
            df['experience_years'] = df['age'] - 16
            df['experience_years'] = df['experience_years'].clip(lower=0, upper=20)
            player_features.append('experience_years')
            
            # 年龄分组
            df['age_group'] = pd.cut(df['age'], bins=self.age_bins, labels=self.age_labels)
            df['age_group_encoded'] = df['age_group'].cat.codes
            
            # 年龄权重特征
            age_weights = self.age_processing['age_weights']
            df['age_weight'] = df['age_group'].map({
                'teen': age_weights['youth_factor'],
                'young': age_weights['youth_factor'],
                'prime': age_weights['prime_factor'],
                'veteran': age_weights['veteran_factor'],
                'legend': age_weights['legend_factor']
            })
            player_features.extend(['age_group_encoded', 'age_weight'])
        
        # 地图相关特征
        if 'mapsPlayed' in df.columns:
            player_features.append('mapsPlayed')
            # 地图经验等级
            df['map_experience_level'] = pd.cut(
                df['mapsPlayed'], 
                bins=[0, 100, 500, 1000, 2000, float('inf')],
                labels=['Novice', 'Beginner', 'Intermediate', 'Advanced', 'Expert']
            )
            df['map_experience_encoded'] = df['map_experience_level'].cat.codes
            player_features.append('map_experience_encoded')
        
        # 计算综合评分
        if all(feature in df.columns for feature in ['rating', 'impact', 'adr']):
            df['composite_score'] = (
                df['rating'] * 0.4 + 
                df['impact'] * 0.3 + 
                df['adr'] * 0.3
            )
            player_features.append('composite_score')
        
        # 选择最终特征
        final_features = ['id', 'nickname', 'team'] + player_features
        
        return df[final_features].copy()
    
    def engineer_team_features(self, teams_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
        """工程化团队特征"""
        logger.info("开始工程化团队特征")
        
        if teams_df is None or teams_df.empty:
            return pd.DataFrame()
        
        # 复制数据
        df = teams_df.copy()
        
        # 如果有选手数据，计算团队统计
        if players_df is not None and not players_df.empty:
            team_stats = players_df.groupby('team').agg({
                'rating': ['mean', 'std', 'count'],
                'age': ['mean', 'std'],
                'experience_years': ['mean', 'std'],
                'mapsPlayed': ['mean', 'sum'],
                'composite_score': ['mean', 'std']
            }).round(3)
            
            # 扁平化列名
            team_stats.columns = ['_'.join(col).strip() for col in team_stats.columns]
            team_stats = team_stats.reset_index()
            
            # 合并团队数据
            df = df.merge(team_stats, left_on='name', right_on='team', how='left')
            
            # 计算团队化学指标
            if 'rating_std' in df.columns:
                df['team_chemistry'] = 1.0 / (1.0 + df['rating_std'])
                df['team_consistency'] = 1.0 - df['rating_std'] / df['rating_mean']
            
            # 计算团队年龄优势
            if 'age_mean' in df.columns:
                df['team_age_advantage'] = np.where(
                    df['age_mean'] <= 25, 1.2,  # 年轻团队优势
                    np.where(df['age_mean'] <= 28, 1.0,  # 黄金年龄
                    np.where(df['age_mean'] <= 32, 0.9,  # 老将团队
                    0.8))  # 传奇团队
                )
        
        # 选择最终特征
        final_features = ['id', 'name']
        if 'rating_mean' in df.columns:
            final_features.extend(['rating_mean', 'team_chemistry', 'team_consistency', 
                                'team_age_advantage', 'age_mean', 'experience_years_mean'])
        
        return df[final_features].copy()
    
    def engineer_match_features(self, matches_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
        """工程化比赛特征"""
        logger.info("开始工程化比赛特征")
        
        if matches_df is None or matches_df.empty:
            return pd.DataFrame()
        
        # 复制数据
        df = matches_df.copy()
        
        # 基础特征
        match_features = ['id', 'time', 'event', 'stars', 'maps', 'teams']
        
        # 时间特征
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['year'] = df['time'].dt.year
            df['month'] = df['time'].dt.month
            df['day_of_week'] = df['time'].dt.dayofweek
            match_features.extend(['year', 'month', 'day_of_week'])
        
        # 星级特征
        if 'stars' in df.columns:
            df['is_high_stakes'] = (df['stars'] >= 3).astype(int)
            match_features.append('is_high_stakes')
        
        # 地图特征
        if 'maps' in df.columns:
            df['map_count'] = df['maps'].str.extract(r'bo(\d+)').astype(int)
            df['is_bo3'] = (df['map_count'] == 3).astype(int)
            df['is_bo5'] = (df['map_count'] == 5).astype(int)
            match_features.extend(['map_count', 'is_bo3', 'is_bo5'])
        
        # 如果有结果数据，合并
        if results_df is not None and not results_df.empty():
            # 这里需要根据实际数据结构进行合并
            pass
        
        return df[match_features].copy()
    
    def create_training_dataset(self, cleaned_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """创建训练数据集"""
        logger.info("开始创建训练数据集")
        
        # 工程化特征
        player_features = self.engineer_player_features(cleaned_data.get('players'))
        team_features = self.engineer_team_features(cleaned_data.get('teams'), cleaned_data.get('players'))
        match_features = self.engineer_match_features(cleaned_data.get('matches'), cleaned_data.get('results'))
        
        # 创建选图预测数据集
        map_dataset = self._create_map_dataset(player_features, team_features, match_features)
        
        # 创建胜率预测数据集
        winrate_dataset = self._create_winrate_dataset(player_features, team_features, match_features)
        
        return map_dataset, winrate_dataset
    
    def _create_map_dataset(self, player_features: pd.DataFrame, team_features: pd.DataFrame, 
                           match_features: pd.DataFrame) -> pd.DataFrame:
        """创建选图预测数据集"""
        logger.info("创建选图预测数据集")
        
        # 这里需要根据实际的比赛数据结构来创建训练样本
        # 暂时返回空DataFrame，实际实现需要根据数据结构调整
        return pd.DataFrame()
    
    def _create_winrate_dataset(self, player_features: pd.DataFrame, team_features: pd.DataFrame, 
                               match_features: pd.DataFrame) -> pd.DataFrame:
        """创建胜率预测数据集"""
        logger.info("创建胜率预测数据集")
        
        # 这里需要根据实际的比赛数据结构来创建训练样本
        # 暂时返回空DataFrame，实际实现需要根据数据结构调整
        return pd.DataFrame()
    
    def normalize_features(self, dataset: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """特征标准化"""
        logger.info("开始特征标准化")
        
        if dataset.empty or not feature_columns:
            return dataset
        
        # 复制数据
        df = dataset.copy()
        
        # 标准化指定特征
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        
        logger.info(f"标准化完成，特征: {feature_columns}")
        return df
    
    def save_processed_data(self, map_dataset: pd.DataFrame, winrate_dataset: pd.DataFrame):
        """保存处理后的数据"""
        logger.info("保存处理后的数据")
        
        # 保存选图数据集
        if not map_dataset.empty:
            map_path = self.processed_data_dir / "map_prediction_dataset.parquet"
            map_dataset.to_parquet(map_path, compression='gzip')
            logger.info(f"选图数据集已保存: {map_path}")
        
        # 保存胜率数据集
        if not winrate_dataset.empty:
            winrate_path = self.processed_data_dir / "winrate_prediction_dataset.parquet"
            winrate_dataset.to_parquet(winrate_path, compression='gzip')
            logger.info(f"胜率数据集已保存: {winrate_path}")
        
        # 保存特征处理器
        processor_path = self.processed_data_dir / "feature_processors.pkl"
        with open(processor_path, 'wb') as f:
            pickle.dump({
                'imputer': self.imputer,
                'scaler': self.scaler
            }, f)
        logger.info(f"特征处理器已保存: {processor_path}")
    
    def process_all_data(self):
        """处理所有数据"""
        logger.info("开始处理所有数据")
        
        # 加载原始数据
        raw_data = self.load_raw_data()
        
        # 数据清洗
        cleaned_data = self.clean_data(raw_data)
        
        # 创建训练数据集
        map_dataset, winrate_dataset = self.create_training_dataset(cleaned_data)
        
        # 特征标准化
        if not map_dataset.empty:
            numeric_features = map_dataset.select_dtypes(include=[np.number]).columns.tolist()
            map_dataset = self.normalize_features(map_dataset, numeric_features)
        
        if not winrate_dataset.empty:
            numeric_features = winrate_dataset.select_dtypes(include=[np.number]).columns.tolist()
            winrate_dataset = self.normalize_features(winrate_dataset, numeric_features)
        
        # 保存处理后的数据
        self.save_processed_data(map_dataset, winrate_dataset)
        
        logger.info("数据处理完成")


def main():
    """主函数"""
    # 创建特征工程师
    engineer = CSFeatureEngineer()
    
    # 处理所有数据
    engineer.process_all_data()


if __name__ == "__main__":
    main()