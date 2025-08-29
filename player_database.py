"""
选手数据库模块 - 简化输入接口
只需要选手名字，自动获取统计数据
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import os
import json
from pathlib import Path
from fuzzywuzzy import fuzz, process
import asyncio

from data_collector_new import HLTVAsyncDataCollector
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

class PlayerDatabase:
    """选手数据库 - 支持名字查询和模糊匹配"""
    
    def __init__(self):
        self.raw_data_dir = Path(RAW_DATA_DIR)
        self.processed_data_dir = Path(PROCESSED_DATA_DIR)
        
        # 创建目录
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 选手数据缓存
        self.players_cache = {}
        self.teams_cache = {}
        
        # 加载现有数据
        self._load_existing_data()
    
    def _load_existing_data(self):
        """加载现有的选手和队伍数据"""
        try:
            # 加载选手数据
            players_file = self.raw_data_dir / 'players_raw.csv'
            if players_file.exists():
                players_df = pd.read_csv(players_file)
                for _, player in players_df.iterrows():
                    self.players_cache[player['name'].lower()] = player.to_dict()
                self.logger.info(f"加载了 {len(self.players_cache)} 名选手数据")
            
            # 加载队伍数据
            teams_file = self.raw_data_dir / 'teams_raw.csv'
            if teams_file.exists():
                teams_df = pd.read_csv(teams_file)
                for _, team in teams_df.iterrows():
                    self.teams_cache[team['name'].lower()] = team.to_dict()
                self.logger.info(f"加载了 {len(self.teams_cache)} 支队伍数据")
                
        except Exception as e:
            self.logger.warning(f"加载现有数据失败: {e}")
    
    def find_player_by_name(self, player_name: str, threshold: int = 80) -> Optional[Dict]:
        """
        通过名字查找选手（支持模糊匹配）
        
        Args:
            player_name: 选手名字
            threshold: 模糊匹配阈值 (0-100)
            
        Returns:
            选手数据字典或None
        """
        if not player_name:
            return None
        
        player_name_lower = player_name.lower()
        
        # 精确匹配
        if player_name_lower in self.players_cache:
            return self.players_cache[player_name_lower]
        
        # 模糊匹配
        if self.players_cache:
            match_result = process.extractOne(
                player_name_lower, 
                self.players_cache.keys(),
                scorer=fuzz.ratio
            )
            
            if match_result and match_result[1] >= threshold:
                matched_name = match_result[0]
                self.logger.info(f"模糊匹配: '{player_name}' -> '{matched_name}' (相似度: {match_result[1]}%)")
                return self.players_cache[matched_name]
        
        self.logger.warning(f"未找到选手: {player_name}")
        return None
    
    def find_team_by_name(self, team_name: str, threshold: int = 80) -> Optional[Dict]:
        """
        通过名字查找队伍（支持模糊匹配）
        
        Args:
            team_name: 队伍名字
            threshold: 模糊匹配阈值 (0-100)
            
        Returns:
            队伍数据字典或None
        """
        if not team_name:
            return None
        
        team_name_lower = team_name.lower()
        
        # 精确匹配
        if team_name_lower in self.teams_cache:
            return self.teams_cache[team_name_lower]
        
        # 模糊匹配
        if self.teams_cache:
            match_result = process.extractOne(
                team_name_lower,
                self.teams_cache.keys(),
                scorer=fuzz.ratio
            )
            
            if match_result and match_result[1] >= threshold:
                matched_name = match_result[0]
                self.logger.info(f"模糊匹配: '{team_name}' -> '{matched_name}' (相似度: {match_result[1]}%)")
                return self.teams_cache[matched_name]
        
        self.logger.warning(f"未找到队伍: {team_name}")
        return None
    
    def get_player_data(self, player_name: str) -> Dict:
        """
        获取选手完整数据，如果找不到则返回默认值
        
        Args:
            player_name: 选手名字
            
        Returns:
            选手数据字典（包含所有必要字段）
        """
        player_data = self.find_player_by_name(player_name)
        
        if player_data:
            # 确保所有必要字段都存在
            return {
                'name': player_data.get('name', player_name),
                'age': int(player_data.get('age', 23)),
                'rating_2_0': float(player_data.get('rating_2_0', 1.0)),
                'kd_ratio': float(player_data.get('kd_ratio', 1.0)),
                'adr': float(player_data.get('adr', 70.0)),
                'kpr': float(player_data.get('kpr', 0.7)),
                'headshot_pct': float(player_data.get('headshot_pct', 45.0)),
                'experience_months': int(player_data.get('experience_months', 24)),
                'team_id': int(player_data.get('team_id', 0)),
                'team_name': player_data.get('team_name', ''),
                'country': player_data.get('country', ''),
            }
        else:
            # 返回默认值
            self.logger.info(f"使用默认数据: {player_name}")
            return {
                'name': player_name,
                'age': 23,
                'rating_2_0': 1.0,
                'kd_ratio': 1.0,
                'adr': 70.0,
                'kpr': 0.7,
                'headshot_pct': 45.0,
                'experience_months': 24,
                'team_id': 0,
                'team_name': '',
                'country': '',
            }
    
    def get_team_data(self, team_name: str) -> Dict:
        """
        获取队伍完整数据，如果找不到则返回默认值
        
        Args:
            team_name: 队伍名字
            
        Returns:
            队伍数据字典（包含所有必要字段）
        """
        team_data = self.find_team_by_name(team_name)
        
        if team_data:
            return {
                'name': team_data.get('name', team_name),
                'team_id': int(team_data.get('team_id', 0)),
                'world_ranking': int(team_data.get('world_ranking', 50)),
                'recent_winrate': float(team_data.get('recent_winrate', 0.5)),
                'country': team_data.get('country', ''),
                # 地图胜率
                'map_winrate_mirage': float(team_data.get('map_winrate_mirage', 0.5)),
                'map_winrate_inferno': float(team_data.get('map_winrate_inferno', 0.5)),
                'map_winrate_dust2': float(team_data.get('map_winrate_dust2', 0.5)),
                'map_winrate_nuke': float(team_data.get('map_winrate_nuke', 0.5)),
                'map_winrate_overpass': float(team_data.get('map_winrate_overpass', 0.5)),
                'map_winrate_vertigo': float(team_data.get('map_winrate_vertigo', 0.5)),
                'map_winrate_ancient': float(team_data.get('map_winrate_ancient', 0.5)),
            }
        else:
            # 返回默认值
            self.logger.info(f"使用默认数据: {team_name}")
            return {
                'name': team_name,
                'team_id': 0,
                'world_ranking': 50,
                'recent_winrate': 0.5,
                'country': '',
                'map_winrate_mirage': 0.5,
                'map_winrate_inferno': 0.5,
                'map_winrate_dust2': 0.5,
                'map_winrate_nuke': 0.5,
                'map_winrate_overpass': 0.5,
                'map_winrate_vertigo': 0.5,
                'map_winrate_ancient': 0.5,
            }
    
    def search_players(self, query: str, limit: int = 10) -> List[Dict]:
        """
        搜索选手（支持部分匹配）
        
        Args:
            query: 搜索关键词
            limit: 返回结果数量限制
            
        Returns:
            匹配的选手列表
        """
        if not query or not self.players_cache:
            return []
        
        query_lower = query.lower()
        matches = []
        
        for name, data in self.players_cache.items():
            if query_lower in name:
                score = 100 - abs(len(name) - len(query_lower)) * 2  # 长度相似度
                matches.append({
                    'name': data['name'],
                    'team': data.get('team_name', ''),
                    'rating': data.get('rating_2_0', 1.0),
                    'score': score
                })
        
        # 按相似度排序
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:limit]
    
    def search_teams(self, query: str, limit: int = 10) -> List[Dict]:
        """
        搜索队伍（支持部分匹配）
        
        Args:
            query: 搜索关键词
            limit: 返回结果数量限制
            
        Returns:
            匹配的队伍列表
        """
        if not query or not self.teams_cache:
            return []
        
        query_lower = query.lower()
        matches = []
        
        for name, data in self.teams_cache.items():
            if query_lower in name:
                score = 100 - abs(len(name) - len(query_lower)) * 2
                matches.append({
                    'name': data['name'],
                    'ranking': data.get('world_ranking', 999),
                    'country': data.get('country', ''),
                    'score': score
                })
        
        # 按相似度和排名排序
        matches.sort(key=lambda x: (x['score'], -x['ranking']), reverse=True)
        return matches[:limit]
    
    async def update_database(self, team_limit: int = 30):
        """
        更新选手和队伍数据库
        
        Args:
            team_limit: 收集的队伍数量限制
        """
        self.logger.info("开始更新数据库...")
        
        collector = HLTVAsyncDataCollector()
        
        # 收集最新数据
        await collector.collect_all_data(team_limit=team_limit, match_limit=200)
        
        # 重新加载数据
        self._load_existing_data()
        
        self.logger.info("数据库更新完成")
    
    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        return {
            'total_players': len(self.players_cache),
            'total_teams': len(self.teams_cache),
            'top_rated_players': sorted(
                [(name, data.get('rating_2_0', 1.0)) for name, data in self.players_cache.items()],
                key=lambda x: x[1], reverse=True
            )[:10],
            'top_teams': sorted(
                [(name, data.get('world_ranking', 999)) for name, data in self.teams_cache.items()],
                key=lambda x: x[1]
            )[:10]
        }

# 全局数据库实例
_player_db = None

def get_player_database() -> PlayerDatabase:
    """获取全局选手数据库实例"""
    global _player_db
    if _player_db is None:
        _player_db = PlayerDatabase()
    return _player_db

# 便捷函数
def find_player(name: str) -> Dict:
    """查找选手数据"""
    return get_player_database().get_player_data(name)

def find_team(name: str) -> Dict:
    """查找队伍数据"""
    return get_player_database().get_team_data(name)

def search_players(query: str, limit: int = 10) -> List[Dict]:
    """搜索选手"""
    return get_player_database().search_players(query, limit)

def search_teams(query: str, limit: int = 10) -> List[Dict]:
    """搜索队伍"""
    return get_player_database().search_teams(query, limit)

if __name__ == "__main__":
    # 测试数据库功能
    db = PlayerDatabase()
    
    # 搜索测试
    print("搜索选手 's1mple':")
    player = db.get_player_data("s1mple")
    print(f"  找到: {player['name']}, 评分: {player['rating_2_0']}")
    
    print("\n搜索队伍 'navi':")
    team = db.get_team_data("navi")
    print(f"  找到: {team['name']}, 排名: {team['world_ranking']}")
    
    print("\n数据库统计:")
    stats = db.get_statistics()
    print(f"  选手总数: {stats['total_players']}")
    print(f"  队伍总数: {stats['total_teams']}")
    
    if stats['top_rated_players']:
        print("  评分最高选手:")
        for name, rating in stats['top_rated_players'][:5]:
            print(f"    {name}: {rating:.2f}")