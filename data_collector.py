"""
数据收集模块 - 从HLTV API获取选手和队伍数据
"""
import requests
import time
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os
from config import (
    HLTV_API_BASE_URL, REQUEST_DELAY, MAX_RETRIES, 
    RAW_DATA_DIR, CS2_MAP_POOL
)

class HLTVDataCollector:
    """HLTV数据收集器"""
    
    def __init__(self):
        self.base_url = HLTV_API_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # 创建数据目录
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """发送API请求"""
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY)  # 控制请求频率
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"请求失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES - 1:
                    self.logger.error(f"请求最终失败: {url}")
                    return None
                time.sleep(REQUEST_DELAY * (attempt + 1))
        
        return None
    
    def get_player_stats(self, player_id: int) -> Optional[Dict]:
        """获取选手统计数据"""
        return self._make_request(f"players/{player_id}/stats")
    
    def get_team_info(self, team_id: int) -> Optional[Dict]:
        """获取队伍信息"""
        return self._make_request(f"teams/{team_id}")
    
    def get_team_stats(self, team_id: int) -> Optional[Dict]:
        """获取队伍统计数据"""
        return self._make_request(f"teams/{team_id}/stats")
    
    def get_match_results(self, team_id: int, limit: int = 50) -> Optional[List[Dict]]:
        """获取队伍近期比赛结果"""
        return self._make_request(f"teams/{team_id}/results", {"limit": limit})
    
    def get_rankings(self) -> Optional[List[Dict]]:
        """获取世界排名"""
        return self._make_request("rankings")
    
    def collect_player_data(self, player_ids: List[int]) -> pd.DataFrame:
        """批量收集选手数据"""
        players_data = []
        
        for player_id in player_ids:
            self.logger.info(f"收集选手数据: {player_id}")
            
            # 获取选手基本信息和统计数据
            player_stats = self.get_player_stats(player_id)
            if not player_stats:
                continue
            
            # 提取关键特征
            player_data = {
                'player_id': player_id,
                'name': player_stats.get('name', ''),
                'age': player_stats.get('age', 0),
                'country': player_stats.get('country', ''),
                'team_id': player_stats.get('team', {}).get('id', 0),
                'rating_2_0': player_stats.get('statistics', {}).get('rating2', 0.0),
                'kd_ratio': player_stats.get('statistics', {}).get('killDeathRatio', 0.0),
                'adr': player_stats.get('statistics', {}).get('averageDamagePerRound', 0.0),
                'kpr': player_stats.get('statistics', {}).get('killsPerRound', 0.0),
                'headshot_pct': player_stats.get('statistics', {}).get('headshotPercentage', 0.0),
                'maps_played': player_stats.get('statistics', {}).get('mapsPlayed', 0),
                'rounds_played': player_stats.get('statistics', {}).get('roundsPlayed', 0),
            }
            
            # 计算经验月数（基于首次记录时间）
            first_match_date = player_stats.get('firstMatchDate')
            if first_match_date:
                start_date = datetime.strptime(first_match_date, '%Y-%m-%d')
                experience_months = (datetime.now() - start_date).days / 30.44
                player_data['experience_months'] = max(0, experience_months)
            else:
                player_data['experience_months'] = 0
            
            players_data.append(player_data)
        
        df = pd.DataFrame(players_data)
        
        # 保存原始数据
        df.to_csv(os.path.join(RAW_DATA_DIR, 'players_raw.csv'), index=False)
        self.logger.info(f"收集了 {len(df)} 名选手的数据")
        
        return df
    
    def collect_team_data(self, team_ids: List[int]) -> pd.DataFrame:
        """批量收集队伍数据"""
        teams_data = []
        
        for team_id in team_ids:
            self.logger.info(f"收集队伍数据: {team_id}")
            
            # 获取队伍基本信息
            team_info = self.get_team_info(team_id)
            if not team_info:
                continue
            
            # 获取队伍统计数据
            team_stats = self.get_team_stats(team_id)
            
            # 获取近期比赛结果
            recent_matches = self.get_match_results(team_id, 30)
            
            # 计算近期胜率
            recent_winrate = 0.0
            if recent_matches:
                wins = sum(1 for match in recent_matches if match.get('result') == 'win')
                recent_winrate = wins / len(recent_matches)
            
            # 提取地图胜率
            map_winrates = {}
            if team_stats and 'mapStatistics' in team_stats:
                for map_stat in team_stats['mapStatistics']:
                    map_name = map_stat.get('mapName', '').lower()
                    if map_name in CS2_MAP_POOL:
                        wins = map_stat.get('wins', 0)
                        total = map_stat.get('totalMaps', 1)
                        map_winrates[map_name] = wins / total if total > 0 else 0.0
            
            team_data = {
                'team_id': team_id,
                'name': team_info.get('name', ''),
                'country': team_info.get('country', ''),
                'world_ranking': team_info.get('rank', 999),
                'recent_winrate': recent_winrate,
                'total_maps_played': team_stats.get('mapsPlayed', 0) if team_stats else 0,
                **{f'map_winrate_{map_name}': map_winrates.get(map_name, 0.0) 
                   for map_name in CS2_MAP_POOL}
            }
            
            teams_data.append(team_data)
        
        df = pd.DataFrame(teams_data)
        
        # 保存原始数据
        df.to_csv(os.path.join(RAW_DATA_DIR, 'teams_raw.csv'), index=False)
        self.logger.info(f"收集了 {len(df)} 支队伍的数据")
        
        return df
    
    def collect_match_data(self, limit: int = 1000) -> pd.DataFrame:
        """收集历史比赛数据用于训练"""
        # 这里需要根据实际API端点调整
        # 假设有获取历史比赛的端点
        matches_data = []
        
        # 示例：获取近期比赛数据
        matches = self._make_request("matches/results", {"limit": limit})
        
        if matches:
            for match in matches:
                match_data = {
                    'match_id': match.get('id'),
                    'date': match.get('date'),
                    'team1_id': match.get('team1', {}).get('id'),
                    'team2_id': match.get('team2', {}).get('id'),
                    'map_name': match.get('map', '').lower(),
                    'winner_id': match.get('winner', {}).get('id'),
                    'score_team1': match.get('result', {}).get('team1Score', 0),
                    'score_team2': match.get('result', {}).get('team2Score', 0),
                    'event_name': match.get('event', {}).get('name', ''),
                }
                matches_data.append(match_data)
        
        df = pd.DataFrame(matches_data)
        
        if not df.empty:
            df.to_csv(os.path.join(RAW_DATA_DIR, 'matches_raw.csv'), index=False)
            self.logger.info(f"收集了 {len(df)} 场比赛的数据")
        
        return df

if __name__ == "__main__":
    # 示例使用
    collector = HLTVDataCollector()
    
    # 示例队伍ID（需要替换为实际ID）
    sample_team_ids = [4608, 5995, 6665, 7020]  # G2, FaZe, NAVI, Astralis等
    
    # 收集数据
    teams_df = collector.collect_team_data(sample_team_ids)
    print("队伍数据收集完成")
    
    # 可以继续收集选手数据等...