"""
数据收集模块 - 使用hltv-async-api
从HLTV获取选手和队伍数据的异步版本
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os
import json
from pathlib import Path

# 导入新的HLTV API
try:
    from hltv_async_api import Hltv
    from hltv_async_api.models import Player, Team, Match, Event
except ImportError:
    print("❌ 请安装hltv-async-api: pip install hltv-async-api")
    raise

from config import (
    RAW_DATA_DIR, CS2_MAP_POOL, REQUEST_DELAY
)

class HLTVAsyncDataCollector:
    """HLTV异步数据收集器"""
    
    def __init__(self):
        self.raw_data_dir = Path(RAW_DATA_DIR)
        self.raw_data_dir.mkdir(exist_ok=True)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 请求限制
        self.request_delay = REQUEST_DELAY
        self.semaphore = asyncio.Semaphore(5)  # 限制并发请求数
        
    async def _rate_limit(self):
        """请求频率限制"""
        await asyncio.sleep(self.request_delay)
    
    async def get_top_teams(self, limit: int = 30) -> List[Dict]:
        """获取世界排名前N的队伍"""
        self.logger.info(f"获取世界排名前{limit}的队伍...")
        
        async with self.semaphore:
            try:
                async with Hltv() as hltv:
                    ranking = await hltv.get_team_ranking()
                    
                    teams_data = []
                    for i, team_info in enumerate(ranking[:limit]):
                        await self._rate_limit()
                        
                        team_data = {
                            'team_id': team_info.id,
                            'name': team_info.name,
                            'ranking': i + 1,
                            'points': getattr(team_info, 'points', 0),
                            'country': getattr(team_info, 'country', ''),
                        }
                        teams_data.append(team_data)
                        
                        self.logger.info(f"收集队伍: {team_info.name} (排名: {i + 1})")
                    
                    return teams_data
                    
            except Exception as e:
                self.logger.error(f"获取队伍排名失败: {e}")
                return []
    
    async def get_team_detailed_info(self, team_id: int) -> Optional[Dict]:
        """获取队伍详细信息"""
        async with self.semaphore:
            try:
                async with Hltv() as hltv:
                    await self._rate_limit()
                    
                    # 获取队伍基本信息
                    team_info = await hltv.get_team_info(team_id)
                    
                    # 获取队伍统计数据
                    await self._rate_limit()
                    team_stats = await hltv.get_team_stats(team_id)
                    
                    # 获取近期比赛
                    await self._rate_limit()
                    recent_matches = await hltv.get_team_results(team_id, limit=30)
                    
                    # 计算近期胜率
                    if recent_matches:
                        wins = sum(1 for match in recent_matches if match.won)
                        recent_winrate = wins / len(recent_matches)
                    else:
                        recent_winrate = 0.5
                    
                    # 提取地图统计
                    map_stats = {}
                    if hasattr(team_stats, 'map_stats') and team_stats.map_stats:
                        for map_stat in team_stats.map_stats:
                            map_name = map_stat.map_name.lower()
                            if map_name in CS2_MAP_POOL:
                                wins = getattr(map_stat, 'wins', 0)
                                total = getattr(map_stat, 'total_maps', 1)
                                map_stats[f'map_winrate_{map_name}'] = wins / max(total, 1)
                    
                    # 填充缺失的地图数据
                    for map_name in CS2_MAP_POOL:
                        key = f'map_winrate_{map_name}'
                        if key not in map_stats:
                            map_stats[key] = 0.5  # 默认50%胜率
                    
                    team_data = {
                        'team_id': team_id,
                        'name': team_info.name,
                        'country': getattr(team_info, 'country', ''),
                        'world_ranking': getattr(team_info, 'world_ranking', 999),
                        'recent_winrate': recent_winrate,
                        'total_maps_played': len(recent_matches) if recent_matches else 0,
                        **map_stats
                    }
                    
                    return team_data
                    
            except Exception as e:
                self.logger.error(f"获取队伍 {team_id} 详细信息失败: {e}")
                return None
    
    async def get_player_detailed_info(self, player_id: int) -> Optional[Dict]:
        """获取选手详细信息"""
        async with self.semaphore:
            try:
                async with Hltv() as hltv:
                    await self._rate_limit()
                    
                    # 获取选手基本信息
                    player_info = await hltv.get_player_info(player_id)
                    
                    # 获取选手统计数据
                    await self._rate_limit()
                    player_stats = await hltv.get_player_stats(player_id)
                    
                    # 计算经验月数
                    experience_months = 0
                    if hasattr(player_info, 'career_start') and player_info.career_start:
                        try:
                            start_date = datetime.strptime(str(player_info.career_start), '%Y-%m-%d')
                            experience_months = (datetime.now() - start_date).days / 30.44
                        except:
                            experience_months = 12  # 默认1年经验
                    
                    player_data = {
                        'player_id': player_id,
                        'name': player_info.nickname,
                        'real_name': getattr(player_info, 'real_name', ''),
                        'age': getattr(player_info, 'age', 23),
                        'country': getattr(player_info, 'country', ''),
                        'team_id': getattr(player_info, 'team_id', 0),
                        'team_name': getattr(player_info, 'team_name', ''),
                        
                        # 统计数据
                        'rating_2_0': getattr(player_stats, 'rating_2_0', 1.0),
                        'kd_ratio': getattr(player_stats, 'kd_ratio', 1.0),
                        'adr': getattr(player_stats, 'adr', 70.0),
                        'kpr': getattr(player_stats, 'kills_per_round', 0.7),
                        'spr': getattr(player_stats, 'survived_per_round', 0.5),
                        'headshot_pct': getattr(player_stats, 'headshot_percentage', 50.0),
                        'maps_played': getattr(player_stats, 'maps_played', 0),
                        'rounds_played': getattr(player_stats, 'rounds_played', 0),
                        'experience_months': max(0, experience_months),
                    }
                    
                    return player_data
                    
            except Exception as e:
                self.logger.error(f"获取选手 {player_id} 详细信息失败: {e}")
                return None
    
    async def get_team_players(self, team_id: int) -> List[int]:
        """获取队伍的选手ID列表"""
        try:
            async with Hltv() as hltv:
                await self._rate_limit()
                team_info = await hltv.get_team_info(team_id)
                
                player_ids = []
                if hasattr(team_info, 'players') and team_info.players:
                    player_ids = [player.id for player in team_info.players]
                
                return player_ids[:5]  # 最多5名主力选手
                
        except Exception as e:
            self.logger.error(f"获取队伍 {team_id} 选手列表失败: {e}")
            return []
    
    async def collect_teams_data(self, team_ids: List[int] = None, use_ranking: bool = True) -> pd.DataFrame:
        """批量收集队伍数据"""
        self.logger.info("开始收集队伍数据...")
        
        if team_ids is None and use_ranking:
            # 使用世界排名获取队伍
            top_teams = await self.get_top_teams(30)
            team_ids = [team['team_id'] for team in top_teams]
        elif team_ids is None:
            # 使用默认队伍ID
            team_ids = [4608, 5995, 6665, 7020, 6137]  # NAVI, G2, FaZe, Astralis, Liquid
        
        # 并发收集队伍详细信息
        tasks = [self.get_team_detailed_info(team_id) for team_id in team_ids]
        teams_data = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤成功的结果
        valid_teams = [team for team in teams_data if isinstance(team, dict)]
        
        if not valid_teams:
            self.logger.error("没有成功收集到任何队伍数据")
            return pd.DataFrame()
        
        df = pd.DataFrame(valid_teams)
        
        # 保存原始数据
        output_file = self.raw_data_dir / 'teams_raw.csv'
        df.to_csv(output_file, index=False)
        self.logger.info(f"队伍数据已保存: {output_file} (共{len(df)}支队伍)")
        
        return df
    
    async def collect_players_data(self, team_ids: List[int] = None) -> pd.DataFrame:
        """批量收集选手数据"""
        self.logger.info("开始收集选手数据...")
        
        if team_ids is None:
            # 尝试从已保存的队伍数据中获取
            try:
                teams_df = pd.read_csv(self.raw_data_dir / 'teams_raw.csv')
                team_ids = teams_df['team_id'].tolist()[:10]  # 限制队伍数量
            except:
                team_ids = [4608, 5995, 6665, 7020, 6137]  # 默认队伍
        
        # 收集所有队伍的选手ID
        all_player_ids = []
        for team_id in team_ids:
            player_ids = await self.get_team_players(team_id)
            all_player_ids.extend(player_ids)
        
        # 去重
        unique_player_ids = list(set(all_player_ids))
        self.logger.info(f"找到 {len(unique_player_ids)} 名独特选手")
        
        # 并发收集选手详细信息
        batch_size = 10  # 分批处理，避免过多并发请求
        players_data = []
        
        for i in range(0, len(unique_player_ids), batch_size):
            batch_ids = unique_player_ids[i:i + batch_size]
            self.logger.info(f"处理选手批次 {i//batch_size + 1}/{(len(unique_player_ids) + batch_size - 1)//batch_size}")
            
            tasks = [self.get_player_detailed_info(player_id) for player_id in batch_ids]
            batch_data = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 过滤成功的结果
            valid_players = [player for player in batch_data if isinstance(player, dict)]
            players_data.extend(valid_players)
            
            # 批次间延迟
            await asyncio.sleep(2)
        
        if not players_data:
            self.logger.error("没有成功收集到任何选手数据")
            return pd.DataFrame()
        
        df = pd.DataFrame(players_data)
        
        # 保存原始数据
        output_file = self.raw_data_dir / 'players_raw.csv'
        df.to_csv(output_file, index=False)
        self.logger.info(f"选手数据已保存: {output_file} (共{len(df)}名选手)")
        
        return df
    
    async def collect_matches_data(self, limit: int = 500) -> pd.DataFrame:
        """收集历史比赛数据"""
        self.logger.info(f"开始收集历史比赛数据 (限制: {limit} 场)...")
        
        try:
            async with Hltv() as hltv:
                # 获取近期比赛结果
                matches = await hltv.get_results(limit=limit)
                
                matches_data = []
                for match in matches:
                    match_data = {
                        'match_id': match.id,
                        'date': match.date,
                        'team1_id': match.team1.id if match.team1 else 0,
                        'team1_name': match.team1.name if match.team1 else '',
                        'team2_id': match.team2.id if match.team2 else 0,
                        'team2_name': match.team2.name if match.team2 else '',
                        'team1_score': getattr(match, 'team1_score', 0),
                        'team2_score': getattr(match, 'team2_score', 0),
                        'winner_id': match.winner.id if hasattr(match, 'winner') and match.winner else 0,
                        'event_name': match.event.name if hasattr(match, 'event') and match.event else '',
                        'map_name': getattr(match, 'map', '').lower(),
                        'format': getattr(match, 'format', ''),
                    }
                    matches_data.append(match_data)
                
                df = pd.DataFrame(matches_data)
                
                # 保存数据
                output_file = self.raw_data_dir / 'matches_raw.csv'
                df.to_csv(output_file, index=False)
                self.logger.info(f"比赛数据已保存: {output_file} (共{len(df)}场比赛)")
                
                return df
                
        except Exception as e:
            self.logger.error(f"收集比赛数据失败: {e}")
            return pd.DataFrame()
    
    async def collect_all_data(self, team_limit: int = 20, match_limit: int = 500):
        """收集所有数据"""
        self.logger.info("🚀 开始全量数据收集...")
        
        # 1. 收集队伍数据
        self.logger.info("📊 第1步: 收集队伍数据")
        teams_df = await self.collect_teams_data(use_ranking=True)
        
        if teams_df.empty:
            self.logger.error("队伍数据收集失败，停止后续收集")
            return
        
        # 2. 收集选手数据
        self.logger.info("👥 第2步: 收集选手数据")
        team_ids = teams_df['team_id'].tolist()[:team_limit]
        players_df = await self.collect_players_data(team_ids)
        
        # 3. 收集比赛数据
        self.logger.info("🎮 第3步: 收集比赛数据")
        matches_df = await self.collect_matches_data(match_limit)
        
        # 4. 生成数据报告
        self.logger.info("📋 生成数据收集报告...")
        report = {
            'collection_time': datetime.now().isoformat(),
            'teams_collected': len(teams_df),
            'players_collected': len(players_df),
            'matches_collected': len(matches_df),
            'team_ids': teams_df['team_id'].tolist() if not teams_df.empty else [],
            'maps_in_matches': matches_df['map_name'].value_counts().to_dict() if not matches_df.empty else {}
        }
        
        report_file = self.raw_data_dir / 'collection_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info("✅ 数据收集完成!")
        self.logger.info(f"   - 队伍: {len(teams_df)}")
        self.logger.info(f"   - 选手: {len(players_df)}")
        self.logger.info(f"   - 比赛: {len(matches_df)}")
        self.logger.info(f"   - 报告: {report_file}")

# 便捷函数
async def quick_collect_data():
    """快速数据收集"""
    collector = HLTVAsyncDataCollector()
    await collector.collect_all_data(team_limit=15, match_limit=300)

# 示例使用
if __name__ == "__main__":
    async def main():
        # 创建收集器
        collector = HLTVAsyncDataCollector()
        
        # 选择收集模式
        print("选择数据收集模式:")
        print("1. 快速收集 (15支队伍, 300场比赛)")
        print("2. 完整收集 (30支队伍, 500场比赛)")
        print("3. 自定义收集")
        
        choice = input("请选择 (1-3): ").strip()
        
        if choice == '1':
            await collector.collect_all_data(team_limit=15, match_limit=300)
        elif choice == '2':
            await collector.collect_all_data(team_limit=30, match_limit=500)
        elif choice == '3':
            team_limit = int(input("队伍数量: "))
            match_limit = int(input("比赛数量: "))
            await collector.collect_all_data(team_limit, match_limit)
        else:
            print("无效选择，使用快速收集模式")
            await collector.collect_all_data(team_limit=15, match_limit=300)
    
    # 运行异步主函数
    asyncio.run(main())