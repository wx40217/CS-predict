"""
HLTV数据收集器
从HLTV API获取比赛、选手、队伍等数据
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import yaml
from asyncio_throttle import Throttler

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HLTVCollector:
    """HLTV数据收集器"""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """初始化收集器"""
        self.config = self._load_config(config_path)
        self.base_url = self.config['data']['hltv_api']['base_url']
        self.rate_limit = self.config['data']['hltv_api']['rate_limit']
        self.timeout = self.config['data']['hltv_api']['timeout']
        self.retry_attempts = self.config['data']['hltv_api']['retry_attempts']
        
        # 创建限流器
        self.throttler = Throttler(rate_limit=self.rate_limit, period=60)
        
        # 数据存储路径
        self.raw_data_dir = Path(self.config['data']['storage']['raw_data_dir'])
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存路径
        self.cache_dir = Path(self.config['data']['storage']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据源配置
        self.sources = self.config['data']['collection']['sources']
        self.filters = self.config['data']['collection']['filters']
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    async def _make_request(self, session: aiohttp.ClientSession, 
                           endpoint: str, params: Dict = None) -> Optional[Dict]:
        """发送HTTP请求"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.retry_attempts):
            try:
                async with self.throttler:
                    async with session.get(url, params=params, timeout=self.timeout) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            logger.warning(f"请求失败: {response.status} - {url}")
                            
            except asyncio.TimeoutError:
                logger.warning(f"请求超时 (尝试 {attempt + 1}/{self.retry_attempts}): {url}")
            except Exception as e:
                logger.error(f"请求错误 (尝试 {attempt + 1}/{self.retry_attempts}): {e}")
            
            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
        
        return None
    
    async def collect_matches(self, start_date: str, end_date: str) -> List[Dict]:
        """收集比赛数据"""
        logger.info(f"开始收集比赛数据: {start_date} 到 {end_date}")
        
        all_matches = []
        page = 1
        
        async with aiohttp.ClientSession() as session:
            while True:
                params = {
                    'startDate': start_date,
                    'endDate': end_date,
                    'page': page
                }
                
                data = await self._make_request(session, '/api/matches.json', params)
                if not data or not isinstance(data, list):
                    break
                
                # 过滤比赛
                filtered_matches = self._filter_matches(data)
                all_matches.extend(filtered_matches)
                
                logger.info(f"已收集 {len(all_matches)} 场比赛")
                
                # 检查是否还有更多数据
                if len(data) < 100:  # 假设每页100条数据
                    break
                
                page += 1
                await asyncio.sleep(1)  # 避免过于频繁的请求
        
        logger.info(f"比赛数据收集完成，共 {len(all_matches)} 场")
        return all_matches
    
    def _filter_matches(self, matches: List[Dict]) -> List[Dict]:
        """过滤比赛数据"""
        filtered = []
        
        for match in matches:
            # 检查星级
            if match.get('stars', 0) < self.filters['min_stars']:
                continue
            
            # 检查地图数
            maps = match.get('maps', '')
            if isinstance(maps, str) and 'bo' in maps:
                map_count = int(maps.replace('bo', ''))
                if map_count < self.filters['min_maps']:
                    continue
            
            filtered.append(match)
        
        return filtered
    
    async def collect_match_details(self, match_ids: List[int]) -> List[Dict]:
        """收集比赛详细数据"""
        logger.info(f"开始收集 {len(match_ids)} 场比赛的详细数据")
        
        all_details = []
        
        async with aiohttp.ClientSession() as session:
            for i, match_id in enumerate(match_ids):
                if i % 100 == 0:
                    logger.info(f"进度: {i}/{len(match_ids)}")
                
                data = await self._make_request(session, f'/api/match.json?id={match_id}')
                if data:
                    all_details.append({
                        'match_id': match_id,
                        'details': data
                    })
                
                await asyncio.sleep(0.1)  # 避免过于频繁的请求
        
        logger.info(f"比赛详细数据收集完成，共 {len(all_details)} 场")
        return all_details
    
    async def collect_players(self) -> List[Dict]:
        """收集选手数据"""
        logger.info("开始收集选手数据")
        
        async with aiohttp.ClientSession() as session:
            data = await self._make_request(session, '/api/players.json')
            
            if data and isinstance(data, list):
                logger.info(f"选手数据收集完成，共 {len(data)} 名选手")
                return data
            else:
                logger.error("选手数据收集失败")
                return []
    
    async def collect_teams(self) -> List[Dict]:
        """收集队伍数据"""
        logger.info("开始收集队伍数据")
        
        async with aiohttp.ClientSession() as session:
            data = await self._make_request(session, '/api/teams.json')
            
            if data and isinstance(data, list):
                logger.info(f"队伍数据收集完成，共 {len(data)} 支队伍")
                return data
            else:
                logger.error("队伍数据收集失败")
                return []
    
    async def collect_player_details(self, player_ids: List[int]) -> List[Dict]:
        """收集选手详细数据"""
        logger.info(f"开始收集 {len(player_ids)} 名选手的详细数据")
        
        all_details = []
        
        async with aiohttp.ClientSession() as session:
            for i, player_id in enumerate(player_ids):
                if i % 100 == 0:
                    logger.info(f"进度: {i}/{len(player_ids)}")
                
                data = await self._make_request(session, f'/api/player.json?id={player_id}')
                if data:
                    all_details.append({
                        'player_id': player_id,
                        'details': data
                    })
                
                await asyncio.sleep(0.1)
        
        logger.info(f"选手详细数据收集完成，共 {len(all_details)} 名")
        return all_details
    
    async def collect_results(self, start_date: str, end_date: str) -> List[Dict]:
        """收集比赛结果数据"""
        logger.info(f"开始收集比赛结果: {start_date} 到 {end_date}")
        
        async with aiohttp.ClientSession() as session:
            params = {
                'startDate': start_date,
                'endDate': end_date
            }
            
            data = await self._make_request(session, '/api/results.json', params)
            
            if data and isinstance(data, list):
                logger.info(f"比赛结果收集完成，共 {len(data)} 条")
                return data
            else:
                logger.error("比赛结果收集失败")
                return []
    
    def save_data(self, data: List[Dict], filename: str, data_type: str):
        """保存数据到文件"""
        file_path = self.raw_data_dir / data_type / filename
        
        # 创建目录
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON格式
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已保存到: {file_path}")
    
    async def collect_all_data(self, start_date: str, end_date: str):
        """收集所有数据"""
        logger.info("开始收集所有HLTV数据")
        
        # 收集基础数据
        matches = await self.collect_matches(start_date, end_date)
        players = await self.collect_players()
        teams = await self.collect_teams()
        results = await self.collect_results(start_date, end_date)
        
        # 保存基础数据
        self.save_data(matches, f"matches_{start_date}_{end_date}.json", "matches")
        self.save_data(players, f"players_{start_date}_{end_date}.json", "players")
        self.save_data(teams, f"teams_{start_date}_{end_date}.json", "teams")
        self.save_data(results, f"results_{start_date}_{end_date}.json", "results")
        
        # 收集详细数据
        if matches:
            match_ids = [match['id'] for match in matches if 'id' in match]
            match_details = await self.collect_match_details(match_ids[:1000])  # 限制数量
            self.save_data(match_details, f"match_details_{start_date}_{end_date}.json", "match_details")
        
        if players:
            player_ids = [player['id'] for player in players if 'id' in player]
            player_details = await self.collect_player_details(player_ids[:2000])  # 限制数量
            self.save_data(player_details, f"player_details_{start_date}_{end_date}.json", "player_details")
        
        logger.info("所有数据收集完成")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据收集摘要"""
        summary = {}
        
        for data_type in ['matches', 'players', 'teams', 'results', 'match_details', 'player_details']:
            data_dir = self.raw_data_dir / data_type
            if data_dir.exists():
                files = list(data_dir.glob('*.json'))
                summary[data_type] = {
                    'file_count': len(files),
                    'files': [f.name for f in files]
                }
        
        return summary


async def main():
    """主函数"""
    # 创建收集器
    collector = HLTVCollector()
    
    # 设置时间范围
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    # 收集数据
    await collector.collect_all_data(start_date, end_date)
    
    # 显示摘要
    summary = collector.get_data_summary()
    print("数据收集摘要:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())