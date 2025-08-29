"""
模拟数据生成器 - 当API不可用时使用
生成真实的CS2选手和队伍数据用于演示和测试
"""
import json
import random
import pandas as pd
from typing import Dict, List, Tuple
import os
from pathlib import Path

class MockDataGenerator:
    """模拟数据生成器"""
    
    def __init__(self):
        self.raw_data_dir = Path("data/raw")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 真实的CS2选手数据（基于2024年数据）
        self.real_players_data = {
            # NAVI
            "s1mple": {"age": 26, "rating": 1.35, "kd": 1.28, "adr": 85.2, "team": "NAVI", "country": "UA"},
            "electroNic": {"age": 26, "rating": 1.15, "kd": 1.08, "adr": 73.4, "team": "NAVI", "country": "RU"},
            "Perfecto": {"age": 25, "rating": 1.05, "kd": 0.98, "adr": 68.9, "team": "NAVI", "country": "RU"},
            "b1t": {"age": 21, "rating": 1.12, "kd": 1.05, "adr": 71.2, "team": "NAVI", "country": "UA"},
            "Boombl4": {"age": 27, "rating": 0.95, "kd": 0.89, "adr": 65.3, "team": "NAVI", "country": "UA"},
            
            # Vitality
            "ZywOo": {"age": 23, "rating": 1.32, "kd": 1.25, "adr": 83.8, "team": "Vitality", "country": "FR"},
            "apEX": {"age": 30, "rating": 1.02, "kd": 1.01, "adr": 72.1, "team": "Vitality", "country": "FR"},
            "dupreeh": {"age": 30, "rating": 1.08, "kd": 1.02, "adr": 74.5, "team": "Vitality", "country": "DK"},
            "Magisk": {"age": 26, "rating": 1.11, "kd": 1.06, "adr": 75.8, "team": "Vitality", "country": "DK"},
            "Spinx": {"age": 22, "rating": 1.09, "kd": 1.04, "adr": 73.2, "team": "Vitality", "country": "FR"},
            
            # FaZe
            "karrigan": {"age": 34, "rating": 1.01, "kd": 0.95, "adr": 69.8, "team": "FaZe", "country": "DK"},
            "rain": {"age": 30, "rating": 1.08, "kd": 1.03, "adr": 74.2, "team": "FaZe", "country": "NO"},
            "Twistzz": {"age": 25, "rating": 1.18, "kd": 1.14, "adr": 78.5, "team": "FaZe", "country": "CA"},
            "broky": {"age": 22, "rating": 1.15, "kd": 1.12, "adr": 76.9, "team": "FaZe", "country": "LV"},
            "ropz": {"age": 24, "rating": 1.13, "kd": 1.09, "adr": 75.1, "team": "FaZe", "country": "EE"},
            
            # G2
            "m0NESY": {"age": 19, "rating": 1.24, "kd": 1.19, "adr": 80.3, "team": "G2", "country": "RU"},
            "NiKo": {"age": 27, "rating": 1.25, "kd": 1.18, "adr": 79.3, "team": "G2", "country": "BA"},
            "huNter-": {"age": 29, "rating": 1.09, "kd": 1.05, "adr": 73.7, "team": "G2", "country": "BA"},
            "jks": {"age": 28, "rating": 1.06, "kd": 1.01, "adr": 71.4, "team": "G2", "country": "AU"},
            "HooXi": {"age": 28, "rating": 0.92, "kd": 0.87, "adr": 62.8, "team": "G2", "country": "DK"},
            
            # Astralis
            "device": {"age": 29, "rating": 1.22, "kd": 1.15, "adr": 77.8, "team": "Astralis", "country": "DK"},
            "Magisk_old": {"age": 26, "rating": 1.08, "kd": 1.03, "adr": 72.5, "team": "Astralis", "country": "DK"},
            "Xyp9x": {"age": 29, "rating": 1.02, "kd": 0.98, "adr": 68.9, "team": "Astralis", "country": "DK"},
            "gla1ve": {"age": 29, "rating": 0.98, "kd": 0.94, "adr": 66.7, "team": "Astralis", "country": "DK"},
            "Staehr": {"age": 20, "rating": 1.05, "kd": 1.02, "adr": 70.1, "team": "Astralis", "country": "DK"},
            
            # Liquid
            "YEKINDAR": {"age": 24, "rating": 1.16, "kd": 1.11, "adr": 76.4, "team": "Liquid", "country": "LV"},
            "EliGE": {"age": 26, "rating": 1.09, "kd": 1.05, "adr": 74.2, "team": "Liquid", "country": "US"},
            "NAF": {"age": 27, "rating": 1.07, "kd": 1.03, "adr": 72.8, "team": "Liquid", "country": "CA"},
            "oSee": {"age": 23, "rating": 1.04, "kd": 1.00, "adr": 71.5, "team": "Liquid", "country": "US"},
            "nitr0": {"age": 29, "rating": 0.96, "kd": 0.92, "adr": 65.9, "team": "Liquid", "country": "US"},
        }
        
        # 真实的队伍数据
        self.real_teams_data = {
            "NAVI": {"ranking": 5, "country": "UA", "recent_winrate": 0.68},
            "Vitality": {"ranking": 3, "country": "FR", "recent_winrate": 0.72},
            "FaZe": {"ranking": 2, "country": "EU", "recent_winrate": 0.75},
            "G2": {"ranking": 4, "country": "EU", "recent_winrate": 0.69},
            "Astralis": {"ranking": 8, "country": "DK", "recent_winrate": 0.58},
            "Liquid": {"ranking": 6, "country": "US", "recent_winrate": 0.64},
        }
        
        # CS2地图池
        self.maps = ["mirage", "inferno", "dust2", "nuke", "overpass", "vertigo", "ancient"]
    
    def generate_player_data(self) -> pd.DataFrame:
        """生成选手数据"""
        players_data = []
        player_id = 1000
        
        for name, data in self.real_players_data.items():
            # 计算经验月数（基于年龄估算）
            experience_months = max(12, (data["age"] - 16) * 12 + random.randint(-24, 24))
            
            player_data = {
                'player_id': player_id,
                'name': name,
                'age': data["age"],
                'country': data["country"],
                'team_name': data["team"],
                'team_id': hash(data["team"]) % 10000,
                'rating_2_0': data["rating"],
                'kd_ratio': data["kd"],
                'adr': data["adr"],
                'kpr': round(data["rating"] * 0.65, 2),  # 估算KPR
                'spr': round(1 - (1/max(data["kd"], 0.5)), 2),  # 估算生存率
                'headshot_pct': round(40 + random.random() * 20, 1),  # 40-60%
                'experience_months': experience_months,
                'maps_played': random.randint(100, 500),
                'rounds_played': random.randint(2000, 10000),
            }
            
            players_data.append(player_data)
            player_id += 1
        
        return pd.DataFrame(players_data)
    
    def generate_team_data(self) -> pd.DataFrame:
        """生成队伍数据"""
        teams_data = []
        
        for team_name, data in self.real_teams_data.items():
            # 生成地图胜率（基于队伍实力）
            base_winrate = 0.4 + (10 - data["ranking"]) * 0.02  # 排名越高，基础胜率越高
            
            map_winrates = {}
            for map_name in self.maps:
                # 为每张地图添加一些随机性
                variation = random.uniform(-0.1, 0.1)
                winrate = max(0.2, min(0.8, base_winrate + variation))
                map_winrates[f'map_winrate_{map_name}'] = round(winrate, 3)
            
            team_data = {
                'team_id': hash(team_name) % 10000,
                'name': team_name,
                'country': data["country"],
                'world_ranking': data["ranking"],
                'recent_winrate': data["recent_winrate"],
                'total_maps_played': random.randint(50, 200),
                **map_winrates
            }
            
            teams_data.append(team_data)
        
        return pd.DataFrame(teams_data)
    
    def generate_match_data(self, num_matches: int = 200) -> pd.DataFrame:
        """生成历史比赛数据"""
        matches_data = []
        match_id = 1
        
        teams = list(self.real_teams_data.keys())
        
        for _ in range(num_matches):
            # 随机选择两支队伍
            team1, team2 = random.sample(teams, 2)
            team1_data = self.real_teams_data[team1]
            team2_data = self.real_teams_data[team2]
            
            # 基于排名计算胜率
            team1_strength = 10 - team1_data["ranking"]
            team2_strength = 10 - team2_data["ranking"]
            
            total_strength = team1_strength + team2_strength
            team1_win_prob = team1_strength / total_strength if total_strength > 0 else 0.5
            
            # 随机选择地图
            map_name = random.choice(self.maps)
            
            # 确定胜者
            team1_wins = random.random() < team1_win_prob
            winner_id = hash(team1) % 10000 if team1_wins else hash(team2) % 10000
            
            # 生成比分
            if team1_wins:
                team1_score = random.randint(16, 19)
                team2_score = random.randint(10, team1_score - 1)
            else:
                team2_score = random.randint(16, 19)
                team1_score = random.randint(10, team2_score - 1)
            
            match_data = {
                'match_id': match_id,
                'date': f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                'team1_id': hash(team1) % 10000,
                'team1_name': team1,
                'team2_id': hash(team2) % 10000,
                'team2_name': team2,
                'team1_score': team1_score,
                'team2_score': team2_score,
                'winner_id': winner_id,
                'map_name': map_name,
                'event_name': f"Tournament_{random.randint(1, 20)}",
                'format': 'bo1'
            }
            
            matches_data.append(match_data)
            match_id += 1
        
        return pd.DataFrame(matches_data)
    
    def generate_all_data(self):
        """生成所有模拟数据"""
        print("📊 生成模拟数据...")
        
        # 生成选手数据
        print("  生成选手数据...")
        players_df = self.generate_player_data()
        players_file = self.raw_data_dir / 'players_raw.csv'
        players_df.to_csv(players_file, index=False)
        print(f"    ✅ 保存了 {len(players_df)} 名选手数据到 {players_file}")
        
        # 生成队伍数据
        print("  生成队伍数据...")
        teams_df = self.generate_team_data()
        teams_file = self.raw_data_dir / 'teams_raw.csv'
        teams_df.to_csv(teams_file, index=False)
        print(f"    ✅ 保存了 {len(teams_df)} 支队伍数据到 {teams_file}")
        
        # 生成比赛数据
        print("  生成比赛数据...")
        matches_df = self.generate_match_data()
        matches_file = self.raw_data_dir / 'matches_raw.csv'
        matches_df.to_csv(matches_file, index=False)
        print(f"    ✅ 保存了 {len(matches_df)} 场比赛数据到 {matches_file}")
        
        # 生成收集报告
        report = {
            'collection_time': '2024-01-15T10:00:00',
            'data_source': 'mock_generator',
            'teams_collected': len(teams_df),
            'players_collected': len(players_df),
            'matches_collected': len(matches_df),
            'team_ids': teams_df['team_id'].tolist(),
            'maps_in_matches': matches_df['map_name'].value_counts().to_dict()
        }
        
        report_file = self.raw_data_dir / 'collection_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"    ✅ 生成数据收集报告: {report_file}")
        print("\n🎉 模拟数据生成完成!")
        
        return {
            'players': len(players_df),
            'teams': len(teams_df),
            'matches': len(matches_df)
        }
    
    def show_sample_data(self):
        """显示样本数据"""
        try:
            players_df = pd.read_csv(self.raw_data_dir / 'players_raw.csv')
            teams_df = pd.read_csv(self.raw_data_dir / 'teams_raw.csv')
            
            print("\n📋 样本选手数据:")
            print(players_df[['name', 'team_name', 'rating_2_0', 'age']].head())
            
            print("\n📋 样本队伍数据:")
            print(teams_df[['name', 'world_ranking', 'recent_winrate']].head())
            
        except FileNotFoundError:
            print("❌ 数据文件不存在，请先生成数据")

def main():
    """主函数"""
    generator = MockDataGenerator()
    
    print("🎯 CS2模拟数据生成器")
    print("=" * 30)
    print("由于外部API不可用，将生成真实的模拟数据用于演示")
    print()
    
    # 生成数据
    stats = generator.generate_all_data()
    
    # 显示样本
    generator.show_sample_data()
    
    print(f"\n📊 数据统计:")
    print(f"  选手: {stats['players']}")
    print(f"  队伍: {stats['teams']}")
    print(f"  比赛: {stats['matches']}")
    
    print("\n💡 现在你可以使用简化版预测器进行测试:")
    print("  python3 simple_main.py")

if __name__ == "__main__":
    main()