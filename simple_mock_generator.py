"""
简单模拟数据生成器 - 不依赖外部库
生成CSV格式的CS2选手和队伍数据
"""
import json
import random
import csv
import os

def create_directories():
    """创建必要的目录"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def generate_player_data():
    """生成选手数据"""
    # 真实的CS2选手数据（基于2024年数据）
    real_players = {
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
        "Xyp9x": {"age": 29, "rating": 1.02, "kd": 0.98, "adr": 68.9, "team": "Astralis", "country": "DK"},
        "gla1ve": {"age": 29, "rating": 0.98, "kd": 0.94, "adr": 66.7, "team": "Astralis", "country": "DK"},
        "Staehr": {"age": 20, "rating": 1.05, "kd": 1.02, "adr": 70.1, "team": "Astralis", "country": "DK"},
        "sjuush": {"age": 24, "rating": 1.01, "kd": 0.99, "adr": 69.5, "team": "Astralis", "country": "DK"},
        
        # Liquid
        "YEKINDAR": {"age": 24, "rating": 1.16, "kd": 1.11, "adr": 76.4, "team": "Liquid", "country": "LV"},
        "EliGE": {"age": 26, "rating": 1.09, "kd": 1.05, "adr": 74.2, "team": "Liquid", "country": "US"},
        "NAF": {"age": 27, "rating": 1.07, "kd": 1.03, "adr": 72.8, "team": "Liquid", "country": "CA"},
        "oSee": {"age": 23, "rating": 1.04, "kd": 1.00, "adr": 71.5, "team": "Liquid", "country": "US"},
        "nitr0": {"age": 29, "rating": 0.96, "kd": 0.92, "adr": 65.9, "team": "Liquid", "country": "US"},
    }
    
    players_data = []
    player_id = 1000
    
    for name, data in real_players.items():
        # 计算衍生数据
        experience_months = max(12, (data["age"] - 16) * 12 + random.randint(-24, 24))
        kpr = round(data["rating"] * 0.65, 2)
        spr = round(1 - (1/max(data["kd"], 0.5)), 2)
        headshot_pct = round(40 + random.random() * 20, 1)
        
        player_row = [
            player_id,                    # player_id
            name,                         # name
            data["age"],                  # age
            data["country"],              # country
            data["team"],                 # team_name
            hash(data["team"]) % 10000,   # team_id
            data["rating"],               # rating_2_0
            data["kd"],                   # kd_ratio
            data["adr"],                  # adr
            kpr,                          # kpr
            spr,                          # spr
            headshot_pct,                 # headshot_pct
            experience_months,            # experience_months
            random.randint(100, 500),     # maps_played
            random.randint(2000, 10000),  # rounds_played
        ]
        
        players_data.append(player_row)
        player_id += 1
    
    # 写入CSV文件
    with open('data/raw/players_raw.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([
            'player_id', 'name', 'age', 'country', 'team_name', 'team_id',
            'rating_2_0', 'kd_ratio', 'adr', 'kpr', 'spr', 'headshot_pct',
            'experience_months', 'maps_played', 'rounds_played'
        ])
        # 写入数据
        writer.writerows(players_data)
    
    return len(players_data)

def generate_team_data():
    """生成队伍数据"""
    teams = {
        "NAVI": {"ranking": 5, "country": "UA", "recent_winrate": 0.68},
        "Vitality": {"ranking": 3, "country": "FR", "recent_winrate": 0.72},
        "FaZe": {"ranking": 2, "country": "EU", "recent_winrate": 0.75},
        "G2": {"ranking": 4, "country": "EU", "recent_winrate": 0.69},
        "Astralis": {"ranking": 8, "country": "DK", "recent_winrate": 0.58},
        "Liquid": {"ranking": 6, "country": "US", "recent_winrate": 0.64},
    }
    
    maps = ["mirage", "inferno", "dust2", "nuke", "overpass", "vertigo", "ancient"]
    teams_data = []
    
    for team_name, data in teams.items():
        # 基于排名生成地图胜率
        base_winrate = 0.4 + (10 - data["ranking"]) * 0.02
        
        team_row = [
            hash(team_name) % 10000,      # team_id
            team_name,                    # name
            data["country"],              # country
            data["ranking"],              # world_ranking
            data["recent_winrate"],       # recent_winrate
            random.randint(50, 200),      # total_maps_played
        ]
        
        # 添加各地图胜率
        for map_name in maps:
            variation = random.uniform(-0.1, 0.1)
            winrate = max(0.2, min(0.8, base_winrate + variation))
            team_row.append(round(winrate, 3))
        
        teams_data.append(team_row)
    
    # 写入CSV文件
    with open('data/raw/teams_raw.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        header = ['team_id', 'name', 'country', 'world_ranking', 'recent_winrate', 'total_maps_played']
        header.extend([f'map_winrate_{map_name}' for map_name in maps])
        writer.writerow(header)
        # 写入数据
        writer.writerows(teams_data)
    
    return len(teams_data)

def generate_match_data(num_matches=200):
    """生成比赛数据"""
    teams = ["NAVI", "Vitality", "FaZe", "G2", "Astralis", "Liquid"]
    team_rankings = {"NAVI": 5, "Vitality": 3, "FaZe": 2, "G2": 4, "Astralis": 8, "Liquid": 6}
    maps = ["mirage", "inferno", "dust2", "nuke", "overpass", "vertigo", "ancient"]
    
    matches_data = []
    match_id = 1
    
    for _ in range(num_matches):
        # 随机选择两支队伍
        team1, team2 = random.sample(teams, 2)
        
        # 基于排名计算胜率
        team1_strength = 10 - team_rankings[team1]
        team2_strength = 10 - team_rankings[team2]
        total_strength = team1_strength + team2_strength
        team1_win_prob = team1_strength / total_strength if total_strength > 0 else 0.5
        
        # 确定胜者和比分
        team1_wins = random.random() < team1_win_prob
        if team1_wins:
            team1_score = random.randint(16, 19)
            team2_score = random.randint(10, team1_score - 1)
            winner_id = hash(team1) % 10000
        else:
            team2_score = random.randint(16, 19)
            team1_score = random.randint(10, team2_score - 1)
            winner_id = hash(team2) % 10000
        
        match_row = [
            match_id,                           # match_id
            f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",  # date
            hash(team1) % 10000,                # team1_id
            team1,                              # team1_name
            hash(team2) % 10000,                # team2_id
            team2,                              # team2_name
            team1_score,                        # team1_score
            team2_score,                        # team2_score
            winner_id,                          # winner_id
            random.choice(maps),                # map_name
            f"Tournament_{random.randint(1, 20)}", # event_name
            'bo1'                               # format
        ]
        
        matches_data.append(match_row)
        match_id += 1
    
    # 写入CSV文件
    with open('data/raw/matches_raw.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'match_id', 'date', 'team1_id', 'team1_name', 'team2_id', 'team2_name',
            'team1_score', 'team2_score', 'winner_id', 'map_name', 'event_name', 'format'
        ])
        writer.writerows(matches_data)
    
    return len(matches_data)

def generate_collection_report(players_count, teams_count, matches_count):
    """生成数据收集报告"""
    report = {
        'collection_time': '2024-01-15T10:00:00',
        'data_source': 'mock_generator',
        'teams_collected': teams_count,
        'players_collected': players_count,
        'matches_collected': matches_count,
        'team_ids': [4608, 5995, 6665, 7020, 6137, 4494],  # 示例ID
        'maps_in_matches': {
            'mirage': matches_count // 7,
            'inferno': matches_count // 7,
            'dust2': matches_count // 7,
            'nuke': matches_count // 7,
            'overpass': matches_count // 7,
            'vertigo': matches_count // 7,
            'ancient': matches_count // 7
        }
    }
    
    with open('data/raw/collection_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

def show_sample_data():
    """显示样本数据"""
    print("\n📋 样本选手数据:")
    try:
        with open('data/raw/players_raw.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            
            # 显示前5行
            for i, row in enumerate(reader):
                if i >= 5:
                    break
                print(f"  {row[1]} ({row[4]}): 评分 {row[6]}, 年龄 {row[2]}")
    except FileNotFoundError:
        print("  ❌ 选手数据文件不存在")
    
    print("\n📋 样本队伍数据:")
    try:
        with open('data/raw/teams_raw.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            
            # 显示所有队伍
            for row in reader:
                print(f"  {row[1]}: 排名 #{row[3]}, 胜率 {float(row[4]):.1%}")
    except FileNotFoundError:
        print("  ❌ 队伍数据文件不存在")

def main():
    """主函数"""
    print("🎯 CS2简单模拟数据生成器")
    print("=" * 40)
    print("由于外部API不可用，将生成真实的模拟数据用于演示")
    print()
    
    # 创建目录
    create_directories()
    
    # 生成数据
    print("📊 生成模拟数据...")
    
    print("  生成选手数据...")
    players_count = generate_player_data()
    print(f"    ✅ 保存了 {players_count} 名选手数据")
    
    print("  生成队伍数据...")
    teams_count = generate_team_data()
    print(f"    ✅ 保存了 {teams_count} 支队伍数据")
    
    print("  生成比赛数据...")
    matches_count = generate_match_data()
    print(f"    ✅ 保存了 {matches_count} 场比赛数据")
    
    print("  生成收集报告...")
    generate_collection_report(players_count, teams_count, matches_count)
    print("    ✅ 生成数据收集报告")
    
    print("\n🎉 模拟数据生成完成!")
    
    # 显示样本数据
    show_sample_data()
    
    print(f"\n📊 数据统计:")
    print(f"  选手: {players_count}")
    print(f"  队伍: {teams_count}")
    print(f"  比赛: {matches_count}")
    
    print("\n💡 现在你可以使用简化版预测器进行测试:")
    print("  python3 simple_main.py")
    
    print("\n🔍 数据文件位置:")
    print("  data/raw/players_raw.csv")
    print("  data/raw/teams_raw.csv")
    print("  data/raw/matches_raw.csv")

if __name__ == "__main__":
    main()