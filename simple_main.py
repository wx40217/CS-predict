#!/usr/bin/env python3
"""
简化版主程序 - CS2比赛预测
只需要输入选手名字，系统自动处理其他所有事情
"""
import argparse
import sys
import os
import asyncio
from typing import List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_predictor import SimpleCS2Predictor, quick_predict
from player_database import get_player_database, search_players, search_teams

def interactive_player_selection() -> List[str]:
    """交互式选手选择"""
    players = []
    db = get_player_database()
    
    print("请输入5名选手的名字（支持模糊匹配）:")
    
    for i in range(5):
        while True:
            name = input(f"选手 {i+1}: ").strip()
            
            if not name:
                print("请输入选手名字")
                continue
            
            # 搜索选手
            matches = search_players(name, limit=5)
            
            if not matches:
                print(f"⚠️  未找到选手 '{name}'，将使用默认数据")
                players.append(name)
                break
            elif len(matches) == 1:
                # 只有一个匹配结果
                player = matches[0]
                print(f"✅ 找到: {player['name']} (评分: {player['rating']:.2f})")
                players.append(player['name'])
                break
            else:
                # 多个匹配结果，让用户选择
                print(f"找到多个匹配结果:")
                for j, player in enumerate(matches):
                    team_info = f" ({player['team']})" if player['team'] else ""
                    print(f"  {j+1}. {player['name']}{team_info} - 评分: {player['rating']:.2f}")
                
                try:
                    choice = int(input("请选择 (输入数字): ")) - 1
                    if 0 <= choice < len(matches):
                        selected = matches[choice]
                        print(f"✅ 选择: {selected['name']}")
                        players.append(selected['name'])
                        break
                    else:
                        print("无效选择，请重新输入")
                except ValueError:
                    print("请输入有效数字")
    
    return players

def interactive_prediction():
    """交互式预测"""
    print("🎮 CS2比赛预测 - 简化版")
    print("=" * 50)
    print("只需要输入选手名字，系统会自动获取统计数据！")
    print()
    
    # 选择队伍1
    print("📋 队伍1 选手选择:")
    team1_players = interactive_player_selection()
    team1_name = input("\n队伍1名字 (可选): ").strip() or None
    
    print("\n" + "-" * 30)
    
    # 选择队伍2
    print("📋 队伍2 选手选择:")
    team2_players = interactive_player_selection()
    team2_name = input("\n队伍2名字 (可选): ").strip() or None
    
    print("\n🤖 开始预测...")
    
    # 进行预测
    predictor = SimpleCS2Predictor()
    result = predictor.predict_match(
        team1_players=team1_players,
        team2_players=team2_players,
        team1_name=team1_name,
        team2_name=team2_name
    )
    
    # 显示结果
    predictor.print_simple_report(result)

def quick_prediction_mode():
    """快速预测模式 - 使用预设队伍"""
    print("🚀 快速预测模式")
    print("=" * 30)
    
    # 一些知名选手组合
    famous_lineups = {
        "1": {
            "name": "NAVI 2021",
            "players": ["s1mple", "electroNic", "Perfecto", "b1t", "Boombl4"]
        },
        "2": {
            "name": "Vitality 2023", 
            "players": ["ZywOo", "apEX", "dupreeh", "Magisk", "Spinx"]
        },
        "3": {
            "name": "FaZe 2022",
            "players": ["karrigan", "rain", "Twistzz", "broky", "ropz"]
        },
        "4": {
            "name": "G2 2023",
            "players": ["m0NESY", "NiKo", "huNter-", "jks", "HooXi"]
        }
    }
    
    print("选择预设阵容:")
    for key, team in famous_lineups.items():
        players_str = ", ".join(team["players"])
        print(f"  {key}. {team['name']}: {players_str}")
    
    print("  5. 自定义输入")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice in famous_lineups:
        team1 = famous_lineups[choice]
        print(f"\n选择了 {team1['name']}")
        
        # 选择对手
        print("\n选择对手:")
        remaining = {k: v for k, v in famous_lineups.items() if k != choice}
        for key, team in remaining.items():
            players_str = ", ".join(team["players"])
            print(f"  {key}. {team['name']}: {players_str}")
        
        opponent_choice = input("请选择对手: ").strip()
        
        if opponent_choice in remaining:
            team2 = remaining[opponent_choice]
            
            print(f"\n🎯 预测: {team1['name']} vs {team2['name']}")
            
            # 进行预测
            result = quick_predict(
                team1_players=team1['players'],
                team2_players=team2['players'],
                team1_name=team1['name'],
                team2_name=team2['name']
            )
            
            # 显示结果
            predictor = SimpleCS2Predictor()
            predictor.print_simple_report(result)
        else:
            print("无效选择")
    elif choice == "5":
        interactive_prediction()
    else:
        print("无效选择")

def search_mode():
    """搜索模式"""
    print("🔍 选手/队伍搜索")
    print("=" * 30)
    
    while True:
        print("\n选择搜索类型:")
        print("1. 搜索选手")
        print("2. 搜索队伍")
        print("3. 返回主菜单")
        
        choice = input("请选择 (1-3): ").strip()
        
        if choice == "1":
            query = input("输入选手名字 (支持部分匹配): ").strip()
            if query:
                matches = search_players(query, limit=10)
                if matches:
                    print(f"\n找到 {len(matches)} 名选手:")
                    for i, player in enumerate(matches, 1):
                        team_info = f" ({player['team']})" if player['team'] else ""
                        print(f"  {i:2d}. {player['name']}{team_info} - 评分: {player['rating']:.2f}")
                else:
                    print("未找到匹配的选手")
        
        elif choice == "2":
            query = input("输入队伍名字 (支持部分匹配): ").strip()
            if query:
                matches = search_teams(query, limit=10)
                if matches:
                    print(f"\n找到 {len(matches)} 支队伍:")
                    for i, team in enumerate(matches, 1):
                        country_info = f" ({team['country']})" if team['country'] else ""
                        ranking = team['ranking'] if team['ranking'] < 999 else "未排名"
                        print(f"  {i:2d}. {team['name']}{country_info} - 排名: {ranking}")
                else:
                    print("未找到匹配的队伍")
        
        elif choice == "3":
            break
        else:
            print("无效选择")

async def update_database():
    """更新数据库"""
    print("🔄 更新选手数据库...")
    print("这可能需要几分钟时间，请耐心等待...")
    
    db = get_player_database()
    await db.update_database(team_limit=25)
    
    print("✅ 数据库更新完成!")
    
    # 显示统计信息
    stats = db.get_statistics()
    print(f"\n📊 数据库统计:")
    print(f"  选手总数: {stats['total_players']}")
    print(f"  队伍总数: {stats['total_teams']}")
    
    if stats['top_rated_players']:
        print(f"\n🌟 评分最高选手:")
        for i, (name, rating) in enumerate(stats['top_rated_players'][:5], 1):
            print(f"  {i}. {name}: {rating:.2f}")

def show_database_stats():
    """显示数据库统计信息"""
    db = get_player_database()
    stats = db.get_statistics()
    
    print("📊 数据库统计信息")
    print("=" * 30)
    print(f"选手总数: {stats['total_players']}")
    print(f"队伍总数: {stats['total_teams']}")
    
    if stats['top_rated_players']:
        print(f"\n🌟 评分最高选手 (前10名):")
        for i, (name, rating) in enumerate(stats['top_rated_players'][:10], 1):
            print(f"  {i:2d}. {name}: {rating:.2f}")
    
    if stats['top_teams']:
        print(f"\n🏆 世界排名前10:")
        for i, (name, ranking) in enumerate(stats['top_teams'][:10], 1):
            ranking_str = str(ranking) if ranking < 999 else "未排名"
            print(f"  {i:2d}. {name}: #{ranking_str}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CS2比赛预测 - 简化版")
    
    parser.add_argument('--team1', nargs=5, metavar='PLAYER',
                       help='队伍1的5名选手')
    parser.add_argument('--team2', nargs=5, metavar='PLAYER', 
                       help='队伍2的5名选手')
    parser.add_argument('--team1-name', help='队伍1名字')
    parser.add_argument('--team2-name', help='队伍2名字')
    parser.add_argument('--update-db', action='store_true',
                       help='更新选手数据库')
    
    args = parser.parse_args()
    
    # 如果指定了更新数据库
    if args.update_db:
        asyncio.run(update_database())
        return
    
    # 如果提供了命令行参数
    if args.team1 and args.team2:
        print("🎯 命令行预测模式")
        print("=" * 30)
        
        result = quick_predict(
            team1_players=args.team1,
            team2_players=args.team2,
            team1_name=args.team1_name,
            team2_name=args.team2_name
        )
        
        predictor = SimpleCS2Predictor()
        predictor.print_simple_report(result)
        return
    
    # 交互式菜单
    while True:
        print("\n🎯 CS2比赛预测系统 - 简化版")
        print("=" * 40)
        print("1. 🎮 交互式预测 (自定义选手)")
        print("2. 🚀 快速预测 (预设阵容)")
        print("3. 🔍 搜索选手/队伍")
        print("4. 📊 数据库统计")
        print("5. 🔄 更新数据库")
        print("6. ❌ 退出")
        
        choice = input("\n请选择 (1-6): ").strip()
        
        if choice == "1":
            interactive_prediction()
        elif choice == "2":
            quick_prediction_mode()
        elif choice == "3":
            search_mode()
        elif choice == "4":
            show_database_stats()
        elif choice == "5":
            asyncio.run(update_database())
        elif choice == "6":
            print("👋 再见!")
            break
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，再见!")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("请检查环境配置或联系开发者")