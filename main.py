#!/usr/bin/env python3
"""
CS2比赛预测系统主程序
提供命令行接口进行数据收集、模型训练和预测
"""
import argparse
import sys
import os
import logging
from typing import List, Dict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collector import HLTVDataCollector
from data_preprocessor import DataPreprocessor
from trainer import CS2ModelTrainer
from predictor import CS2MatchPredictor_Inference, PlayerInfo, TeamInfo, create_sample_teams

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cs2_predictor.log')
        ]
    )

def collect_data(team_ids: List[int] = None, player_ids: List[int] = None):
    """收集数据"""
    print("🔄 开始收集HLTV数据...")
    
    collector = HLTVDataCollector()
    
    # 默认队伍ID（示例）
    if team_ids is None:
        team_ids = [4608, 5995, 6665, 7020, 6137]  # G2, FaZe, NAVI, Astralis, Liquid
    
    # 收集队伍数据
    print(f"收集 {len(team_ids)} 支队伍的数据...")
    teams_df = collector.collect_team_data(team_ids)
    
    # 如果提供了选手ID，收集选手数据
    if player_ids:
        print(f"收集 {len(player_ids)} 名选手的数据...")
        players_df = collector.collect_player_data(player_ids)
    else:
        print("未提供选手ID，跳过选手数据收集")
        players_df = None
    
    print("✅ 数据收集完成")
    return teams_df, players_df

def preprocess_data():
    """预处理数据"""
    print("🔄 开始预处理数据...")
    
    preprocessor = DataPreprocessor()
    
    try:
        # 加载原始数据
        import pandas as pd
        from config import RAW_DATA_DIR
        
        players_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'players_raw.csv'))
        teams_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'teams_raw.csv'))
        
        # 预处理
        players_processed = preprocessor.preprocess_player_data(players_df)
        teams_processed = preprocessor.preprocess_team_data(teams_df)
        
        # 保存
        preprocessor.save_preprocessed_data(players_processed, teams_processed)
        
        print("✅ 数据预处理完成")
        
    except FileNotFoundError:
        print("❌ 未找到原始数据文件，请先运行数据收集")
        return False
    
    return True

def train_model(epochs: int = 100, use_ensemble: bool = False):
    """训练模型"""
    print("🔄 开始训练模型...")
    print(f"训练轮数: {epochs}")
    print(f"使用集成模型: {use_ensemble}")
    
    trainer = CS2ModelTrainer(use_ensemble=use_ensemble)
    
    try:
        results = trainer.train(epochs=epochs)
        
        print("✅ 模型训练完成")
        print(f"最终测试准确率: {results['test_accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return False

def predict_match():
    """预测比赛"""
    print("🔄 开始预测比赛...")
    
    try:
        # 创建预测器
        predictor = CS2MatchPredictor_Inference()
        
        # 创建示例队伍
        team1, team2 = create_sample_teams()
        
        print(f"预测比赛: {team1.name} vs {team2.name}")
        
        # 进行预测
        prediction = predictor.predict_match(team1, team2)
        
        # 打印报告
        predictor.print_prediction_report(prediction)
        
        print("✅ 比赛预测完成")
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        print("请确保已经训练并保存了模型")

def interactive_predict():
    """交互式预测"""
    print("🎮 交互式比赛预测")
    print("=" * 50)
    
    try:
        predictor = CS2MatchPredictor_Inference()
        
        while True:
            print("\n选择操作:")
            print("1. 使用示例队伍预测")
            print("2. 自定义队伍预测")
            print("3. 退出")
            
            choice = input("\n请输入选择 (1-3): ").strip()
            
            if choice == '1':
                team1, team2 = create_sample_teams()
                prediction = predictor.predict_match(team1, team2)
                predictor.print_prediction_report(prediction)
                
            elif choice == '2':
                print("自定义队伍功能开发中...")
                # TODO: 实现自定义队伍输入
                
            elif choice == '3':
                print("退出程序")
                break
                
            else:
                print("无效选择，请重新输入")
    
    except Exception as e:
        print(f"❌ 交互式预测失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CS2比赛预测系统")
    
    parser.add_argument('--mode', choices=['collect', 'preprocess', 'train', 'predict', 'interactive'],
                       required=True, help='运行模式')
    
    # 数据收集参数
    parser.add_argument('--team-ids', nargs='+', type=int,
                       help='要收集的队伍ID列表')
    parser.add_argument('--player-ids', nargs='+', type=int,
                       help='要收集的选手ID列表')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数 (默认: 100)')
    parser.add_argument('--ensemble', action='store_true',
                       help='使用集成模型')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    print("🎯 CS2比赛预测系统")
    print("=" * 50)
    
    if args.mode == 'collect':
        collect_data(args.team_ids, args.player_ids)
        
    elif args.mode == 'preprocess':
        preprocess_data()
        
    elif args.mode == 'train':
        train_model(args.epochs, args.ensemble)
        
    elif args.mode == 'predict':
        predict_match()
        
    elif args.mode == 'interactive':
        interactive_predict()

if __name__ == "__main__":
    main()