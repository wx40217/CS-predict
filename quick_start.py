#!/usr/bin/env python3
"""
CS比赛预测系统快速启动脚本
演示如何使用整个系统进行预测
"""

import asyncio
import json
from pathlib import Path
from models.predictor import CSPredictor

# 示例比赛数据
EXAMPLE_MATCHES = [
    {
        "team1": ["ZywOo", "shox", "apEX", "misutaaa", "Kyojin"],
        "team2": ["s1mple", "electronic", "Boombl4", "Perfecto", "flamie"]
    },
    {
        "team1": ["ropz", "broky", "Twistzz", "karrigan", "rain"],
        "team2": ["sh1ro", "Ax1Le", "interz", "HObbit", "nafany"]
    },
    {
        "team1": ["device", "dupreeh", "Xyp9x", "gla1ve", "Magisk"],
        "team2": ["EliGE", "NAF", "Stewie2K", "nitr0", "Twistzz"]
    }
]


async def collect_data_demo():
    """数据收集演示"""
    print("=== 数据收集演示 ===")
    print("注意：这需要网络连接和HLTV API访问")
    
    try:
        from data_collection.hltv_collector import HLTVCollector
        
        # 创建收集器
        collector = HLTVCollector()
        
        # 收集数据（限制时间范围以节省时间）
        print("开始收集HLTV数据...")
        await collector.collect_all_data("2024-01-01", "2024-12-31")
        
        # 显示数据摘要
        summary = collector.get_data_summary()
        print("数据收集摘要:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        
    except ImportError:
        print("数据收集模块未安装，跳过演示")
    except Exception as e:
        print(f"数据收集失败: {e}")


def preprocess_data_demo():
    """数据预处理演示"""
    print("\n=== 数据预处理演示 ===")
    
    try:
        from data_preprocessing.feature_engineer import CSFeatureEngineer
        
        # 创建特征工程师
        engineer = CSFeatureEngineer()
        
        # 处理数据
        print("开始数据预处理...")
        engineer.process_all_data()
        
        print("数据预处理完成")
        
    except ImportError:
        print("数据预处理模块未安装，跳过演示")
    except Exception as e:
        print(f"数据预处理失败: {e}")


def train_models_demo():
    """模型训练演示"""
    print("\n=== 模型训练演示 ===")
    print("注意：这需要GPU和大量时间")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
            print("可以开始模型训练")
            
            # 这里可以启动训练
            # 但由于需要大量数据和训练时间，暂时跳过
            print("训练过程需要大量数据和训练时间，建议在准备好数据后单独运行")
            
        else:
            print("未检测到GPU，建议使用GPU进行训练")
            
    except ImportError:
        print("PyTorch未安装，无法进行模型训练")


def prediction_demo():
    """预测演示"""
    print("\n=== 预测演示 ===")
    
    try:
        # 创建预测器（使用规则基础预测）
        predictor = CSPredictor(
            map_model_path="./checkpoints/map_model_best.pth",
            winrate_model_path="./checkpoints/winrate_model_best.pth"
        )
        
        # 单场比赛预测
        print("单场比赛预测:")
        team1 = ["ZywOo", "shox", "apEX", "misutaaa", "Kyojin"]
        team2 = ["s1mple", "electronic", "Boombl4", "Perfecto", "flamie"]
        
        result = predictor.predict(team1, team2)
        
        print(f"Team1: {team1}")
        print(f"Team2: {team2}")
        print(f"推荐选图: {result['recommended_maps'][:3]}")  # 显示前3张地图
        print(f"胜率预测: Team1 {result['winrate_team1']:.2%}, Team2 {result['winrate_team2']:.2%}")
        print(f"预测置信度: {result['confidence']:.2%}")
        
        # 年龄分析
        age_analysis = result['age_analysis']
        print(f"年龄分析:")
        print(f"  Team1平均年龄: {age_analysis['team1_avg_age']:.1f}")
        print(f"  Team2平均年龄: {age_analysis['team2_avg_age']:.1f}")
        print(f"  年龄优势: {age_analysis['age_advantage']}")
        print(f"  年龄因素影响: {age_analysis['age_factor_impact']:.2%}")
        
        # 批量预测
        print("\n批量预测演示:")
        batch_results = predictor.batch_predict(EXAMPLE_MATCHES)
        
        for i, match_result in enumerate(batch_results):
            if 'error' not in match_result:
                pred = match_result['prediction']
                print(f"比赛 {i+1}: Team1胜率 {pred['winrate_team1']:.2%}, Team2胜率 {pred['winrate_team2']:.2%}")
            else:
                print(f"比赛 {i+1}: 预测失败 - {match_result['error']}")
        
        # 保存预测结果
        output_path = "./predictions_demo.json"
        predictor.save_predictions(batch_results, output_path)
        print(f"\n预测结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"预测演示失败: {e}")


def system_info():
    """显示系统信息"""
    print("=== 系统信息 ===")
    
    # 检查Python版本
    import sys
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch未安装")
    
    # 检查其他依赖
    dependencies = ['numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn']
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"{dep}: {version}")
        except ImportError:
            print(f"{dep}: 未安装")
    
    # 检查目录结构
    print("\n目录结构:")
    directories = ['configs', 'models', 'data_collection', 'data_preprocessing', 'checkpoints', 'logs']
    for dir_name in directories:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  {dir_name}: ✓")
        else:
            print(f"  {dir_name}: ✗")


def main():
    """主函数"""
    print("CS比赛预测系统快速启动")
    print("=" * 50)
    
    # 显示系统信息
    system_info()
    
    print("\n" + "=" * 50)
    print("开始演示各个功能模块")
    
    # 数据收集演示
    asyncio.run(collect_data_demo())
    
    # 数据预处理演示
    preprocess_data_demo()
    
    # 模型训练演示
    train_models_demo()
    
    # 预测演示
    prediction_demo()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("\n下一步:")
    print("1. 收集HLTV数据: python data_collection/hltv_collector.py")
    print("2. 预处理数据: python data_preprocessing/feature_engineer.py")
    print("3. 训练选图模型: python train_map_model.py --gpu 0")
    print("4. 训练胜率模型: python train_winrate_model.py --gpu 0")
    print("5. 使用预测器: python models/predictor.py")


if __name__ == "__main__":
    main()