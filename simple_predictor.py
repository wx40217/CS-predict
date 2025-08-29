"""
简化版预测器 - 只需要输入选手名字
自动从数据库获取统计信息并进行预测
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

from player_database import get_player_database, find_player, find_team
from predictor import CS2MatchPredictor_Inference

@dataclass
class SimpleMatchResult:
    """简化的比赛预测结果"""
    team1_name: str
    team2_name: str
    team1_players: List[str]
    team2_players: List[str]
    
    # 预测结果
    map_predictions: Dict[str, float]  # 地图 -> team1胜率
    recommended_pick_team1: str        # team1推荐选图
    recommended_pick_team2: str        # team2推荐选图
    recommended_ban_team1: str         # team1推荐禁图
    recommended_ban_team2: str         # team2推荐禁图
    
    # 简化的实力对比
    overall_advantage: str             # "team1" | "team2" | "even"
    confidence: float                  # 预测置信度 0-1

class SimpleCS2Predictor:
    """简化版CS2比赛预测器"""
    
    def __init__(self):
        """初始化预测器"""
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 获取数据库
        self.player_db = get_player_database()
        
        # 尝试加载训练好的模型
        try:
            self.predictor = CS2MatchPredictor_Inference()
            self.model_available = True
            self.logger.info("✅ 加载训练好的模型")
        except Exception as e:
            self.logger.warning(f"⚠️  未找到训练好的模型，将使用简化算法: {e}")
            self.model_available = False
    
    def predict_match(self, 
                     team1_players: List[str], 
                     team2_players: List[str],
                     team1_name: str = None,
                     team2_name: str = None) -> SimpleMatchResult:
        """
        预测比赛结果
        
        Args:
            team1_players: 队伍1选手名字列表 (5名选手)
            team2_players: 队伍2选手名字列表 (5名选手)
            team1_name: 队伍1名字 (可选)
            team2_name: 队伍2名字 (可选)
            
        Returns:
            简化的预测结果
        """
        # 验证输入
        if len(team1_players) != 5 or len(team2_players) != 5:
            raise ValueError("每支队伍必须有5名选手")
        
        # 设置队伍名字
        if not team1_name:
            team1_name = f"Team1({team1_players[0]}等)"
        if not team2_name:
            team2_name = f"Team2({team2_players[0]}等)"
        
        self.logger.info(f"🎯 预测比赛: {team1_name} vs {team2_name}")
        
        # 获取选手数据
        team1_data = [self.player_db.get_player_data(name) for name in team1_players]
        team2_data = [self.player_db.get_player_data(name) for name in team2_players]
        
        # 显示找到的选手信息
        self._log_player_info(team1_name, team1_data)
        self._log_player_info(team2_name, team2_data)
        
        if self.model_available:
            # 使用训练好的模型
            map_predictions = self._predict_with_model(team1_data, team2_data, team1_name, team2_name)
        else:
            # 使用简化算法
            map_predictions = self._predict_with_simple_algorithm(team1_data, team2_data)
        
        # 分析结果
        result = self._analyze_predictions(
            team1_name, team2_name, 
            team1_players, team2_players,
            map_predictions, team1_data, team2_data
        )
        
        return result
    
    def _log_player_info(self, team_name: str, players_data: List[Dict]):
        """记录选手信息"""
        self.logger.info(f"📋 {team_name} 选手:")
        for player in players_data:
            rating = player['rating_2_0']
            status = "⭐" if rating > 1.2 else "✅" if rating > 1.0 else "📈"
            self.logger.info(f"  {status} {player['name']}: 评分 {rating:.2f}")
    
    def _predict_with_model(self, team1_data: List[Dict], team2_data: List[Dict], 
                           team1_name: str, team2_name: str) -> Dict[str, float]:
        """使用训练好的模型进行预测"""
        try:
            # 转换为模型需要的格式
            from predictor import PlayerInfo, TeamInfo
            
            # 创建PlayerInfo对象
            team1_players = [
                PlayerInfo(
                    p['name'], p['age'], p['rating_2_0'], p['kd_ratio'],
                    p['adr'], p['kpr'], p['headshot_pct'], p['experience_months']
                ) for p in team1_data
            ]
            
            team2_players = [
                PlayerInfo(
                    p['name'], p['age'], p['rating_2_0'], p['kd_ratio'],
                    p['adr'], p['kpr'], p['headshot_pct'], p['experience_months']
                ) for p in team2_data
            ]
            
            # 获取队伍数据
            team1_info = self.player_db.get_team_data(team1_name)
            team2_info = self.player_db.get_team_data(team2_name)
            
            # 创建TeamInfo对象
            team1_obj = TeamInfo(
                name=team1_name,
                players=team1_players,
                world_ranking=team1_info['world_ranking'],
                recent_winrate=team1_info['recent_winrate'],
                map_winrates={
                    'mirage': team1_info['map_winrate_mirage'],
                    'inferno': team1_info['map_winrate_inferno'],
                    'dust2': team1_info['map_winrate_dust2'],
                    'nuke': team1_info['map_winrate_nuke'],
                    'overpass': team1_info['map_winrate_overpass'],
                    'vertigo': team1_info['map_winrate_vertigo'],
                    'ancient': team1_info['map_winrate_ancient'],
                }
            )
            
            team2_obj = TeamInfo(
                name=team2_name,
                players=team2_players,
                world_ranking=team2_info['world_ranking'],
                recent_winrate=team2_info['recent_winrate'],
                map_winrates={
                    'mirage': team2_info['map_winrate_mirage'],
                    'inferno': team2_info['map_winrate_inferno'],
                    'dust2': team2_info['map_winrate_dust2'],
                    'nuke': team2_info['map_winrate_nuke'],
                    'overpass': team2_info['map_winrate_overpass'],
                    'vertigo': team2_info['map_winrate_vertigo'],
                    'ancient': team2_info['map_winrate_ancient'],
                }
            )
            
            # 使用模型预测
            prediction = self.predictor.predict_match(team1_obj, team2_obj)
            return prediction.map_predictions
            
        except Exception as e:
            self.logger.warning(f"模型预测失败，使用简化算法: {e}")
            return self._predict_with_simple_algorithm(team1_data, team2_data)
    
    def _predict_with_simple_algorithm(self, team1_data: List[Dict], team2_data: List[Dict]) -> Dict[str, float]:
        """使用简化算法进行预测"""
        self.logger.info("🤖 使用简化算法进行预测")
        
        # 计算队伍平均实力
        team1_avg_rating = np.mean([p['rating_2_0'] for p in team1_data])
        team2_avg_rating = np.mean([p['rating_2_0'] for p in team2_data])
        
        team1_avg_kd = np.mean([p['kd_ratio'] for p in team1_data])
        team2_avg_kd = np.mean([p['kd_ratio'] for p in team2_data])
        
        team1_avg_adr = np.mean([p['adr'] for p in team1_data])
        team2_avg_adr = np.mean([p['adr'] for p in team2_data])
        
        # 综合实力评分
        team1_strength = (team1_avg_rating * 0.5 + team1_avg_kd * 0.3 + team1_avg_adr/100 * 0.2)
        team2_strength = (team2_avg_rating * 0.5 + team2_avg_kd * 0.3 + team2_avg_adr/100 * 0.2)
        
        # 基础胜率
        total_strength = team1_strength + team2_strength
        base_winrate = team1_strength / total_strength if total_strength > 0 else 0.5
        
        # 为不同地图添加一些随机性和特色
        map_modifiers = {
            'mirage': 0.0,      # 平衡地图
            'inferno': 0.02,    # 略偏向技术好的队伍
            'dust2': -0.01,     # 略偏向aim好的队伍
            'nuke': 0.03,       # 偏向战术队伍
            'overpass': 0.01,   # 略偏向配合好的队伍
            'vertigo': -0.02,   # 新地图，经验影响较小
            'ancient': 0.01,    # 需要适应的地图
        }
        
        map_predictions = {}
        for map_name, modifier in map_modifiers.items():
            # 添加地图修正和一些随机性
            winrate = base_winrate + modifier + np.random.normal(0, 0.05)
            winrate = np.clip(winrate, 0.1, 0.9)  # 限制在10%-90%之间
            map_predictions[map_name] = winrate
        
        return map_predictions
    
    def _analyze_predictions(self, team1_name: str, team2_name: str,
                           team1_players: List[str], team2_players: List[str],
                           map_predictions: Dict[str, float],
                           team1_data: List[Dict], team2_data: List[Dict]) -> SimpleMatchResult:
        """分析预测结果"""
        
        # 按胜率排序地图
        sorted_maps = sorted(map_predictions.items(), key=lambda x: x[1], reverse=True)
        
        # 推荐选图和禁图
        recommended_pick_team1 = sorted_maps[0][0]     # team1胜率最高的地图
        recommended_pick_team2 = sorted_maps[-1][0]    # team1胜率最低的地图（team2优势）
        recommended_ban_team1 = sorted_maps[-1][0]     # team1应该禁用胜率最低的地图
        recommended_ban_team2 = sorted_maps[0][0]      # team2应该禁用team1优势最大的地图
        
        # 计算整体优势
        avg_winrate = np.mean(list(map_predictions.values()))
        if avg_winrate > 0.6:
            overall_advantage = "team1"
        elif avg_winrate < 0.4:
            overall_advantage = "team2"
        else:
            overall_advantage = "even"
        
        # 计算置信度（基于预测的一致性）
        winrate_variance = np.var(list(map_predictions.values()))
        confidence = max(0.5, 1.0 - winrate_variance * 10)  # 方差越小，置信度越高
        
        return SimpleMatchResult(
            team1_name=team1_name,
            team2_name=team2_name,
            team1_players=team1_players,
            team2_players=team2_players,
            map_predictions=map_predictions,
            recommended_pick_team1=recommended_pick_team1,
            recommended_pick_team2=recommended_pick_team2,
            recommended_ban_team1=recommended_ban_team1,
            recommended_ban_team2=recommended_ban_team2,
            overall_advantage=overall_advantage,
            confidence=confidence
        )
    
    def print_simple_report(self, result: SimpleMatchResult):
        """打印简化的预测报告"""
        print(f"\n🎯 CS2比赛预测")
        print("=" * 60)
        print(f"{result.team1_name} vs {result.team2_name}")
        print("=" * 60)
        
        print(f"\n👥 选手阵容:")
        print(f"{result.team1_name}: {', '.join(result.team1_players)}")
        print(f"{result.team2_name}: {', '.join(result.team2_players)}")
        
        print(f"\n🗺️  各地图预测 ({result.team1_name}获胜概率):")
        print("-" * 40)
        
        # 按胜率排序显示
        sorted_maps = sorted(result.map_predictions.items(), key=lambda x: x[1], reverse=True)
        
        for map_name, winrate in sorted_maps:
            percentage = winrate * 100
            
            # 生成可视化条形图
            bar_length = int(winrate * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            # 添加优势指示
            if winrate > 0.6:
                indicator = f"🔥 {result.team1_name}优势"
            elif winrate < 0.4:
                indicator = f"🔥 {result.team2_name}优势"
            else:
                indicator = "⚖️  势均力敌"
            
            print(f"{map_name:10} │ {bar} │ {percentage:5.1f}% │ {indicator}")
        
        print(f"\n🎯 选图建议:")
        print("-" * 30)
        print(f"📍 {result.team1_name} 应该选择: {result.recommended_pick_team1}")
        print(f"📍 {result.team2_name} 应该选择: {result.recommended_pick_team2}")
        print(f"❌ {result.team1_name} 应该禁用: {result.recommended_ban_team1}")
        print(f"❌ {result.team2_name} 应该禁用: {result.recommended_ban_team2}")
        
        print(f"\n⚔️  整体预测:")
        print("-" * 20)
        
        if result.overall_advantage == "team1":
            print(f"🏆 {result.team1_name} 整体优势")
        elif result.overall_advantage == "team2":
            print(f"🏆 {result.team2_name} 整体优势")
        else:
            print("⚖️  双方实力接近")
        
        print(f"🎲 预测置信度: {result.confidence:.1%}")
        
        # 添加一些实用建议
        print(f"\n💡 实战建议:")
        print("-" * 20)
        best_map = sorted_maps[0][0]
        worst_map = sorted_maps[-1][0]
        print(f"• 如果是BO1，推荐选择 {best_map}")
        print(f"• 如果是BO3，优先ban掉 {worst_map}")
        print(f"• 关键地图可能是中间胜率的地图")

def quick_predict(team1_players: List[str], team2_players: List[str], 
                 team1_name: str = None, team2_name: str = None) -> SimpleMatchResult:
    """快速预测函数"""
    predictor = SimpleCS2Predictor()
    return predictor.predict_match(team1_players, team2_players, team1_name, team2_name)

if __name__ == "__main__":
    # 示例使用
    predictor = SimpleCS2Predictor()
    
    # 只需要输入选手名字！
    team1 = ["s1mple", "electroNic", "Perfecto", "b1t", "Boombl4"]
    team2 = ["ZywOo", "apEX", "dupreeh", "Magisk", "Spinx"]
    
    print("🚀 开始简化预测...")
    print(f"输入: {team1} vs {team2}")
    
    # 进行预测
    result = predictor.predict_match(
        team1_players=team1,
        team2_players=team2,
        team1_name="NAVI",
        team2_name="Vitality"
    )
    
    # 打印报告
    predictor.print_simple_report(result)