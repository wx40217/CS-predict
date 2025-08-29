"""
API检查器 - 检查各种HLTV数据源的可用性
"""
import requests
import asyncio
import aiohttp
from typing import Dict, List, Optional
import json
import logging

class APIChecker:
    """API可用性检查器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def check_hltv_website(self) -> Dict:
        """检查HLTV官网可用性"""
        try:
            response = requests.get('https://www.hltv.org', timeout=10)
            return {
                'name': 'HLTV官网',
                'status': 'available' if response.status_code == 200 else 'error',
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds()
            }
        except Exception as e:
            return {
                'name': 'HLTV官网',
                'status': 'error',
                'error': str(e)
            }
    
    def check_hltv_api_vercel(self) -> Dict:
        """检查之前使用的Vercel API"""
        try:
            response = requests.get('https://hltv-api.vercel.app/api/rankings', timeout=10)
            return {
                'name': 'HLTV API Vercel',
                'status': 'available' if response.status_code == 200 else 'error',
                'status_code': response.status_code,
                'data_available': bool(response.json() if response.status_code == 200 else False)
            }
        except Exception as e:
            return {
                'name': 'HLTV API Vercel',
                'status': 'error',
                'error': str(e)
            }
    
    def check_liquipedia_api(self) -> Dict:
        """检查Liquipedia API"""
        try:
            headers = {'User-Agent': 'CS2PredictionBot/1.0'}
            response = requests.get(
                'https://liquipedia.net/counterstrike/api.php?action=query&list=categorymembers&cmtitle=Category:Teams&format=json',
                headers=headers,
                timeout=10
            )
            return {
                'name': 'Liquipedia API',
                'status': 'available' if response.status_code == 200 else 'error',
                'status_code': response.status_code,
                'data_available': 'query' in response.json() if response.status_code == 200 else False
            }
        except Exception as e:
            return {
                'name': 'Liquipedia API',
                'status': 'error',
                'error': str(e)
            }
    
    async def check_hltv_async_api(self) -> Dict:
        """检查hltv-async-api库"""
        try:
            # 尝试导入库
            from hltv_async_api import Hltv
            
            # 尝试创建实例
            async with Hltv() as hltv:
                # 尝试获取数据
                rankings = await hltv.get_team_ranking()
                
            return {
                'name': 'HLTV Async API',
                'status': 'available',
                'library_imported': True,
                'data_retrieved': len(rankings) > 0 if rankings else False
            }
        except ImportError:
            return {
                'name': 'HLTV Async API',
                'status': 'not_installed',
                'error': 'Library not installed'
            }
        except Exception as e:
            return {
                'name': 'HLTV Async API',
                'status': 'error',
                'error': str(e)
            }
    
    def check_all_apis(self) -> List[Dict]:
        """检查所有API源"""
        results = []
        
        print("🔍 检查API可用性...")
        
        # 检查HLTV官网
        print("  检查HLTV官网...")
        results.append(self.check_hltv_website())
        
        # 检查Vercel API
        print("  检查Vercel API...")
        results.append(self.check_hltv_api_vercel())
        
        # 检查Liquipedia
        print("  检查Liquipedia API...")
        results.append(self.check_liquipedia_api())
        
        # 检查异步API
        print("  检查HLTV Async API...")
        async_result = asyncio.run(self.check_hltv_async_api())
        results.append(async_result)
        
        return results
    
    def print_results(self, results: List[Dict]):
        """打印检查结果"""
        print("\n📊 API检查结果:")
        print("=" * 50)
        
        for result in results:
            name = result['name']
            status = result['status']
            
            if status == 'available':
                status_icon = "✅"
            elif status == 'not_installed':
                status_icon = "📦"
            else:
                status_icon = "❌"
            
            print(f"{status_icon} {name}: {status}")
            
            if 'status_code' in result:
                print(f"   状态码: {result['status_code']}")
            
            if 'response_time' in result:
                print(f"   响应时间: {result['response_time']:.2f}s")
            
            if 'data_available' in result:
                print(f"   数据可用: {'是' if result['data_available'] else '否'}")
            
            if 'error' in result:
                print(f"   错误: {result['error']}")
            
            print()
    
    def recommend_solution(self, results: List[Dict]) -> str:
        """根据检查结果推荐解决方案"""
        available_apis = [r for r in results if r['status'] == 'available']
        
        if not available_apis:
            return """
🚨 所有API都不可用！推荐解决方案：

1. 📊 使用模拟数据进行演示
2. 🔧 实现网页爬取方案
3. 📁 使用本地数据文件
4. ⏳ 等待API恢复或寻找新的数据源
"""
        
        # 找到最好的API
        best_api = None
        for result in available_apis:
            if 'HLTV' in result['name'] and result.get('data_available', False):
                best_api = result
                break
        
        if not best_api:
            best_api = available_apis[0]
        
        return f"""
✅ 推荐使用: {best_api['name']}

该API当前可用，建议：
1. 优先使用此API获取数据
2. 实现数据缓存机制
3. 准备备用方案以防API失效
4. 定期检查API状态
"""

def main():
    """主函数"""
    checker = APIChecker()
    
    print("🔍 开始检查CS2/HLTV数据API...")
    
    results = checker.check_all_apis()
    checker.print_results(results)
    
    recommendation = checker.recommend_solution(results)
    print(recommendation)
    
    # 生成检查报告
    report = {
        'check_time': str(asyncio.get_event_loop().time()),
        'results': results,
        'recommendation': recommendation
    }
    
    with open('api_check_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("📄 检查报告已保存到: api_check_report.json")

if __name__ == "__main__":
    main()