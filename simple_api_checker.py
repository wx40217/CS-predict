"""
简单API检查器 - 不依赖外部库
"""
import urllib.request
import urllib.error
import json
import sys

def check_url(url, name):
    """检查URL可用性"""
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        with urllib.request.urlopen(req, timeout=10) as response:
            status_code = response.getcode()
            content_length = len(response.read())
            
            return {
                'name': name,
                'url': url,
                'status': 'available' if status_code == 200 else 'error',
                'status_code': status_code,
                'content_length': content_length
            }
    except Exception as e:
        return {
            'name': name,
            'url': url,
            'status': 'error',
            'error': str(e)
        }

def main():
    """检查主要API端点"""
    print("🔍 检查CS2/HLTV数据源可用性...")
    print("=" * 50)
    
    # 要检查的API端点
    apis_to_check = [
        ('https://www.hltv.org', 'HLTV官网'),
        ('https://hltv-api.vercel.app/api/rankings', 'HLTV API Vercel'),
        ('https://liquipedia.net/counterstrike/Main_Page', 'Liquipedia'),
    ]
    
    results = []
    
    for url, name in apis_to_check:
        print(f"  检查 {name}...")
        result = check_url(url, name)
        results.append(result)
        
        # 显示结果
        if result['status'] == 'available':
            print(f"    ✅ 可用 (状态码: {result['status_code']}, 大小: {result['content_length']} bytes)")
        else:
            print(f"    ❌ 不可用 - {result.get('error', '未知错误')}")
    
    print("\n📊 检查总结:")
    print("-" * 30)
    
    available_count = sum(1 for r in results if r['status'] == 'available')
    total_count = len(results)
    
    print(f"可用API: {available_count}/{total_count}")
    
    if available_count == 0:
        print("\n🚨 警告: 所有外部API都不可用!")
        print("推荐解决方案:")
        print("1. 📊 使用内置模拟数据")
        print("2. 📁 使用本地数据文件")
        print("3. 🔧 等待网络恢复后重试")
    elif available_count < total_count:
        print("\n⚠️  部分API不可用，建议使用可用的API源")
    else:
        print("\n✅ 所有API都可用!")
    
    # 保存结果
    with open('api_check_simple.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 检查结果已保存到: api_check_simple.json")
    
    return results

if __name__ == "__main__":
    main()