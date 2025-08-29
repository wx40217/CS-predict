#!/usr/bin/env python3
"""
虚拟环境设置脚本 - CS2比赛预测模型
自动创建和配置虚拟环境
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, check=True):
    """执行命令并处理错误"""
    print(f"执行命令: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        if e.stderr:
            print(f"错误信息: {e.stderr}")
        return False

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python版本检查通过: {version.major}.{version.minor}.{version.micro}")
    return True

def check_system_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    # 检查操作系统
    os_name = platform.system()
    print(f"操作系统: {os_name}")
    
    # 检查可用内存（Linux）
    if os_name == "Linux":
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if 'MemTotal' in line:
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb / 1024 / 1024
                        print(f"系统内存: {mem_gb:.1f} GB")
                        
                        if mem_gb < 16:
                            print("⚠️  警告: 建议至少16GB内存，当前内存可能不足")
                        else:
                            print("✅ 内存检查通过")
                        break
        except:
            print("无法检测内存信息")
    
    # 检查CUDA（如果可用）
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 检测到NVIDIA GPU")
            # 提取GPU信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX' in line or 'GTX' in line or 'Tesla' in line:
                    print(f"GPU: {line.strip()}")
                    break
        else:
            print("⚠️  未检测到NVIDIA GPU，将使用CPU训练")
    except FileNotFoundError:
        print("⚠️  未安装NVIDIA驱动或nvidia-smi不可用")

def create_virtual_environment():
    """创建虚拟环境"""
    print("🔧 创建虚拟环境...")
    
    venv_name = "cs2_predictor_env"
    venv_path = Path(venv_name)
    
    # 如果虚拟环境已存在，询问是否重新创建
    if venv_path.exists():
        response = input(f"虚拟环境 '{venv_name}' 已存在，是否重新创建？(y/N): ")
        if response.lower() in ['y', 'yes']:
            print("删除现有虚拟环境...")
            if platform.system() == "Windows":
                run_command(f'rmdir /s /q "{venv_name}"', check=False)
            else:
                run_command(f'rm -rf "{venv_name}"', check=False)
        else:
            print("使用现有虚拟环境")
            return venv_path
    
    # 创建虚拟环境
    if not run_command(f'python -m venv "{venv_name}"'):
        print("❌ 虚拟环境创建失败")
        return None
    
    print(f"✅ 虚拟环境 '{venv_name}' 创建成功")
    return venv_path

def get_activation_command(venv_path):
    """获取虚拟环境激活命令"""
    if platform.system() == "Windows":
        return f'"{venv_path}\\Scripts\\activate"'
    else:
        return f'source "{venv_path}/bin/activate"'

def install_dependencies(venv_path):
    """安装依赖包"""
    print("📦 安装依赖包...")
    
    # 获取pip路径
    if platform.system() == "Windows":
        pip_path = venv_path / "Scripts" / "pip"
    else:
        pip_path = venv_path / "bin" / "pip"
    
    # 升级pip
    print("升级pip...")
    if not run_command(f'"{pip_path}" install --upgrade pip'):
        print("⚠️  pip升级失败，继续安装依赖...")
    
    # 安装PyTorch（根据系统选择合适版本）
    print("安装PyTorch...")
    try:
        # 检测CUDA版本
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0 and 'CUDA Version' in result.stdout:
            # 安装CUDA版本的PyTorch
            torch_cmd = f'"{pip_path}" install torch torchvision --index-url https://download.pytorch.org/whl/cu118'
        else:
            # 安装CPU版本的PyTorch
            torch_cmd = f'"{pip_path}" install torch torchvision --index-url https://download.pytorch.org/whl/cpu'
    except:
        # 默认安装CPU版本
        torch_cmd = f'"{pip_path}" install torch torchvision --index-url https://download.pytorch.org/whl/cpu'
    
    if not run_command(torch_cmd):
        print("❌ PyTorch安装失败")
        return False
    
    # 安装其他依赖
    print("安装项目依赖...")
    if not run_command(f'"{pip_path}" install -r requirements.txt'):
        print("❌ 依赖安装失败")
        return False
    
    print("✅ 所有依赖安装完成")
    return True

def create_activation_script(venv_path):
    """创建激活脚本"""
    print("📝 创建激活脚本...")
    
    activation_cmd = get_activation_command(venv_path)
    
    if platform.system() == "Windows":
        script_content = f"""@echo off
echo 🎯 激活CS2预测模型虚拟环境...
call {activation_cmd}
echo ✅ 虚拟环境已激活
echo 📁 当前目录: %CD%
echo 🐍 Python版本:
python --version
echo 📦 已安装的包:
pip list | findstr -i "torch hltv pandas"
echo.
echo 💡 使用说明:
echo   - 训练模型: python main.py --mode train
echo   - 进行预测: python main.py --mode predict
echo   - 交互模式: python main.py --mode interactive
echo.
cmd /k
"""
        script_file = "activate_env.bat"
    else:
        script_content = f"""#!/bin/bash
echo "🎯 激活CS2预测模型虚拟环境..."
{activation_cmd}
echo "✅ 虚拟环境已激活"
echo "📁 当前目录: $(pwd)"
echo "🐍 Python版本:"
python --version
echo "📦 已安装的包:"
pip list | grep -E "(torch|hltv|pandas)"
echo ""
echo "💡 使用说明:"
echo "  - 训练模型: python main.py --mode train"
echo "  - 进行预测: python main.py --mode predict"
echo "  - 交互模式: python main.py --mode interactive"
echo ""
exec $SHELL
"""
        script_file = "activate_env.sh"
    
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 在Unix系统上设置执行权限
    if platform.system() != "Windows":
        os.chmod(script_file, 0o755)
    
    print(f"✅ 激活脚本已创建: {script_file}")
    return script_file

def test_installation(venv_path):
    """测试安装"""
    print("🧪 测试安装...")
    
    # 获取python路径
    if platform.system() == "Windows":
        python_path = venv_path / "Scripts" / "python"
    else:
        python_path = venv_path / "bin" / "python"
    
    # 测试导入关键包
    test_script = '''
import sys
print(f"Python版本: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ PyTorch导入失败: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas导入失败: {e}")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy导入失败: {e}")

try:
    from hltv_async_api import Hltv
    print("✅ HLTV Async API")
except ImportError as e:
    print(f"❌ HLTV Async API导入失败: {e}")
'''
    
    with open('test_imports.py', 'w') as f:
        f.write(test_script)
    
    success = run_command(f'"{python_path}" test_imports.py')
    
    # 清理测试文件
    try:
        os.remove('test_imports.py')
    except:
        pass
    
    return success

def main():
    """主函数"""
    print("🎯 CS2比赛预测模型 - 环境设置")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        return False
    
    # 检查系统要求
    check_system_requirements()
    
    # 创建虚拟环境
    venv_path = create_virtual_environment()
    if not venv_path:
        return False
    
    # 安装依赖
    if not install_dependencies(venv_path):
        return False
    
    # 创建激活脚本
    script_file = create_activation_script(venv_path)
    
    # 测试安装
    if not test_installation(venv_path):
        print("⚠️  安装测试失败，但可能仍然可用")
    
    print("\n" + "=" * 50)
    print("🎉 环境设置完成！")
    print("=" * 50)
    print(f"📁 虚拟环境位置: {venv_path.absolute()}")
    print(f"🚀 激活脚本: {script_file}")
    
    print("\n🔥 下一步操作:")
    if platform.system() == "Windows":
        print(f"1. 双击运行: {script_file}")
        print("2. 或在命令行执行:")
        print(f"   {get_activation_command(venv_path)}")
    else:
        print(f"1. 运行: ./{script_file}")
        print("2. 或手动激活:")
        print(f"   {get_activation_command(venv_path)}")
    
    print("\n3. 激活环境后，可以运行:")
    print("   python main.py --mode collect    # 收集数据")
    print("   python main.py --mode train      # 训练模型")
    print("   python main.py --mode predict    # 进行预测")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ 环境设置失败")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)