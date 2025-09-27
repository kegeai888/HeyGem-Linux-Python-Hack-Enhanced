#!/bin/bash

# =============================================================================
# HeyGem AI 数字人生成器启动脚本
# 作者: 科哥
# 功能: 安全启动 app.py，包含环境检查、端口清理、内存管理
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # 无颜色

# 打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

print_header() {
    echo
    echo -e "${PURPLE}=================================${NC}"
    echo -e "${PURPLE}🎭 HeyGem AI 数字人生成器启动器${NC}"
    echo -e "${PURPLE}=================================${NC}"
    echo
}

# 检查是否为root用户
check_user() {
    if [ "$EUID" -eq 0 ]; then
        print_message $YELLOW "⚠️  警告: 检测到以root用户运行，建议使用普通用户"
        print_message $CYAN "   💡 提示: 继续以root用户运行（自动跳过确认）"
    fi
}

# 检查conda是否可用
check_conda() {
    print_message $BLUE "🔍 检查Conda环境..."

    if ! command -v conda &> /dev/null; then
        print_message $RED "❌ 错误: 未找到conda命令"
        print_message $YELLOW "   请确保已安装Anaconda或Miniconda"
        exit 1
    fi

    # 初始化conda（如果需要）
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc"
    fi

    # 确保conda命令可用
    eval "$(conda shell.bash hook)" 2>/dev/null || true

    print_message $GREEN "✅ Conda环境检查完成"
}

# 激活虚拟环境
activate_environment() {
    print_message $BLUE "🚀 激活虚拟环境 'base'..."

    # 尝试多种方式初始化和激活conda
    if command -v conda &> /dev/null; then
        # 方法1: 直接使用conda activate
        if conda activate base &> /dev/null; then
            print_message $GREEN "✅ 成功激活虚拟环境 'base' (方法1)"
        # 方法2: 使用eval初始化后激活
        elif eval "$(conda shell.bash hook)" && conda activate base &> /dev/null; then
            print_message $GREEN "✅ 成功激活虚拟环境 'base' (方法2)"
        # 方法3: 使用source方式
        elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ] && source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate base &> /dev/null; then
            print_message $GREEN "✅ 成功激活虚拟环境 'base' (方法3)"
        # 方法4: 使用miniconda路径
        elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] && source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate base &> /dev/null; then
            print_message $GREEN "✅ 成功激活虚拟环境 'base' (方法4)"
        else
            print_message $YELLOW "⚠️  无法激活conda环境，使用系统Python"
            print_message $CYAN "   💡 提示: 如果需要conda环境，请手动运行 'conda activate base'"
        fi
    else
        print_message $YELLOW "⚠️  未找到conda，使用系统Python"
    fi

    # 显示当前Python信息
    if command -v python &> /dev/null; then
        python_info=$(python --version 2>&1)
        python_path=$(which python)
        print_message $CYAN "   当前Python: $python_info"
        print_message $CYAN "   Python路径: $python_path"
    else
        print_message $RED "❌ 未找到Python解释器"
        exit 1
    fi
}

# 检查Python环境和依赖
check_dependencies() {
    print_message $BLUE "🔍 检查Python依赖..."

    # 检查Python版本
    python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [ "$python_version" != "3.8" ]; then
        print_message $YELLOW "⚠️  警告: 当前Python版本为 $python_version，推荐使用Python 3.8"
    fi

    # 检查关键依赖包
    local required_packages=("gradio" "cv2" "numpy")
    local missing_packages=()

    for package in "${required_packages[@]}"; do
        # 特殊处理opencv-python包名
        local import_name=$package
        if [ "$package" = "opencv-python" ]; then
            import_name="cv2"
        fi

        if ! python -c "import $import_name" &> /dev/null; then
            missing_packages+=("$package")
        fi
    done

    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_message $RED "❌ 缺少以下依赖包: ${missing_packages[*]}"
        print_message $YELLOW "   请运行: pip install -r requirements.txt"
        print_message $CYAN "   💡 提示: 继续运行，某些功能可能受影响"
    else
        print_message $GREEN "✅ Python依赖检查完成"
    fi
}

# 检查GPU和CUDA
check_gpu() {
    print_message $BLUE "🎮 检查GPU环境..."

    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
        print_message $GREEN "✅ 检测到 $gpu_count 个NVIDIA GPU"

        # 显示GPU内存使用情况
        print_message $CYAN "   GPU内存使用情况:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | while read line; do
            print_message $CYAN "   $line"
        done
    else
        print_message $YELLOW "⚠️  警告: 未检测到NVIDIA GPU，将使用CPU运行（速度较慢）"
    fi
}

# 检查并清理7860端口
check_and_kill_port() {
    local port=7860
    print_message $BLUE "🔍 检查端口 $port..."

    # 查找占用端口的进程
    local pids=$(lsof -ti:$port 2>/dev/null || true)

    if [ -n "$pids" ]; then
        print_message $YELLOW "⚠️  端口 $port 被以下进程占用:"
        for pid in $pids; do
            local process_info=$(ps -p $pid -o pid,ppid,cmd --no-headers 2>/dev/null || echo "$pid Unknown Unknown")
            print_message $YELLOW "   PID: $process_info"
        done

        print_message $BLUE "🗑️  正在终止占用端口的进程..."
        for pid in $pids; do
            if kill -TERM $pid 2>/dev/null; then
                print_message $GREEN "   ✅ 成功终止进程 $pid"
            else
                print_message $YELLOW "   ⚠️  尝试强制终止进程 $pid"
                kill -KILL $pid 2>/dev/null || true
            fi
        done

        # 等待进程真正退出
        sleep 2

        # 再次检查
        local remaining_pids=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$remaining_pids" ]; then
            print_message $RED "❌ 仍有进程占用端口 $port: $remaining_pids"
            print_message $YELLOW "   正在强制清理..."
            for pid in $remaining_pids; do
                kill -KILL $pid 2>/dev/null || true
            done
            sleep 1
        fi
    fi

    print_message $GREEN "✅ 端口 $port 已清理完成"
}

# 内存和显存清理
cleanup_memory() {
    print_message $BLUE "🧹 清理系统内存..."

    # 清理Python缓存
    if command -v python &> /dev/null; then
        python -c "
import gc
import torch if 'torch' in locals() else None
if 'torch' in globals():
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
gc.collect()
print('Python内存清理完成')
" 2>/dev/null || true
    fi

    # 系统内存清理（需要root权限）
    if [ "$EUID" -eq 0 ]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
        print_message $GREEN "✅ 系统内存缓存已清理"
    else
        print_message $CYAN "   💡 提示: 以root用户运行可进行更深度的内存清理"
    fi
}

# 检查磁盘空间
check_disk_space() {
    print_message $BLUE "💾 检查磁盘空间..."

    local current_dir=$(pwd)
    local available_space=$(df "$current_dir" | tail -1 | awk '{print $4}')
    local available_gb=$((available_space / 1024 / 1024))

    if [ $available_gb -lt 5 ]; then
        print_message $RED "❌ 磁盘空间不足: 仅剩 ${available_gb}GB"
        print_message $YELLOW "   建议清理磁盘空间或移动到其他目录"
        exit 1
    elif [ $available_gb -lt 10 ]; then
        print_message $YELLOW "⚠️  磁盘空间较低: 剩余 ${available_gb}GB"
    else
        print_message $GREEN "✅ 磁盘空间充足: 剩余 ${available_gb}GB"
    fi
}

# 启动应用程序
start_application() {
    print_message $BLUE "🚀 启动 HeyGem AI 数字人生成器..."

    # 检查app.py是否存在
    if [ ! -f "app.py" ]; then
        print_message $RED "❌ 错误: 未找到 app.py 文件"
        print_message $YELLOW "   请确保在正确的项目目录中运行此脚本"
        exit 1
    fi

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU
    export PYTHONUNBUFFERED=1      # 实时输出
    export GRADIO_SERVER_NAME=0.0.0.0
    export GRADIO_SERVER_PORT=7860

    print_message $GREEN "✅ 环境变量已设置"
    print_message $CYAN "   🌐 访问地址: http://localhost:7860"
    print_message $CYAN "   🌐 网络地址: http://$(hostname -I | awk '{print $1}'):7860"
    print_message $CYAN ""
    print_message $PURPLE "🎉 正在启动应用程序，请稍候..."
    print_message $YELLOW "   💡 提示: 按 Ctrl+C 可以停止应用程序"
    echo

    # 启动应用
    python app.py
}

# 信号处理函数
cleanup_on_exit() {
    print_message $YELLOW "🛑 接收到停止信号，正在清理..."

    # 终止可能的子进程
    jobs -p | xargs -r kill 2>/dev/null || true

    # 清理临时文件
    rm -rf /tmp/gradio_* 2>/dev/null || true

    print_message $GREEN "✅ 清理完成，程序已退出"
    exit 0
}

# 主函数
main() {
    # 设置信号处理
    trap cleanup_on_exit SIGINT SIGTERM

    print_header

    # 执行各项检查
    check_user
    check_conda
    activate_environment
    check_dependencies
    check_gpu
    check_disk_space
    cleanup_memory
    check_and_kill_port

    print_message $GREEN "🎯 所有检查完成，准备启动应用程序"
    echo

    # 启动应用程序
    start_application
}

# 如果脚本被直接执行，则运行主函数
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi