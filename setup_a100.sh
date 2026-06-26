#!/bin/bash
# A100服务器环境配置脚本（使用conda）
# 服务器配置：4x NVIDIA A100-SXM4-40GB, CUDA 12.9

echo "=========================================="
echo "  A100服务器环境配置"
echo "=========================================="

# 检查NVIDIA驱动
echo ""
echo "[1/5] 检查NVIDIA驱动..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 创建conda环境
echo ""
echo "[2/5] 创建conda环境..."
ENV_NAME="robotics_final"

# 如果环境已存在，先删除
conda remove -n $ENV_NAME --all -y 2>/dev/null || true

# 创建新环境
conda create -n $ENV_NAME python=3.10 -y

echo "环境 $ENV_NAME 创建完成"

# 安装PyTorch（CUDA 12.4兼容版本）
echo ""
echo "[3/5] 安装PyTorch..."
# 使用conda激活环境并安装
source activate $ENV_NAME 2>/dev/null || conda activate $ENV_NAME 2>/dev/null
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
echo ""
echo "[4/5] 安装其他依赖..."
pip install numpy mujoco matplotlib

# 验证安装
echo ""
echo "[5/5] 验证安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import mujoco; print(f'MuJoCo版本: {mujoco.__version__}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"

echo ""
echo "=========================================="
echo "  环境配置完成！"
echo "=========================================="
echo ""
echo "使用方法："
echo "  1. 激活环境: conda activate $ENV_NAME"
echo "  2. 进入项目目录: cd /data/yuwj/linhuajie/robotics-final-project"
echo "  3. 开始训练: python train_lstm.py --data_dir data/20260626_005525 --epochs 200"
echo ""
echo "环境名称: $ENV_NAME"
echo "Python版本: 3.10"
echo "PyTorch: CUDA 12.4兼容版本"
