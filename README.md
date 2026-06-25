# 机器人导论大作业 - 行为克隆

## 项目结构

```
大作业/
├── env/                    # MuJoCo环境模块
│   ├── __init__.py
│   └── franka_env.py
├── controllers/            # 控制器模块
│   ├── __init__.py
│   ├── base_controller.py
│   └── ik_controller.py
├── franka_emika_panda/     # MuJoCo模型文件
├── collect.py              # 数据采集脚本
├── train.py                # 训练脚本
├── convert_to_lerobot.py   # 数据转换脚本
├── evaluate.py             # 评估脚本
└── README.md
```

## 使用流程

### 1. 环境配置

```bash
# 安装依赖
pip install mujoco numpy torch
```

### 2. 数据采集

```bash
# 采集100个episode的数据（默认方块位置固定）
python collect.py

# 采集500个episode的数据（方块位置随机）
python collect.py --episodes 500 --rand

# 指定输出目录
python collect.py --episodes 200 --out data/my_dataset
```

**采集参数说明：**
- `--episodes`: 采集的episode数量（默认100）
- `--rand`: 随机化方块初始位置
- `--out`: 指定输出目录
- `--seed`: 随机种子（默认42）

**输出格式：**
```
data/
├── metadata.npz          # 元数据（episode索引、成功标志等）
├── episode_0000.npz      # 第1个episode的观测和动作
├── episode_0001.npz      # 第2个episode
└── ...
```

### 3. 数据转换（可选）

如果使用LeRobot框架训练，需要将数据转换为zarr格式：

```bash
python convert_to_lerobot.py --data_dir data/20260625_123456
```

### 4. 训练模型

```bash
# 使用采集的数据训练MLP模型
python train.py --data_dir data/20260625_123456

# 自定义训练参数
python train.py \
    --data_dir data/20260625_123456 \
    --epochs 200 \
    --batch_size 128 \
    --lr 5e-4 \
    --output_dir checkpoints/my_model
```

**训练参数说明：**
- `--data_dir`: 数据目录（包含episode_*.npz文件）
- `--output_dir`: 模型保存目录（默认：data_dir/checkpoints）
- `--epochs`: 训练轮数（默认100）
- `--batch_size`: 批大小（默认64）
- `--lr`: 学习率（默认1e-3）
- `--weight_decay`: 权重衰减（默认1e-4）
- `--device`: 训练设备（默认：cuda if available）

**输出：**
- `checkpoints/model_best.pt`: 最佳模型权重

### 5. 评估模型

```bash
# 评估训练好的模型
python evaluate.py --model_path checkpoints/model_best.pt

# 自定义评估参数
python evaluate.py \
    --model_path checkpoints/model_best.pt \
    --episodes 20 \
    --max_steps 3000
```

**评估参数说明：**
- `--model_path`: 模型路径
- `--episodes`: 评估episode数（默认10）
- `--max_steps`: 每个episode最大步数（默认3000）
- `--device`: 推理设备

## 观测空间

| 索引 | 名称 | 维度 | 说明 |
|------|------|------|------|
| 0:4 | arm_q | 4 | 机械臂关节角度 |
| 4:8 | arm_dq | 4 | 机械臂关节速度 |
| 8:11 | ee_pos | 3 | 末端执行器位置 |
| 11:14 | blk_pos | 3 | 方块位置 |
| 14:17 | tgt_pos | 3 | 目标位置 |
| 17:20 | gripper_state | 3 | 夹爪状态（开度、距离、闭合标志） |

**观测维度：20**

## 动作空间

| 索引 | 名称 | 维度 | 说明 |
|------|------|------|------|
| 0:4 | arm_target | 4 | 机械臂目标关节角度 |
| 4 | grip_cmd | 1 | 夹爪命令（0=闭合，1=打开） |

**动作维度：5**

## 常见问题

### Q: 采集的数据不成功怎么办？
A: 检查以下几点：
1. 确保MuJoCo环境正常运行
2. 检查IK控制器参数是否合适
3. 尝试增加`--episodes`采集更多数据
4. 使用`--rand`随机化方块位置增加多样性

### Q: 训练loss不下降怎么办？
A: 尝试以下方法：
1. 增加数据量
2. 调整学习率（`--lr`）
3. 检查数据质量
4. 增加网络容量

### Q: 评估成功率低怎么办？
A: 可能的原因：
1. 训练数据不足
2. 过拟合（增加数据多样性）
3. 观测空间设计不合理
4. 动作空间定义有问题

## 云端运行建议

1. 使用桌面容器（需要GUI显示仿真环境）
2. 确保CUDA可用：`python -c "import torch; print(torch.cuda.is_available())"`
3. 数据采集可能需要较长时间，建议后台运行：`nohup python collect.py --episodes 500 > collect.log 2>&1 &`
