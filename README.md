# Text2SQL Reinforcement Learning

使用本地 Qwen 模型进行 Text2SQL 强化学习训练。

## 项目结构

```
text2sql-rl/
├── src/                # 源代码
│   ├── data/           # 数据处理
│   ├── models/         # 模型定义
│   ├── rewards/        # 奖励函数
│   ├── trainers/       # 训练器
│   └── utils/          # 工具函数
├── data/               # 数据集
├── configs/            # 配置文件
├── scripts/            # 训练脚本
├── tests/              # 测试
├── outputs/            # 训练输出
├── checkpoints/        # 模型检查点
├── requirements.txt    # 依赖
└── README.md           # 项目说明
```

## 快速开始

```bash
# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 开始训练
python scripts/train.py --config configs/default.yaml
```

## 模型

基础模型: Qwen2.5-Coder-1.5B (本地路径: ~/models/Qwen2.5-Coder-1.5B)
