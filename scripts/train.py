"""Text2SQL 强化学习训练脚本"""

import argparse
from pathlib import Path

from trl import GRPOTrainer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import Text2SQLDataset
from src.models import load_model_and_tokenizer
from src.trainers import build_grpo_config, build_reward_func
from src.utils import load_config, set_seed, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Text2SQL RL Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging()
    set_seed(args.seed)

    logger.info("=" * 50)
    logger.info("Text2SQL RL Training")
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Algorithm: {config['training']['algorithm']}")
    logger.info(f"Output: {config['output']['dir']}")
    logger.info("=" * 50)

    # 加载模型
    logger.info("[1/4] Loading model...")
    model, tokenizer = load_model_and_tokenizer(config)

    # 梯度检查点
    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # 加载数据集
    logger.info("[2/4] Loading dataset...")
    train_dataset = Text2SQLDataset(
        data_path=config["data"]["train_path"],
        tokenizer=tokenizer,
        max_prompt_length=config["data"]["max_prompt_length"],
        max_completion_length=config["data"]["max_completion_length"],
    )

    # 配置训练参数
    logger.info("[3/4] Configuring trainer...")
    training_args = build_grpo_config(config)
    reward_func = build_reward_func(db_path=config["data"].get("db_path"))

    # 创建 trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=reward_func,
    )

    # 开始训练
    logger.info("[4/4] Starting training...")
    trainer.train()

    # 保存模型
    trainer.save_model(config["output"]["checkpoint_dir"])
    logger.info(f"Model saved to {config['output']['checkpoint_dir']}")


if __name__ == "__main__":
    main()
