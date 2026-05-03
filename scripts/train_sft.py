"""Text2SQL SFT 训练脚本"""

import argparse
from pathlib import Path

import torch
from trl import SFTConfig, SFTTrainer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import Text2SQLDataset
from src.models import load_model_and_tokenizer
from src.utils import load_config, set_seed, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Text2SQL SFT Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging()
    set_seed(args.seed)

    logger.info("=" * 50)
    logger.info("Text2SQL SFT Training")
    logger.info(f"Model: {config['model']['name']}")
    logger.info("=" * 50)

    # 加载模型
    logger.info("[1/3] Loading model...")
    model, tokenizer = load_model_and_tokenizer(config)

    # 加载数据集
    logger.info("[2/3] Loading dataset...")
    train_dataset = Text2SQLDataset(
        data_path=config["data"]["train_path"],
        tokenizer=tokenizer,
        max_prompt_length=config["data"]["max_prompt_length"],
        max_completion_length=config["data"]["max_completion_length"],
    )

    # 构建 SFT 格式的数据集
    def format_sample(example):
        prompt = example["prompt"]
        sql = example["ground_truth_sql"]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": sql},
        ]
        return {"messages": messages}

    from datasets import Dataset
    raw_data = []
    for i in range(len(train_dataset)):
        raw_data.append(format_sample(train_dataset[i]))
    sft_dataset = Dataset.from_list(raw_data)

    # 训练参数
    logger.info("[3/3] Configuring trainer...")
    training = config["training"]
    output = config["output"]

    sft_args = SFTConfig(
        output_dir=output["dir"],
        num_train_epochs=training["num_epochs"],
        per_device_train_batch_size=training["batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        gradient_checkpointing=training.get("gradient_checkpointing", False),
        learning_rate=training["learning_rate"],
        warmup_steps=training["warmup_steps"],
        max_grad_norm=training["max_grad_norm"],
        weight_decay=training["weight_decay"],
        bf16=training.get("bf16", True),
        logging_steps=output["logging_steps"],
        save_steps=output["save_steps"],
        max_length=config["data"]["max_prompt_length"] + config["data"]["max_completion_length"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    trainer.save_model(config["output"]["checkpoint_dir"])
    logger.info(f"Model saved to {config['output']['checkpoint_dir']}")


if __name__ == "__main__":
    main()
