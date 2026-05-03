from trl import GRPOConfig

from src.rewards.sql_reward import format_sql_reward, execution_accuracy


def build_grpo_config(config: dict) -> GRPOConfig:
    """从 YAML 配置构建 GRPOConfig"""
    training = config["training"]
    output = config["output"]
    data = config["data"]
    rl = config["rl"]

    return GRPOConfig(
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
        fp16=training.get("fp16", False),
        logging_steps=output["logging_steps"],
        save_steps=output["save_steps"],
        eval_steps=output.get("eval_steps", 100),
        num_generations=rl["num_generations"],
        max_completion_length=data["max_completion_length"],
        temperature=rl.get("temperature", 1.0),
        top_p=rl.get("top_p", 1.0),
    )


def build_reward_func(db_path: str = None):
    """构建 GRPO 奖励函数，绑定 db_path"""

    def reward_function(completions: list, **kwargs) -> list:
        ground_truths = kwargs.get("ground_truth_sql", [])
        rewards = []
        for completion, gt_sql in zip(completions, ground_truths):
            if db_path and not gt_sql:
                # 无 ground truth 时只检查语法和执行
                from src.rewards.sql_reward import check_sql_syntax
                reward = check_sql_syntax(completion) * 0.5
            else:
                reward = format_sql_reward(completion, gt_sql, db_path=db_path)
            rewards.append(reward)
        return rewards

    return reward_function
