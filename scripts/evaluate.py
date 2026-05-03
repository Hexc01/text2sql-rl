"""Text2SQL 模型评估脚本"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rewards.sql_reward import execution_accuracy, check_sql_syntax
from src.utils import setup_logging, set_seed


def evaluate(model_path: str, test_data_path: str, db_path: str = None, seed: int = 42):
    """评估模型"""
    logger = setup_logging()
    set_seed(seed)

    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    correct = 0
    total = len(test_data)
    syntax_correct = 0

    logger.info(f"Evaluating on {total} examples...")

    for i, item in enumerate(test_data):
        prompt = f"""You are a SQL expert. Generate the SQL query for the given question.

Schema: {item.get('schema', '')}
Question: {item['question']}
SQL:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
            )

        predicted_sql = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ground_truth = item["sql"]

        if check_sql_syntax(predicted_sql) >= 0.5:
            syntax_correct += 1

        if execution_accuracy(predicted_sql, ground_truth, db_path) >= 0.9:
            correct += 1

        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i+1}/{total}")

    results = {
        "execution_accuracy": correct / total,
        "syntax_accuracy": syntax_correct / total,
    }

    logger.info("=" * 50)
    logger.info("Results:")
    logger.info(f"  Execution Accuracy: {results['execution_accuracy']*100:.2f}% ({correct}/{total})")
    logger.info(f"  Syntax Correct: {results['syntax_accuracy']*100:.2f}% ({syntax_correct}/{total})")
    logger.info("=" * 50)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(args.model, args.data, args.db, args.seed)
