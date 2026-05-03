import sqlparse
import sqlite3
from typing import Optional


def format_sql_reward(predicted_sql: str, ground_truth_sql: str, db_path: str = None) -> float:
    """奖励函数：评估生成的 SQL 质量"""
    rewards = []

    # 1. SQL 语法检查 (0-0.3)
    syntax_reward = check_sql_syntax(predicted_sql)
    rewards.append(syntax_reward * 0.3)

    # 2. SQL 格式相似度 (0-0.2)
    format_reward = sql_format_similarity(predicted_sql, ground_truth_sql)
    rewards.append(format_reward * 0.2)

    # 3. 执行结果正确性 (0-0.5)
    exec_reward = execution_accuracy(predicted_sql, ground_truth_sql, db_path)
    rewards.append(exec_reward * 0.5)

    return sum(rewards)


def check_sql_syntax(sql: str) -> float:
    """检查 SQL 语法是否正确"""
    try:
        parsed = sqlparse.parse(sql)
        if parsed and len(parsed) > 0:
            return 1.0
    except Exception:
        pass
    return 0.0


def sql_format_similarity(predicted: str, ground_truth: str) -> float:
    """计算 SQL 格式相似度"""
    pred_normalized = sqlparse.format(predicted, strip_comments=True, reindent=True).strip().lower()
    gt_normalized = sqlparse.format(ground_truth, strip_comments=True, reindent=True).strip().lower()

    if pred_normalized == gt_normalized:
        return 1.0

    # 简单的 token 重叠计算
    pred_tokens = set(pred_normalized.split())
    gt_tokens = set(gt_normalized.split())

    if not gt_tokens:
        return 0.0

    overlap = len(pred_tokens & gt_tokens)
    return overlap / len(gt_tokens)


def execution_accuracy(predicted_sql: str, ground_truth_sql: str, db_path: Optional[str] = None) -> float:
    """执行 SQL 并比较结果"""
    if db_path is None:
        # 无数据库时跳过执行检查
        return 0.0

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(predicted_sql)
        pred_result = set(cursor.fetchall())

        cursor.execute(ground_truth_sql)
        gt_result = set(cursor.fetchall())

        conn.close()

        if pred_result == gt_result:
            return 1.0
        else:
            # 部分匹配
            overlap = len(pred_result & gt_result)
            total = len(gt_result)
            return overlap / total if total > 0 else 0.0

    except Exception:
        return 0.0
