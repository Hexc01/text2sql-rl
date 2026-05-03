import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str = None) -> logging.Logger:
    """设置统一日志格式"""
    logger = logging.getLogger("text2sql-rl")
    logger.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
