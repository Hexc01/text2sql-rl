import os
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件，支持环境变量覆盖"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 环境变量覆盖
    env_overrides = {
        "T2S_MODEL_PATH": ("model", "path"),
        "T2S_TRAIN_DATA": ("data", "train_path"),
        "T2S_VAL_DATA": ("data", "val_path"),
        "T2S_OUTPUT_DIR": ("output", "dir"),
        "T2S_LEARNING_RATE": ("training", "learning_rate"),
        "T2S_BATCH_SIZE": ("training", "batch_size"),
        "T2S_NUM_EPOCHS": ("training", "num_epochs"),
    }

    for env_key, config_path_tuple in env_overrides.items():
        value = os.environ.get(env_key)
        if value is not None:
            section, key = config_path_tuple
            if section in config:
                # 类型转换
                existing = config[section][key]
                if isinstance(existing, int):
                    value = int(value)
                elif isinstance(existing, float):
                    value = float(value)
                config[section][key] = value

    return config
