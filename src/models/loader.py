import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_model_and_tokenizer(config: dict):
    """根据配置加载模型和 tokenizer"""
    model_config = config["model"]
    model_path = model_config["path"]

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 模型加载参数
    dtype_str = model_config.get("torch_dtype", "bfloat16")
    torch_dtype = getattr(torch, dtype_str, torch.bfloat16)

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    # 4bit 量化
    if model_config.get("load_in_4bit", False):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    # LoRA
    lora_config = model_config.get("lora", None)
    if lora_config:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer
