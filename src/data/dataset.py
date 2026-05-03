import json
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset


class Text2SQLDataset(Dataset):
    """Text2SQL 数据集 - 支持 jsonl 和 json 格式"""

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_prompt_length: int = 1024,
        max_completion_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

        data_path = Path(data_path)
        self.data = []

        if data_path.suffix == ".jsonl":
            self._load_jsonl(data_path)
        elif data_path.suffix == ".json":
            self._load_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    def _load_jsonl(self, path: Path):
        """加载 jsonl 格式 (my_text2sql.jsonl)"""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                convs = item.get("conversations", [])
                if len(convs) >= 2:
                    prompt = convs[0]["value"]
                    sql = convs[1]["value"]
                    self.data.append({"prompt": prompt, "sql": sql})

    def _load_json(self, path: Path):
        """加载 json 格式 (final_dataset.json) - 需要 schema"""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 尝试加载 schema
        schema_path = path.parent / "schema.json"
        schema_map = {}
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_list = json.load(f)
                schema_map = {s["table_name"]: s for s in schema_list}

        # 加载领域知识
        knowledge_path = path.parent / "common_knowledge.md"
        common_knowledge = ""
        if knowledge_path.exists():
            with open(knowledge_path, "r", encoding="utf-8") as f:
                common_knowledge = f.read()

        for item in raw:
            question = item.get("question", "")
            tables = item.get("table_list", [])
            knowledge = item.get("knowledge", "")

            # 构建 schema 描述
            schema_desc = self._build_schema_desc(tables, schema_map)

            # 构建完整 prompt
            prompt = self._build_prompt(question, schema_desc, knowledge, common_knowledge)

            self.data.append({
                "prompt": prompt,
                "sql": "",  # final_dataset.json 没有 SQL 答案
                "sql_id": item.get("sql_id", ""),
                "question": question,
            })

    def _build_schema_desc(self, table_names: List[str], schema_map: Dict) -> str:
        """根据表名构建 schema 描述"""
        parts = []
        for table_name in table_names:
            if table_name in schema_map:
                table = schema_map[table_name]
                desc = f"表名：{table_name}\n"
                if table.get("table_description"):
                    desc += f"描述：{table['table_description']}\n"
                desc += "字段：\n"
                for col in table.get("columns", []):
                    desc += f"  - {col['col']} ({col['type']}): {col.get('description', '')}\n"
                parts.append(desc)
            else:
                parts.append(f"表名：{table_name}\n(未找到表结构)")
        return "\n".join(parts)

    def _build_prompt(self, question: str, schema: str, knowledge: str, common_knowledge: str) -> str:
        """构建训练 prompt"""
        prompt = "你是Text2SQL助手。只输出可执行的SQL语句，不要包含任何解释文字。\n\n"
        prompt += f"数据库表结构：\n\n{schema}\n\n"

        if knowledge:
            prompt += f"领域知识：\n{knowledge}\n\n"
        if common_knowledge:
            prompt += f"通用知识：\n{common_knowledge}\n\n"

        prompt += f"问题：{question}"
        return prompt

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "ground_truth_sql": item.get("sql", ""),
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        return {
            "prompt": [item["prompt"] for item in batch],
            "ground_truth_sql": [item["ground_truth_sql"] for item in batch],
        }
