import json
import tempfile
from pathlib import Path

import pytest

from src.data.dataset import Text2SQLDataset


@pytest.fixture
def sample_jsonl(tmp_path):
    """创建测试用 jsonl 文件"""
    data = [
        {"conversations": [{"value": "查询所有用户"}, {"value": "SELECT * FROM users"}]},
        {"conversations": [{"value": "查询活跃用户"}, {"value": "SELECT * FROM users WHERE active=1"}]},
    ]
    path = tmp_path / "test.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return str(path)


@pytest.fixture
def sample_json(tmp_path):
    """创建测试用 json 文件"""
    data = [
        {"question": "查询所有用户", "table_list": ["users"], "knowledge": "", "sql_id": "1"},
        {"question": "查询活跃用户", "table_list": ["users"], "knowledge": "active=1表示活跃", "sql_id": "2"},
    ]
    path = tmp_path / "test.json"
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    return str(path)


class TestText2SQLDataset:
    def test_load_jsonl(self, sample_jsonl):
        dataset = Text2SQLDataset(sample_jsonl)
        assert len(dataset) == 2
        item = dataset[0]
        assert "prompt" in item
        assert "ground_truth_sql" in item
        assert item["ground_truth_sql"] == "SELECT * FROM users"

    def test_load_json(self, sample_json):
        dataset = Text2SQLDataset(sample_json)
        assert len(dataset) == 2
        item = dataset[0]
        assert "prompt" in item
        assert "Text2SQL" in item["prompt"]

    def test_collate_fn(self, sample_jsonl):
        dataset = Text2SQLDataset(sample_jsonl)
        batch = [dataset[0], dataset[1]]
        result = Text2SQLDataset.collate_fn(batch)
        assert "prompt" in result
        assert "ground_truth_sql" in result
        assert len(result["prompt"]) == 2

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("a,b\n1,2")
        with pytest.raises(ValueError, match="Unsupported"):
            Text2SQLDataset(str(path))
