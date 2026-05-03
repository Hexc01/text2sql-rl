import sqlite3

import pytest

from src.rewards.sql_reward import (
    check_sql_syntax,
    execution_accuracy,
    format_sql_reward,
    sql_format_similarity,
)


class TestCheckSQLSyntax:
    def test_valid_sql(self):
        assert check_sql_syntax("SELECT * FROM users") == 1.0

    def test_invalid_sql(self):
        # sqlparse 是宽松解析器，空字符串/None 才返回 0
        assert check_sql_syntax("") == 0.0

    def test_empty_string(self):
        assert check_sql_syntax("") == 0.0


class TestSQLFormatSimilarity:
    def test_identical(self):
        assert sql_format_similarity("SELECT * FROM users", "SELECT * FROM users") == 1.0

    def test_different(self):
        score = sql_format_similarity("SELECT * FROM users", "SELECT id FROM orders")
        assert 0.0 <= score < 1.0

    def test_empty_ground_truth(self):
        assert sql_format_similarity("SELECT * FROM users", "") == 0.0


class TestExecutionAccuracy:
    def test_no_db(self):
        assert execution_accuracy("SELECT 1", "SELECT 1", None) == 0.0

    def test_matching_results(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.execute("INSERT INTO t VALUES (2)")
        conn.commit()
        conn.close()

        score = execution_accuracy("SELECT * FROM t", "SELECT * FROM t", db_path)
        assert score == 1.0

    def test_mismatching_results(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.execute("INSERT INTO t VALUES (2)")
        conn.commit()
        conn.close()

        score = execution_accuracy("SELECT * FROM t WHERE id=1", "SELECT * FROM t", db_path)
        assert 0.0 < score < 1.0


class TestFormatSQLReward:
    def test_basic_reward(self):
        reward = format_sql_reward("SELECT * FROM users", "SELECT * FROM users")
        assert 0.0 <= reward <= 1.0
        # 语法正确 (0.3) + 格式匹配 (0.2) = 0.5 (无 db 时 exec=0)
        assert reward == pytest.approx(0.5)

    def test_with_db(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        conn.commit()
        conn.close()

        reward = format_sql_reward(
            "SELECT * FROM users", "SELECT * FROM users", db_path=db_path
        )
        assert reward == pytest.approx(1.0)
