"""统一配置：从 .env 读取路径、模型与检索参数。

所有模块从这里取配置，避免散落魔法值（也便于面试时解释"配置与代码分离"）。
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Windows 控制台默认 GBK，无法打印 emoji/特殊字符，统一转为 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


# --- 数据与存储 ---
DATA_PATH = _env("DATA_PATH", str(ROOT / "_message_list.json"))
DB_DIR = _env("DB_DIR", str(ROOT / "medical_db"))

# --- 模型 ---
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")
EMBEDDING_LOCAL_ONLY = _env("EMBEDDING_LOCAL_ONLY", "1") == "1"
LLM_MODEL = _env("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = _env("LLM_API_KEY", "")
LLM_API_BASE = _env("LLM_API_BASE", "https://api.deepseek.com")

# --- 检索与切块 ---
RETRIEVAL_K = int(_env("RETRIEVAL_K", "5"))
CHUNK_SIZE = int(_env("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "60"))

# --- 评测 ---
QUERY_SET_SIZE = int(_env("QUERY_SET_SIZE", "200"))
RANDOM_SEED = int(_env("RANDOM_SEED", "42"))
