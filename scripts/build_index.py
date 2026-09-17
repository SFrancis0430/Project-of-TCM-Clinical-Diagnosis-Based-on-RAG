"""离线重建向量索引。

用法：python scripts/build_index.py

会清空并重建 config.DB_DIR 下的 Chroma 索引（可随时从 _message_list.json 重建）。
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.data import build_documents, load_cases
from src.vectorstore import build_store


def main() -> None:
    print(f"读取数据：{config.DATA_PATH}")
    cases = load_cases(config.DATA_PATH)
    print(f"共 {len(cases)} 条病历")

    if Path(config.DB_DIR).exists():
        print(f"⚠ 检测到已存在索引 {config.DB_DIR}，将清空重建…")
        shutil.rmtree(config.DB_DIR)

    docs = build_documents(cases)
    store = build_store(docs, config.DB_DIR)
    print(f"✅ 索引已构建：{len(docs)} 个文档 -> {config.DB_DIR}")
    try:
        print(f"   集合内文档数：{store._collection.count()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
