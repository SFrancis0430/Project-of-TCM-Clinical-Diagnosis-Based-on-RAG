"""一键冒烟测试：验证「数据 -> 建库 -> 检索 -> Agent」全链路。

用法：python scripts/smoke_test.py

用 5 条病历跑通全流程，避免全量数据与大量 LLM 调用。
若 .env 未配置 LLM_API_KEY，则只验证检索链路。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.agent import ask, build_agent
from src.data import build_documents, load_cases
from src.vectorstore import CaseStore, build_store


def main() -> None:
    cases = load_cases(config.DATA_PATH)[:5]
    docs = build_documents(cases)
    chroma = build_store(docs)  # 内存库
    store = CaseStore(chroma, cases)

    print("=" * 50)
    print("[1/3] 检索测试")
    q = cases[0]["主诉"][:40]
    print(f"query: {q}")
    for item in store.retrieve(q, k=2):
        print(f"  - case_id={item['case_id']} 辨证结论={item['辨证结论']}")

    print("[2/3] 元数据过滤测试")
    print(store.filter_by("病性", "寒", k=2)[:1])

    print("[3/3] 完整病历详情测试")
    print(store.get_detail(cases[0]["case_id"])[:200], "...")

    if not config.LLM_API_KEY:
        print("\n⚠ 未配置 LLM_API_KEY，跳过 Agent 测试（检索链路已通）。")
        return

    print("\n" + "=" * 50)
    print("[Agent] 构建 Agent 并提问")
    agent, _ = build_agent(store)
    result = ask(agent, cases[0]["主诉"])
    print("辨证结论:", result.get("辨证结论"))
    print("病位:", result.get("病位"))
    print("病性:", result.get("病性"))
    print("病因:", result.get("病因"))
    print("依据:", result.get("依据"))
    print("✅ 冒烟测试通过")


if __name__ == "__main__":
    main()
