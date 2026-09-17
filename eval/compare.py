"""三方对比：纯 LLM / 单轮 RAG / Agent，同一评测集上量化提升。

用法：python eval/compare.py [--n 20]

用 embedding 相似度做结论一致性（快速、确定），避免大量 LLM-judge 调用；
run_agent.py 中的 LLM-judge 用于精确的头条指标。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

import config
from eval import metrics
from eval.dataset import split_cases
from src import prompts
from src.agent import ask, build_agent
from src.data import build_documents, load_cases
from src.llm import structured_invoke
from src.schemas import Diagnosis
from src.vectorstore import CaseStore, build_store

DIMS = ("病位", "病性", "病因")


def pure_llm(主诉: str) -> dict:
    """方案一：纯 LLM，无检索。"""
    msgs = [
        SystemMessage(content="你是资深中医专家，请根据患者主诉直接进行辨证，输出结构化诊断。"),
        HumanMessage(content=f"患者主诉：{主诉}"),
    ]
    return structured_invoke(Diagnosis, msgs).model_dump()


def single_rag(store: CaseStore, 主诉: str, k: int = 5) -> dict:
    """方案二：单轮 RAG，检索后一次 LLM 调用。"""
    context = json.dumps(store.retrieve(主诉, k=k), ensure_ascii=False, indent=2)
    msgs = [
        SystemMessage(content=prompts.FORMAT_SYSTEM + "\n\n检索到的病历证据：\n" + context),
        HumanMessage(content=f"患者主诉：{主诉}"),
    ]
    return structured_invoke(Diagnosis, msgs).model_dump()


def score_diag(diag: dict, qc: dict) -> dict:
    """对一个诊断结果打分：结论相似度 + 三个维度证据对齐。"""
    s = {"结论相似度": metrics.conclusion_similarity(diag.get("辨证结论", ""), qc["辨证结论"])}
    for dim in DIMS:
        s[f"{dim}对齐"] = metrics.evidence_alignment(diag.get(dim, []), qc[f"{dim}证据"])
    s["平均"] = float(np.mean(list(s.values())))
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="评测用例数")
    args = parser.parse_args()

    cases = load_cases(config.DATA_PATH)
    index_cases, query_cases = split_cases(cases)
    query_cases = query_cases[: args.n]

    store = CaseStore(build_store(build_documents(index_cases)), index_cases)

    accum = {"纯LLM": [], "单轮RAG": [], "问诊Agent": []}
    for i, qc in enumerate(query_cases):
        try:
            d_pure = pure_llm(qc["主诉"])
            d_rag = single_rag(store, qc["主诉"])
            agent, _ = build_agent(store, patient_case=qc)  # 多轮问诊 Agent
            d_agent = ask(agent, qc["主诉"], thread_id=f"cmp-{qc['case_id']}")
            accum["纯LLM"].append(score_diag(d_pure, qc))
            accum["单轮RAG"].append(score_diag(d_rag, qc))
            accum["问诊Agent"].append(score_diag(d_agent, qc))
            print(f"[{i + 1}/{len(query_cases)}] case_id={qc['case_id']} 完成")
        except Exception as e:  # 单个用例失败不中断整场评测
            print(f"[{i + 1}/{len(query_cases)}] case_id={qc['case_id']} 跳过：{e}")

    metrics_names = ["结论相似度", "病位对齐", "病性对齐", "病因对齐", "平均"]
    table = {}
    for name, rows in accum.items():
        table[name] = {m: round(float(np.mean([r[m] for r in rows])), 4) for m in metrics_names}

    print("\n" + "=" * 60)
    header = f"{'指标':<12}" + "".join(f"{m:>12}" for m in metrics_names)
    print(header)
    for name in accum:
        print(f"{name:<12}" + "".join(f"{table[name][m]:>12.4f}" for m in metrics_names))

    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    (out / "compare.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 -> {out / 'compare.json'}")


if __name__ == "__main__":
    main()
