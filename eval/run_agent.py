"""Agent 端到端评测：主诉 -> Agent 结构化诊断 -> 结论一致性(LLM-judge) + 证据对齐。

用法：python eval/run_agent.py [--n 30]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
from eval import metrics
from eval.dataset import split_cases
from eval.judge import judge_conclusion
from src import prompts
from src.agent import ask, build_agent
from src.data import build_documents, load_cases
from src.vectorstore import CaseStore, build_store

DIMS = ("病位", "病性", "病因")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="评测用例数（LLM 调用量大，默认 30）")
    parser.add_argument("--no-ask", action="store_true", help="关闭多轮问诊（作为对照 baseline）")
    args = parser.parse_args()

    cases = load_cases(config.DATA_PATH)
    index_cases, query_cases = split_cases(cases)
    query_cases = query_cases[: args.n]

    store = CaseStore(build_store(build_documents(index_cases)), index_cases)

    concl_scores, dim_scores = [], {d: [] for d in DIMS}
    rows = []
    for i, qc in enumerate(query_cases):
        try:
            if args.no_ask:
                agent, _ = build_agent(store)
                diag = ask(agent, qc["主诉"], thread_id=f"eval-{qc['case_id']}", system_prompt=prompts.AGENT_SYSTEM_NO_ASK)
            else:
                agent, _ = build_agent(store, patient_case=qc)  # 多轮问诊
                diag = ask(agent, qc["主诉"], thread_id=f"eval-{qc['case_id']}")
            score = judge_conclusion(diag.get("辨证结论", ""), qc["辨证结论"]).score
            concl_scores.append(score)
            row = {"case_id": qc["case_id"], "conclusion_score": score}
            for dim in DIMS:
                a = metrics.evidence_alignment(diag.get(dim, []), qc[f"{dim}证据"])
                dim_scores[dim].append(a)
                row[f"{dim}_align"] = a
            rows.append(row)
            print(f"[{i + 1}/{len(query_cases)}] case_id={qc['case_id']} 结论一致性={score:.2f}")
        except Exception as e:  # 单个用例失败不中断整场评测
            print(f"[{i + 1}/{len(query_cases)}] case_id={qc['case_id']} 跳过：{e}")

    report = {
        "n": len(query_cases),
        "结论一致性(mean)": float(np.mean(concl_scores)),
        "病位对齐": float(np.mean(dim_scores["病位"])),
        "病性对齐": float(np.mean(dim_scores["病性"])),
        "病因对齐": float(np.mean(dim_scores["病因"])),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    suffix = "noask" if args.no_ask else "ask"
    name = f"agent_{suffix}.json"
    detail_name = f"agent_{suffix}_details.json"
    (out / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / detail_name).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 -> {out / name}")


if __name__ == "__main__":
    main()
