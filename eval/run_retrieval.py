"""检索评测：主诉 -> 检索 top-k -> 与金标准辨证结论的相似度算 nDCG/MRR/AvgSim。

用法：python eval/run_retrieval.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
from eval import metrics
from eval.dataset import split_cases
from src.data import build_documents, load_cases
from src.vectorstore import CaseStore, build_store


def main() -> None:
    cases = load_cases(config.DATA_PATH)
    index_cases, query_cases = split_cases(cases)
    print(f"index={len(index_cases)} 条建库，query={len(query_cases)} 条评测")

    docs = build_documents(index_cases)
    store = CaseStore(build_store(docs), index_cases)

    # 预计算 index 病历的辨证结论 embedding（按 case_id 索引）
    case_id_to_emb = {
        int(c["case_id"]): emb
        for c, emb in zip(index_cases, metrics.embed_many([c["辨证结论"] for c in index_cases]))
    }

    k = config.RETRIEVAL_K
    ndcgs, mrrs, avgs = [], [], []
    for qc in query_cases:
        gold_emb = metrics.embed_many([qc["辨证结论"]])[0]
        sims = []
        for doc, _ in store.retrieve_with_scores(qc["主诉"], k=k):
            emb = case_id_to_emb.get(int(doc.metadata.get("case_id")))
            sims.append(metrics.cosine(emb, gold_emb) if emb is not None else 0.0)
        ndcgs.append(metrics.ndcg_at_k(sims, k))
        mrrs.append(metrics.mrr_at_k(sims, k))
        avgs.append(metrics.avg_sim_at_k(sims, k))

    report = {
        "index_size": len(index_cases),
        "query_size": len(query_cases),
        "k": k,
        "ndcg@k": float(np.mean(ndcgs)),
        "mrr@k": float(np.mean(mrrs)),
        "avg_sim@k": float(np.mean(avgs)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    (out / "retrieval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 -> {out / 'retrieval.json'}")


if __name__ == "__main__":
    main()
