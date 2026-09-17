"""评测指标：检索相关性（nDCG/MRR/AvgSim）与证据对齐。

相关性以「辨证结论」的语义相似度（embedding 余弦）作为分级标注，
适合本数据集（辨证结论为自由文本、无显式相关性标注）。
"""
import numpy as np

from src.vectorstore import get_embeddings


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def embed_many(texts: list[str]) -> np.ndarray:
    """批量 embedding（自动过滤空串）。"""
    valid = [t for t in texts if t and str(t).strip()]
    if not valid:
        return np.zeros((0, 0))
    return np.asarray(get_embeddings().embed_documents(valid))


def conclusion_similarity(pred: str, gold: str) -> float:
    """两个辨证结论的语义相似度（0~1）。"""
    if not pred or not gold:
        return 0.0
    pe = get_embeddings().embed_query(pred)
    ge = get_embeddings().embed_query(gold)
    return cosine(pe, ge)


def _dcg(sims: list[float]) -> float:
    return float(sum((2 ** s - 1) / np.log2(i + 2) for i, s in enumerate(sims)))


def ndcg_at_k(sims: list[float], k: int) -> float:
    """nDCG@k，用相似度作为分级相关性。"""
    sims = sims[:k]
    if not sims:
        return 0.0
    dcg = _dcg(sims)
    idcg = _dcg(sorted(sims, reverse=True))
    return dcg / idcg if idcg > 0 else 0.0


def avg_sim_at_k(sims: list[float], k: int) -> float:
    """top-k 平均相似度。"""
    s = sims[:k]
    return float(np.mean(s)) if s else 0.0


def mrr_at_k(sims: list[float], k: int, threshold: float = 0.85) -> float:
    """MRR@k：首个相似度 >= threshold 的命中位置的倒数。"""
    for i, s in enumerate(sims[:k]):
        if s >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def evidence_alignment(pred_items: list[str], gold_items: list[str]) -> float:
    """预测证据 vs 金标准证据的对齐度（双向最大余弦的 F1）。"""
    pred_items = [p for p in pred_items if p and str(p).strip()]
    gold_items = [g for g in gold_items if g and str(g).strip()]
    if not pred_items or not gold_items:
        return 0.0
    pe = embed_many(pred_items)
    ge = embed_many(gold_items)
    if pe.size == 0 or ge.size == 0:
        return 0.0
    p = float(np.mean([max(cosine(x, y) for y in ge) for x in pe]))
    r = float(np.mean([max(cosine(y, x) for x in pe) for y in ge]))
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
