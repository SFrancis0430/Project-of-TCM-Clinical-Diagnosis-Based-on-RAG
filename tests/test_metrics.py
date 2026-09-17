"""指标单测：只测纯数值函数（不触发 embedding 下载）。"""
from eval import metrics


def test_cosine():
    assert metrics.cosine([1, 0], [1, 0]) == 1.0
    assert metrics.cosine([1, 0], [0, 1]) == 0.0
    assert metrics.cosine([], [1, 0]) == 0.0


def test_ndcg_perfect_rank():
    # 已按相关性降序 -> nDCG = 1
    assert metrics.ndcg_at_k([1.0, 0.5, 0.2], 3) == 1.0


def test_ndcg_empty():
    assert metrics.ndcg_at_k([], 3) == 0.0


def test_avg_sim():
    assert metrics.avg_sim_at_k([0.2, 0.6], 2) == 0.4


def test_mrr():
    assert metrics.mrr_at_k([0.9, 0.5], 2, threshold=0.85) == 1.0
    assert metrics.mrr_at_k([0.5, 0.9], 2, threshold=0.85) == 0.5
    assert metrics.mrr_at_k([0.3, 0.4], 2, threshold=0.85) == 0.0
