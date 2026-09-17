"""评测数据划分：index 集（建库）与 held-out query 集（评测查询）。"""
import random
from typing import Any

import config


def split_cases(
    cases: list[dict[str, Any]],
    n_query: int | None = None,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """随机划分，返回 (index_cases, query_cases)。

    query 集永不进入向量库，避免 Agent 直接检索到原病历作弊。
    """
    n_query = n_query if n_query is not None else config.QUERY_SET_SIZE
    seed = seed if seed is not None else config.RANDOM_SEED
    rng = random.Random(seed)

    idx = list(range(len(cases)))
    rng.shuffle(idx)
    query_ids = set(idx[:n_query])

    index_cases = [c for i, c in enumerate(cases) if i not in query_ids]
    query_cases = [c for i, c in enumerate(cases) if i in query_ids]
    return index_cases, query_cases
