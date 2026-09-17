"""向量库封装：Chroma 建库/检索/元数据过滤 + 病历仓储。"""
import json
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from src.embeddings import get_embeddings


def build_store(documents: list[Document], persist_dir: str | None = None) -> Chroma:
    """离线建库：persist_dir 为 None 时用内存（临时）向量库。"""
    kwargs = {"embedding": get_embeddings()}
    if persist_dir:
        kwargs["persist_directory"] = persist_dir
    return Chroma.from_documents(documents=documents, **kwargs)


def load_store(persist_dir: str) -> Chroma:
    """加载已持久化的向量库。"""
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
    )


class CaseStore:
    """病历仓储：封装向量库检索 + 元数据过滤 + 完整病历详情。

    同时持有 case_id -> 完整病历 的内存映射，供 get_case_detail 使用。
    """

    def __init__(self, chroma: Chroma, cases: list[dict[str, Any]]):
        self._chroma = chroma
        self._by_id = {int(c["case_id"]): c for c in cases}

    @staticmethod
    def _to_item(doc: Document) -> dict[str, Any]:
        m = doc.metadata
        return {
            "case_id": m.get("case_id"),
            "辨证结论": m.get("辨证结论", ""),
            "病位": m.get("病位", ""),
            "病性": m.get("病性", ""),
            "病因": m.get("病因", ""),
            "片段": doc.page_content[:200],
        }

    def retrieve(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        k = k or config.RETRIEVAL_K
        docs = self._chroma.similarity_search(query, k=k)
        return [self._to_item(d) for d in docs]

    def retrieve_with_scores(self, query: str, k: int | None = None):
        """返回 (Document, distance) 列表，供评测使用。"""
        k = k or config.RETRIEVAL_K
        return self._chroma.similarity_search_with_score(query, k=k)

    @staticmethod
    def _case_to_item(case: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_id": case["case_id"],
            "辨证结论": case["辨证结论"],
            "病位": "、".join(case["病位证据"]),
            "病性": "、".join(case["病性证据"]),
            "病因": "、".join(case["病因证据"]),
            "片段": case["主诉"][:200],
        }

    def filter_by(self, dim: str, keyword: str, k: int | None = None) -> list[dict[str, Any]]:
        """按证候维度（病位/病性/病因）关键词过滤病历。

        直接在内存证据链上做子串匹配，不依赖 Chroma 的 where 过滤
        （chromadb 的 $contains 对字符串不做子串匹配）。
        """
        if dim not in ("病位", "病性", "病因"):
            raise ValueError(f"未知维度：{dim}，可选 病位/病性/病因")
        key = f"{dim}证据"
        k = k or config.RETRIEVAL_K
        matches = [c for c in self._by_id.values() if any(keyword in item for item in c[key])]
        return [self._case_to_item(c) for c in matches[:k]]

    def get_detail(self, case_id: int) -> str:
        """返回某条病历的完整诊疗对话文本。"""
        case = self._by_id.get(int(case_id))
        if not case:
            return f"未找到 case_id={case_id} 的病历。"
        lines = [
            f"【case_id】{case['case_id']}",
            f"【主诉】{case['主诉']}",
            f"【辨证结论】{case['辨证结论']}",
            f"【病位】{'、'.join(case['病位证据'])}",
            f"【病性】{'、'.join(case['病性证据'])}",
            f"【病因】{'、'.join(case['病因证据'])}",
            "【诊疗对话】",
        ]
        for msg in case["对话流"]:
            role = "患者" if msg.get("role") == "user" else "医生"
            content = str(msg.get("content", "")).strip()
            if content:
                lines.append(f"{role}：{content}")
        return "\n".join(lines)

    def to_json(self) -> str:
        """把完整病历映射序列化（用于 build_index 落地 case 详情缓存）。"""
        return json.dumps(self._by_id, ensure_ascii=False)
