"""数据层：加载 / 规范化 / 逐条病历切块（带元数据）。

原始 JSON 结构（每条病历）：
    {
      "已知信息": {"性别", "年龄", "舌象", "面象", "脉象", "辨证结论"},
      "主诉": str,
      "对话流": [{"role": "user"/"AIMessage", "content": str}, ...],
      "辨证证据链": {"病位证据": [...], "病性证据": [...], "病因证据": [...]}
    }

说明：相比原「完整.py」把所有病历拼接后再切块（会跨病历边界切错），
这里改为「一病一档」：每条病历作为一个 Document，主诉作为检索主体，
诊断结论与证据链作为元数据，供检索过滤与评测使用。
"""
import json
import os
from typing import Any

from langchain_core.documents import Document

# 证据链三个维度（与原始 JSON 的子键一一对应）
EVIDENCE_DIMS = ("病位证据", "病性证据", "病因证据")


def load_cases(data_path: str) -> list[dict[str, Any]]:
    """读取原始 JSON 并规范化，返回带 case_id 的病历列表。"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在：{data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cases: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        known = item.get("已知信息", {}) or {}
        evidence = item.get("辨证证据链", {}) or {}
        cases.append({
            "case_id": i,
            "主诉": item.get("主诉", ""),
            "性别": known.get("性别", ""),
            "年龄": known.get("年龄", ""),
            "舌象": known.get("舌象", ""),
            "面象": known.get("面象", ""),
            "脉象": known.get("脉象", ""),
            "辨证结论": known.get("辨证结论", ""),
            "病位证据": list(evidence.get("病位证据", [])),
            "病性证据": list(evidence.get("病性证据", [])),
            "病因证据": list(evidence.get("病因证据", [])),
            "对话流": list(item.get("对话流", [])),
        })
    return cases


def _format_dialogue(dialogue: list[dict[str, Any]], max_turns: int = 6) -> str:
    """把对话流格式化为文本，只取前若干轮避免过长。"""
    lines = []
    for msg in dialogue[:max_turns]:
        role = "患者" if msg.get("role") == "user" else "医生"
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}：{content}")
    return "\n".join(lines)


def case_to_document(case: dict[str, Any]) -> Document:
    """一条病历 -> 一个 Document。

    page_content 以「主诉」为主体（评测时 query 即主诉，检索信号最干净），
    关键结论与证据链全部进 metadata，供元数据过滤与评测复用。
    """
    page_content = f"主诉：{case['主诉']}\n" + _format_dialogue(case["对话流"])

    metadata = {
        "case_id": int(case["case_id"]),
        "辨证结论": case["辨证结论"],
        "病位": "、".join(case["病位证据"]),
        "病性": "、".join(case["病性证据"]),
        "病因": "、".join(case["病因证据"]),
        "性别": case["性别"],
        "年龄": case["年龄"],
        "舌象": case["舌象"],
        "面象": case["面象"],
        "脉象": case["脉象"],
    }
    return Document(page_content=page_content, metadata=metadata)


def build_documents(cases: list[dict[str, Any]]) -> list[Document]:
    """批量转 Document。"""
    return [case_to_document(c) for c in cases]
