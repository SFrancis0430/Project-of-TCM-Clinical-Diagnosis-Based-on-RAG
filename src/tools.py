"""工具定义：诊断 Agent 可调用的三个工具。

工具是「Agent vs 单轮 RAG」的核心差异——LLM 能自主决定
调用哪些工具、调用几次，形成多步推理，而不是一次性检索。
"""
import json

from langchain_core.tools import tool

from src.vectorstore import CaseStore


def make_tools(store: CaseStore):
    """绑定到一个具体 CaseStore，返回工具列表。"""

    @tool
    def retrieve_cases(query: str) -> str:
        """根据患者症状/主诉，语义检索最相似的历史病历。

        Args:
            query: 症状或主诉描述，例如「手臂麻木、肩痛、畏寒」。
        """
        items = store.retrieve(query)
        if not items:
            return "未检索到相似病历。"
        return json.dumps(items, ensure_ascii=False, indent=2)

    @tool
    def get_case_detail(case_id: int) -> str:
        """查看某条病历的完整诊疗对话，用于确认辨证依据。

        Args:
            case_id: 病历编号（来自 retrieve_cases / filter_cases 返回结果）。
        """
        return store.get_detail(case_id)

    @tool
    def filter_cases(dim: str, keyword: str) -> str:
        """按证候维度过滤病历。

        Args:
            dim: 维度，取值「病位」「病性」「病因」之一。
            keyword: 该维度下的关键词，例如 dim=病性、keyword=寒湿。
        """
        try:
            items = store.filter_by(dim, keyword)
        except ValueError as e:
            return str(e)
        if not items:
            return f"未找到 {dim} 含「{keyword}」的病历。"
        return json.dumps(items, ensure_ascii=False, indent=2)

    return [retrieve_cases, get_case_detail, filter_cases]


# 追问工具的四诊字段与关键词映射
_FIELD_KEYWORDS = {
    "舌象": ["舌", "苔"],
    "脉象": ["脉"],
    "面象": ["面", "脸", "色"],
    "性别": ["性别", "男女"],
    "年龄": ["年龄", "岁"],
}


def make_ask_patient_tool(case: dict):
    """构造「向患者追问」工具，绑定到具体病例的已知信息 + 对话。

    追问答案只来自该病例的 性别/年龄/舌象/面象/脉象 与对话流中患者的陈述，
    绝不返回「辨证结论」（金标准），避免评测作弊。
    """
    info = {
        "性别": case.get("性别", ""),
        "年龄": case.get("年龄", ""),
        "舌象": case.get("舌象", ""),
        "面象": case.get("面象", ""),
        "脉象": case.get("脉象", ""),
    }
    patient_says = [
        str(m.get("content", "")).strip()
        for m in case.get("对话流", [])
        if m.get("role") == "user" and str(m.get("content", "")).strip()
    ]

    @tool
    def ask_patient(question: str) -> str:
        """向患者追问四诊信息（舌象/脉象/面象/性别/年龄/寒热/睡眠/饮食等），
        用于补全辨证依据。当患者主诉不足以辨证时，先调用本工具问诊。

        Args:
            question: 想问患者的问题，如「舌象如何」「是否怕冷」。
        """
        hits = []
        for field, kws in _FIELD_KEYWORDS.items():
            if info.get(field) and any(k in question for k in kws):
                hits.append(f"{field}：{info[field]}")
        if not hits:
            # 未命中固定四诊字段时，返回患者此前陈述中相关的前几条
            hits = [f"患者：{s}" for s in patient_says[:3]]
        return "\n".join(hits) or "患者未提供相关信息。"

    return ask_patient
