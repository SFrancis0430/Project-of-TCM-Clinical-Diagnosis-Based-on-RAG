"""LLM-as-judge：用 DeepSeek 对「结论一致性 / 忠实度」做结构化打分。"""
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.llm import structured_invoke


class Verdict(BaseModel):
    """评判结果。"""

    score: float = Field(description="0 到 1 的分数")
    reason: str = Field(description="判定理由，一句话")


def _judge(system: str, user: str) -> Verdict:
    return structured_invoke(Verdict, [SystemMessage(content=system), HumanMessage(content=user)])


def judge_conclusion(pred: str, gold: str) -> Verdict:
    """评估预测辨证结论与金标准结论的语义一致性。"""
    system = (
        "你是中医辨证评估专家。请评估「预测辨证结论」与「金标准结论」的语义一致性，"
        "给出 0~1 分数（1 表示完全一致，0 表示完全无关）。只输出分数和理由。"
    )
    user = f"预测辨证结论：{pred}\n金标准结论：{gold}"
    return _judge(system, user)


def judge_faithfulness(answer: str, context: str) -> Verdict:
    """评估回答是否忠实于检索到的病历证据（不臆造）。"""
    system = (
        "你是 RAG 忠实度评估专家。请评估「回答」中的辨证结论与方药建议是否都能"
        "被「检索到的病历证据」支撑，给出 0~1 分数（1 表示完全有据可依，"
        "0 表示存在明显编造）。只输出分数和理由。"
    )
    user = f"检索到的病历证据：\n{context}\n\n回答：\n{answer}"
    return _judge(system, user)
