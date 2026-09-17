"""结构化输出模型：诊断结果。

字段与原始数据的「辨证证据链」（病位/病性/病因）对齐，
从而让 Agent 的输出可以被自动评测（对比金标准标签）。
"""
from pydantic import BaseModel, Field


class Diagnosis(BaseModel):
    """中医辨证诊断结构化结果。"""

    辨证结论: str = Field(description="综合辨证结论，如『阳虚寒湿，营卫不和』")
    病位: list[str] = Field(default_factory=list, description="病位，如手臂、颈项")
    病性: list[str] = Field(default_factory=list, description="病性，如寒、湿、虚")
    病因: list[str] = Field(default_factory=list, description="病因，如风、痰、瘀")
    方药建议: str = Field(default="", description="建议方药倾向（仅供学习演示）")
    依据: list[str] = Field(default_factory=list, description="引用的病历 case_id 或原文片段")
