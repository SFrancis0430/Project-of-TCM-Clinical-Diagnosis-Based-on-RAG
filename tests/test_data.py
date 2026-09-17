"""数据层单测：不依赖大文件，用合成病历验证切块与元数据。"""
from src.data import EVIDENCE_DIMS, build_documents, case_to_document


def make_case(case_id=0):
    return {
        "case_id": case_id,
        "主诉": "手臂麻木、肩痛、畏寒",
        "性别": "男",
        "年龄": "中年",
        "舌象": "苔白",
        "面象": "面色偏暗",
        "脉象": "脉沉细",
        "辨证结论": "阳虚寒湿，营卫不和",
        "病位证据": ["手臂麻木", "肩痛"],
        "病性证据": ["畏寒", "怕风"],
        "病因证据": ["寒湿"],
        "对话流": [
            {"role": "user", "content": "医生，我手臂发麻。"},
            {"role": "AIMessage", "content": "多久了？"},
        ],
    }


def test_evidence_dims():
    assert EVIDENCE_DIMS == ("病位证据", "病性证据", "病因证据")


def test_case_to_document_metadata():
    doc = case_to_document(make_case())
    assert doc.metadata["case_id"] == 0
    assert doc.metadata["辨证结论"] == "阳虚寒湿，营卫不和"
    assert doc.metadata["病位"] == "手臂麻木、肩痛"
    assert doc.metadata["性别"] == "男"
    assert doc.metadata["舌象"] == "苔白"
    assert doc.metadata["脉象"] == "脉沉细"
    assert "主诉" in doc.page_content
    assert "患者：医生，我手臂发麻" in doc.page_content


def test_build_documents_len():
    docs = build_documents([make_case(0), make_case(1)])
    assert len(docs) == 2
    assert docs[1].metadata["case_id"] == 1
