"""Schema 单测。"""
from src.schemas import Diagnosis


def test_diagnosis_defaults():
    d = Diagnosis(辨证结论="寒湿")
    assert d.病位 == []
    assert d.病性 == []
    assert d.病因 == []
    assert d.方药建议 == ""
    assert d.依据 == []


def test_diagnosis_roundtrip():
    d = Diagnosis(
        辨证结论="阳虚寒湿",
        病位=["手臂"],
        病性=["寒"],
        病因=["湿"],
        方药建议="参考温阳散寒方",
        依据=["case_id=12"],
    )
    dump = d.model_dump()
    assert dump["病位"] == ["手臂"]
    assert dump["依据"] == ["case_id=12"]
