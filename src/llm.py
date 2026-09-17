"""LLM 工厂：DeepSeek（OpenAI 兼容）统一入口，可切换 provider。"""
from functools import lru_cache
from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

import config


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """返回 LLM 实例（模块级缓存，避免重复创建连接）。"""
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_API_BASE,
        temperature=temperature,
    )


def get_structured_llm(schema: type, temperature: float = 0.0):
    """返回带结构化输出的 LLM。

    强制 method="function_calling"：DeepSeek 的 OpenAI 兼容接口不支持
    langchain-openai 默认的 response_format=json_schema 模式，但支持
    function calling（工具调用），故统一走工具调用实现结构化输出。
    """
    return get_llm(temperature).with_structured_output(schema, method="function_calling")


def structured_invoke(schema: type, messages: Sequence[BaseMessage], retries: int = 2):
    """结构化输出调用 + 重试。

    DeepSeek 的 function calling 在无工具调用上下文的「冷启动」场景下，
    偶尔会返回普通文本而非工具调用，导致 with_structured_output 返回 None。
    这里在返回 None 时追加强化指令重试。
    """
    structured = get_structured_llm(schema)
    msgs = list(messages)
    for _ in range(retries + 1):
        result = structured.invoke(msgs)
        if result is not None:
            return result
        msgs = msgs + [HumanMessage(content="必须通过调用函数返回结构化结果，禁止输出普通文本。")]
    raise RuntimeError(f"结构化输出连续 {retries + 1} 次返回 None")
