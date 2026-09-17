"""LangGraph 诊断 Agent：ReAct 循环 + 工具节点 + 结构化输出节点。

状态机：
    START -> agent -> (有工具调用 ? tools -> agent : format) -> END

其中 agent 节点由 LLM 决定调用哪些工具；format 节点用
with_structured_output 强制输出结构化 Diagnosis，便于评测。
"""
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src import prompts
from src.llm import get_llm, structured_invoke
from src.schemas import Diagnosis
from src.tools import make_ask_patient_tool, make_tools
from src.vectorstore import CaseStore


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    final_answer: dict[str, Any]


def build_agent(store: CaseStore, patient_case: dict | None = None):
    """构建并编译 Agent，返回 (compiled_graph, tools)。

    patient_case 非空时，Agent 额外获得「向患者追问」工具，可先多轮问诊、
    补全四诊信息，再结合检索病历给出辨证（体现 Agent 相对单轮 RAG 的优势）。
    """
    tools = make_tools(store)
    if patient_case is not None:
        tools = tools + [make_ask_patient_tool(patient_case)]
    model = get_llm(temperature=0.0).bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    tools_node = ToolNode(tools)

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "format"

    def format_node(state: AgentState) -> dict:
        # 从对话历史中提取干净信息，避免把混杂的工具调用历史直接喂给结构化 LLM
        # （DeepSeek 会因此混淆工具，导致结构化输出失败）。
        user_input = next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")
        asked = [m.content for m in state["messages"] if isinstance(m, ToolMessage) and m.name == "ask_patient"]
        evidence = [m.content for m in state["messages"] if isinstance(m, ToolMessage) and m.name != "ask_patient"]

        parts = [f"患者主诉：{user_input}"]
        if asked:
            parts.append("问诊获得的信息：\n" + "\n".join(asked))
        if evidence:
            parts.append("检索到的病历证据：\n" + "\n---\n".join(evidence))

        msgs = [SystemMessage(content=prompts.FORMAT_SYSTEM), HumanMessage(content="\n\n".join(parts))]
        result: Diagnosis = structured_invoke(Diagnosis, msgs)
        return {"final_answer": result.model_dump()}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("format", format_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "format": "format"})
    graph.add_edge("tools", "agent")
    graph.add_edge("format", END)

    return graph.compile(checkpointer=MemorySaver()), tools


def ask(
    agent,
    question: str,
    thread_id: str = "default",
    history: list[BaseMessage] | None = None,
    system_prompt: str | None = None,
) -> dict:
    """单次提问，返回结构化诊断结果（dict）。

    system_prompt 默认用带问诊指令的 AGENT_SYSTEM；无问诊场景传 AGENT_SYSTEM_NO_ASK。
    """
    messages: list[BaseMessage] = []
    if history is None or not history:
        messages.append(SystemMessage(content=system_prompt or prompts.AGENT_SYSTEM))
    else:
        messages.extend(history)
    messages.append(HumanMessage(content=question))

    result = agent.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result.get("final_answer", {})
