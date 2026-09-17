"""Streamlit 界面：流式诊断 + 工具调用过程可视化 + 多轮对话。

运行：streamlit run src/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

import config
from src import prompts
from src.agent import build_agent
from src.data import build_documents, load_cases
from src.vectorstore import CaseStore, build_store, load_store


st.set_page_config(page_title="中医辨证 Agent", layout="wide")
st.title("🌿 中医辨证诊断 Agent（RAG + 工具调用）")


@st.cache_resource(show_spinner=False)
def _get_cases():
    return load_cases(config.DATA_PATH)


@st.cache_resource(show_spinner=False)
def _get_store():
    from pathlib import Path as P
    cases = _get_cases()
    if P(config.DB_DIR).exists():
        chroma = load_store(config.DB_DIR)
    else:
        chroma = build_store(build_documents(cases), config.DB_DIR)
    return CaseStore(chroma, cases)


def render_diagnosis(diag: dict):
    """把结构化诊断结果渲染成卡片。"""
    cols = st.columns(3)
    for col, dim in zip(cols, ("病位", "病性", "病因")):
        with col:
            st.markdown(f"**{dim}**")
            st.markdown("、".join(diag.get(dim, [])) or "—")

    st.markdown("### 🩺 辨证结论")
    st.markdown(diag.get("辨证结论", "") or "—")

    if diag.get("方药建议"):
        st.markdown("### 💊 方药建议")
        st.markdown(diag.get("方药建议"))

    if diag.get("依据"):
        with st.expander("📎 证据溯源（引用病历）", expanded=False):
            for ref in diag.get("依据"):
                st.markdown(f"- {ref}")


def main():
    with st.sidebar:
        st.header("⚙️ 系统设置")
        if st.button("🚀 初始化知识库 + Agent"):
            if not config.LLM_API_KEY:
                st.error("未配置 LLM_API_KEY，请在 .env 中填写。")
            else:
                with st.spinner("正在加载病历与向量库（首次需下载 embedding 模型）…"):
                    store = _get_store()
                    agent, _ = build_agent(store)
                    st.session_state["store"] = store
                    st.session_state["agent"] = agent
                    st.session_state["thread_id"] = "session-1"
                    st.session_state["initialized"] = True
                    st.session_state["messages"] = []
                st.success("初始化完成，可以开始辨证了。")

        st.caption(f"模型：{config.LLM_MODEL} | 检索 top-{config.RETRIEVAL_K}")

    # 免责声明
    st.caption("⚠️ 本系统仅供学习演示，输出不构成任何医疗建议，请勿用于真实诊疗。")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 历史消息回显
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], dict):
                render_diagnosis(msg["content"])
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input("请输入患者的症状描述，例如：长期失眠，伴有心烦潮热…"):
        agent = st.session_state.get("agent")
        if not agent:
            st.error("请先在左侧点击『初始化知识库 + Agent』。")
            return

        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # 构造本轮输入：首轮带 system，后续轮复用 checkpoint 记忆
            msgs = []
            if len(st.session_state["messages"]) == 1:
                msgs.append(SystemMessage(content=prompts.AGENT_SYSTEM_NO_ASK))
            msgs.append(HumanMessage(content=prompt))

            status = st.status("中医辨证 Agent 正在分析…", expanded=True)
            steps = []
            final = {}
            try:
                for chunk in agent.stream(
                    {"messages": msgs},
                    config={"configurable": {"thread_id": st.session_state["thread_id"]}},
                    stream_mode="updates",
                ):
                    for node, update in chunk.items():
                        if node == "agent":
                            for m in update.get("messages", []):
                                for tc in getattr(m, "tool_calls", None) or []:
                                    steps.append(f"🔧 调用工具 `{tc['name']}`")
                                    status.write(f"🔧 调用工具 `{tc['name']}`")
                        elif node == "tools":
                            steps.append("📚 已获取检索结果")
                            status.write("📚 已获取检索结果")
                        elif node == "format":
                            final = update.get("final_answer", {})
                            status.write("✅ 已生成结构化诊断")
            except Exception as e:
                status.update(label="分析出错", state="error")
                st.error(str(e))
                return
            status.update(label="分析完成", state="complete", expanded=False)

            render_diagnosis(final)
            st.session_state["messages"].append({"role": "assistant", "content": final})


if __name__ == "__main__":
    main()
