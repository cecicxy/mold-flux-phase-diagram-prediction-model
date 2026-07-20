# -*- coding: utf-8 -*-
"""
保护渣设计 Agent —— 自然语言对话界面 (Streamlit)
=================================================
启动: .venv/bin/streamlit run app_agent.py
  • 配了 OPENAI_API_KEY   → 真 LLM function-calling 多步 Agent (LangGraph)
  • 没配                  → 规则路由离线模式，仍可端到端演示工具链

和 app_phase_diagram.py 互补：
  app_phase_diagram.py = 表单式精确输入（适合工程师确定性查相图）
  app_agent.py         = 对话式（适合"帮我设计一个…"这类模糊、多步需求）
"""
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import streamlit as st
from src import agent

st.set_page_config(page_title="保护渣设计 Agent", page_icon="🤖", layout="wide")


# ── 密码保护（云端部署用，Secrets 里设 APP_PASSWORD；本地不设则不锁）─────────
def check_password():
    import hmac
    try:
        configured = st.secrets.get("APP_PASSWORD", "")
    except (FileNotFoundError, KeyError):
        configured = ""
    if not configured:
        return True
    if st.session_state.get("_authenticated"):
        return True
    st.title("🔐 保护渣设计 Agent")
    st.caption("此应用受密码保护，请输入访问密码。")
    with st.form("login_form"):
        entered = st.text_input("访问密码", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("进入应用", type="primary")
    if submitted:
        if hmac.compare_digest(str(entered), str(configured)):
            st.session_state["_authenticated"] = True
            st.rerun()
        else:
            st.error("密码错误，请重试。")
    return False


if not check_password():
    st.stop()


# ── LLM 配置：优先读 Streamlit Secrets，回退环境变量（云端在此注入 key）───────
for _k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "AGENT_LLM_MODEL"):
    try:
        _v = st.secrets.get(_k)
    except (FileNotFoundError, KeyError):
        _v = None
    if _v:
        os.environ.setdefault(_k, str(_v))


st.title("🤖 保护渣智能设计 Agent")
st.caption("自然语言驱动：成分性质预测 · 相图推理 · 反向配方设计 · 领域知识问答")

# ── 侧栏：运行状态 / 配置说明 / 示例 ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙ 运行状态")
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    st.metric("LLM", "已接入" if has_key else "未配置 Key")
    if has_key:
        st.caption(f"模型：{os.environ.get('AGENT_LLM_MODEL', 'gpt-4o-mini')}")
        st.caption(f"端点：{os.environ.get('OPENAI_BASE_URL', 'OpenAI 官方')}")
    else:
        st.warning("未检测到 OPENAI_API_KEY，当前为**规则路由离线模式**（不调 LLM，"
                   "按关键词选工具）。配置后即切换为真正的 function-calling Agent。")
        with st.expander("如何接入 LLM"):
            st.code("export OPENAI_API_KEY=sk-xxxx\n"
                    "# 可选：OpenAI 兼容端点（DeepSeek / Moonshot / 自建 vLLM）\n"
                    "# export OPENAI_BASE_URL=https://api.deepseek.com/v1\n"
                    "# export AGENT_LLM_MODEL=deepseek-chat", language="bash")
    try:
        from src.agent_rag import mode as rag_mode
        st.caption(f"RAG 检索：{rag_mode()}")
    except Exception:
        pass

    st.divider()
    st.markdown("**可问示例**")
    examples = [
        "SiO2=30, Al2O3=5, CaO=35, Na2O=8, Li2O=3, F=7, MgO=2 的熔点和黏度是多少？",
        "帮我设计一个熔点 1050°C、碱度 0.9 的保护渣配方",
        "霞石是什么相？有什么作用？",
        "SiO2=33, CaO=30, Na2O=10 的相图，主要结晶相有哪些？",
        "怎么降低保护渣的熔点？",
    ]
    for i, ex in enumerate(examples):
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            hist = st.session_state.get("history", [])
            r = agent.run(ex, history=[{"role": m["role"], "content": m["content"]} for m in hist])
            hist.append({"role": "user", "content": ex})
            hist.append({"role": "assistant", "content": r["reply"], "trace": r["trace"], "mode": r["mode"]})
            st.session_state["history"] = hist
            st.rerun()

# ── 对话历史 ──────────────────────────────────────────────────────────────────
st.session_state.setdefault("history", [])

for m in st.session_state["history"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("trace"):
            with st.expander(f"🔧 工具调用轨迹（{len(m['trace'])} 步）· 模式: {m.get('mode', '')}"):
                for i, tr in enumerate(m["trace"], 1):
                    st.markdown(f"**步骤 {i}：`{tr['tool']}`**")
                    st.json({"args": tr["args"], "result": tr["result"]})

# ── 输入框 ────────────────────────────────────────────────────────────────────
if user := st.chat_input("问点关于保护渣的……（成分预测 / 相图 / 反向设计 / 知识问答）"):
    st.session_state["history"].append({"role": "user", "content": user})
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        with st.spinner("Agent 思考中…"):
            hist = [{"role": m["role"], "content": m["content"]}
                    for m in st.session_state["history"][:-1]]
            r = agent.run(user, history=hist)
        st.markdown(r["reply"])
        if r["trace"]:
            with st.expander(f"🔧 工具调用轨迹（{len(r['trace'])} 步）· 模式: {r['mode']}"):
                for i, tr in enumerate(r["trace"], 1):
                    st.markdown(f"**步骤 {i}：`{tr['tool']}`**")
                    st.json({"args": tr["args"], "result": tr["result"]})
    st.session_state["history"].append(
        {"role": "assistant", "content": r["reply"], "trace": r["trace"], "mode": r["mode"]})
