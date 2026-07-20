# -*- coding: utf-8 -*-
"""
保护渣设计 Agent (LangGraph + OpenAI function-calling)
=======================================================
把 src/agent_tools.py 里的四套科学计算后端，编排成一个能用自然语言驱动的
多步智能体。

状态图 (LangGraph StateGraph):
    START → planner ─┬─ 有 tool_calls ──→ executor ──→ planner  (循环，最多 MAX_STEPS 轮)
                     └─ 无 tool_calls ──→ END
  • planner:   调 LLM，传入工具清单 + 对话历史 + (注入式)RAG 上下文，返回 assistant 消息
  • executor:  执行 assistant 的 tool_calls，把结构化结果以 role=tool 喂回

两种运行模式:
  • 在线: 配置 OPENAI_API_KEY → 真 LLM function-calling 多步 Agent
  • 离线: 无 key → 规则路由（关键词→工具），仍可端到端演示工具链

配置 (环境变量):
  OPENAI_API_KEY    在线模式必填
  OPENAI_BASE_URL   可选，接 OpenAI 兼容端点（DeepSeek / Moonshot / 自建 vLLM）
  AGENT_LLM_MODEL   可选，默认 gpt-4o-mini
"""
import os
import sys
import re
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from src import agent_tools as T
from src.agent_rag import retrieve, mode as rag_mode

MODEL = os.environ.get("AGENT_LLM_MODEL", "gpt-4o-mini")
MAX_STEPS = 6

SYSTEM_PROMPT = """你是「保护渣智能设计助手」，服务钢铁连铸保护渣的研发。
可调用以下工具（科学计算后端）：
- predict_properties(composition): 成分 → 熔点 T_hem(°C) / 1300°C 黏度(Pa·s) / 碱度 R
- predict_phase_diagram(composition): 成分 → 液相50%温度 / 完全液化温度 / 峰值液相 / 主要结晶相
- inverse_design(target_T_hem_C?, target_viscosity?, target_basicity_R?): 目标性质 → 反推成分配方
- lookup_phase(query): 检索保护渣相/概念的化学式、矿物名与机理

工作守则：
1. 成分是氧化物质量百分比，缺失的氧化物按 0，会自动归一化到 100%。
2. 涉及某相或概念的解释，先调 lookup_phase 取准确知识再答，不要凭空编化学式。
3. 做完反向设计后，用 predict_properties 复算验证并报告误差。
4. 回答用中文，数字带单位（°C、Pa·s），附简短的机理/工程含义。
5. 诚实标注不确定性（模型 5-fold CV R²≈0.70，熔点有 ±30°C 量级误差）。
"""


class AgentState(TypedDict):
    messages: list       # OpenAI chat 格式
    trace: list          # 工具调用轨迹，供 UI / 评测展示


def _client():
    from openai import OpenAI
    return OpenAI()      # 自动读 OPENAI_API_KEY / OPENAI_BASE_URL


def _inject_rag(user_text):
    """把与用户问题相关的知识条目注入系统提示（轻量 RAG，降低术语幻觉）。"""
    docs = retrieve(user_text, k=2)
    body = "\n".join(f"- {d['text']}" for d in docs if d["score"] > 0.05)
    return "\n\n[相关领域知识（检索自知识库，仅供参考，勿照抄）]\n" + body if body else ""


# ── 图节点 ────────────────────────────────────────────────────────────────────
def planner(state):
    resp = _client().chat.completions.create(
        model=MODEL, messages=state["messages"],
        tools=T.TOOL_SCHEMAS, tool_choice="auto", temperature=0.2)
    am = resp.choices[0].message.model_dump()
    return {"messages": state["messages"] + [am]}


def executor(state):
    last = state["messages"][-1]
    messages = list(state["messages"])
    trace = list(state.get("trace", []))
    for tc in last.get("tool_calls") or []:
        name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"] or "{}")
        try:
            out = T.DISPATCH[name](**args)
            content = json.dumps(out, ensure_ascii=False)
        except Exception as e:  # 工具执行失败也以结构化错误喂回，让 LLM 自行处理
            out = {"error": f"{type(e).__name__}: {e}"}
            content = json.dumps(out, ensure_ascii=False)
        trace.append({"tool": name, "args": args, "result": out})
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
    return {"messages": messages, "trace": trace}


def _route(state):
    last = state["messages"][-1]
    return "executor" if last.get("tool_calls") else END


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", _route)
    g.add_edge("executor", "planner")
    return g.compile()


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ════════════════════════════════════════════════════════════════════════════
# 离线「规则路由」模式 (无 OPENAI_API_KEY 时使用，演示完整工具链)
# ════════════════════════════════════════════════════════════════════════════
_OX_RE = re.compile(r"([A-Za-z0-9O₂₃]+)\s*[=＝:]\s*([0-9.]+)")


def _parse_comp(text):
    comp, found = {}, False
    for ox, val in _OX_RE.findall(text):
        ox = ox.replace("₂", "2").replace("₃", "3")
        if ox in T.COMP_COLS:
            comp[ox] = float(val)
            found = True
    return comp if found else None


def _mock_route(user_text):
    """关键词路由 → 直接调对应工具，拼一个结构化回复。不调用任何 LLM。"""
    trace = []
    comp = _parse_comp(user_text)
    m_t = re.search(r"(?:熔点|hem)[^0-9]{0,6}([0-9]{3,4})", user_text)
    m_v = re.search(r"[黏粘]度[^0-9]{0,6}([0-9.]+)", user_text)
    m_r = re.search(r"碱度[^0-9]{0,8}([0-9.]+)", user_text)
    want_inv = any(k in user_text for k in ["设计", "反推", "配方", "推荐", "目标"]) or (m_t and ("设计" in user_text or "配方" in user_text))

    if want_inv:
        kw = {}
        if m_t:
            kw["target_T_hem_C"] = float(m_t.group(1))
        if m_v:
            kw["target_viscosity"] = float(m_v.group(1))
        if m_r:
            kw["target_basicity_R"] = float(m_r.group(1))
        if not kw:
            kw["target_T_hem_C"] = 1050.0
        out = T.inverse_design(**kw)
        trace.append({"tool": "inverse_design", "args": kw, "result": out})
        chk = T.predict_properties(out["composition_wt_pct"])
        trace.append({"tool": "predict_properties", "args": {"composition": out["composition_wt_pct"]}, "result": chk})
        top = ", ".join(f"{k}={v}" for k, v in out["composition_wt_pct"].items() if v >= 1.0)
        reply = (f"按目标反推的保护渣配方（主要组分，wt%）：{top}\n\n"
                 f"复算验证：T_hem={chk['T_hem_C']} °C，1300 °C 黏度={chk['viscosity_1300C_Pa_s']} Pa·s，碱度 R={chk['basicity_R']}。\n"
                 f"（优化残差={out['optimization_final_loss']}；模型有 ±30 °C 量级误差，建议实测复核。）")
    elif comp:
        out = T.predict_properties(comp)
        trace.append({"tool": "predict_properties", "args": {"composition": comp}, "result": out})
        ph = T.predict_phase_diagram(comp)
        trace.append({"tool": "predict_phase_diagram", "args": {"composition": comp}, "result": ph})
        tops = "、".join(p["phase"] for p in ph["top_solid_phases"][:3])
        reply = (f"成分预测：T_hem={out['T_hem_C']} °C，1300 °C 黏度={out['viscosity_1300C_Pa_s']} Pa·s，碱度 R={out['basicity_R']}。\n\n"
                 f"相图：液相50%温度≈{ph['liquid_50pct_T_C']} °C，完全液化温度≈{ph['complete_liquefaction_T_C']} °C；"
                 f"主要结晶相：{tops}。")
    else:
        out = T.lookup_phase(user_text)
        trace.append({"tool": "lookup_phase", "args": {"query": user_text}, "result": out})
        reply = "📚 知识检索结果：\n" + "\n".join(f"- {d['knowledge']}" for d in out["results"])
    return reply, trace


# ════════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════════
def run(user_text, history=None):
    """跑一轮 Agent。返回 dict(reply, trace, mode)。"""
    if not os.environ.get("OPENAI_API_KEY"):
        reply, trace = _mock_route(user_text)
        return {"reply": reply, "trace": trace, "mode": "规则路由(离线·无LLM)"}

    sys_msg = {"role": "system", "content": SYSTEM_PROMPT + _inject_rag(user_text)}
    messages = [sys_msg] + (history or []) + [{"role": "user", "content": user_text}]
    out = _graph().invoke({"messages": messages, "trace": []},
                          config={"recursion_limit": 4 * MAX_STEPS})
    last = out["messages"][-1]
    reply = last.get("content") or "(模型未返回文本，请查看下方工具调用轨迹)"
    return {"reply": reply, "trace": out.get("trace", []),
            "mode": f"LLM function-calling · {MODEL} · RAG={rag_mode()}"}


if __name__ == "__main__":
    for q in ["SiO2=30, Al2O3=5, CaO=35, Na2O=8, Li2O=3, F=7, MgO=2 的熔点和黏度",
              "霞石是什么相？有什么作用？"]:
        print("\n" + "=" * 70)
        print("Q:", q)
        r = run(q)
        print("MODE :", r["mode"])
        print("REPLY:", r["reply"])
        for tr in r["trace"]:
            print("  ·", tr["tool"], "->", json.dumps(tr["result"], ensure_ascii=False)[:100])
