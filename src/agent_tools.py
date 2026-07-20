# -*- coding: utf-8 -*-
"""
保护渣 Agent 的「工具层」(Tool layer)
======================================
把项目里三套科学计算能力封装成 LLM 可调用的结构化工具:

  1. predict_properties    成分 → 熔点 T_hem / 黏度 η / 碱度 R   (CatBoost 正向模型)
  2. predict_phase_diagram 成分 → 液相50%温度 / 完全液化温度 / 主要相 (MLP 相图代理)
  3. inverse_design        目标性质 → 反推成分配方 (差分进化优化)
  4. lookup_phase          相化学式 / 矿物学 / 机理 检索 (RAG, 见 src/agent_rag.py)

每个工具 = (a) OpenAI function-calling 用的 JSON Schema + (b) Python 执行函数。
Agent 编排 (LangGraph + LLM tool-calling) 见 src/agent.py。

设计原则:
  • 输入只接受成分 dict；绝不把 T_exp / viscosity 标签当输入特征 (防泄漏)。
  • 成分先归一化到 100%，与代理模型训练分布一致。
  • 模型懒加载并缓存，首次调用才 load pkl。
"""
import os
import numpy as np
import joblib
from scipy.optimize import differential_evolution

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMP_COLS = ["SiO2", "Al2O3", "CaO", "MgO", "Na2O", "Li2O", "F", "B2O3",
             "TiO2", "MnO", "Fe2O3", "BaO"]
FEATURE_COLS = COMP_COLS + ["R"]
TEMPS = np.arange(800, 1501, 5)

# 反向设计搜索边界 (典型保护渣 wt% 区间)
_BOUNDS = [(20, 45), (0, 15), (20, 45), (0, 8), (0, 15), (0, 8),
           (0, 12), (0, 8), (0, 5), (0, 5), (0, 8), (0, 5)]

_cache = {}


# ── 模型懒加载 ────────────────────────────────────────────────────────────────
def _forward():
    if "them" not in _cache:
        _cache["them"] = joblib.load(os.path.join(ROOT, "models", "forward_them.pkl"))
        _cache["visc"] = joblib.load(os.path.join(ROOT, "models", "forward_visc.pkl"))
    return _cache["them"], _cache["visc"]


def _surrogate():
    if "sur" not in _cache:
        _cache["sur"] = joblib.load(os.path.join(ROOT, "models", "phase_surrogate.pkl"))
        _cache["scaler"] = joblib.load(os.path.join(ROOT, "models", "phase_surrogate_scaler.pkl"))
        _cache["pnames"] = list(np.load(
            os.path.join(ROOT, "data", "phase_names_all.npy"), allow_pickle=True))
    return _cache["sur"], _cache["scaler"], _cache["pnames"]


# ── 成分归一化 / 特征构造 ─────────────────────────────────────────────────────
def _normalize(comp):
    tot = sum(float(comp.get(c, 0.0)) for c in COMP_COLS)
    return {c: (float(comp.get(c, 0.0)) / tot * 100 if tot > 0 else 0.0) for c in COMP_COLS}


def _feat13(comp):
    comp = _normalize(comp)
    R = comp["CaO"] / comp["SiO2"] if comp["SiO2"] > 0 else 0.0
    return np.array([comp[c] for c in COMP_COLS] + [R]), comp, R


# ════════════════════════════════════════════════════════════════════════════
# 工具 1: 正向性质预测
# ════════════════════════════════════════════════════════════════════════════
def predict_properties(composition: dict) -> dict:
    """成分 → 熔点 / 黏度 / 碱度。"""
    feat, comp, R = _feat13(composition)
    them, visc = _forward()
    x = feat.reshape(1, -1)
    return {
        "T_hem_C": round(float(them.predict(x)[0]), 1),
        "viscosity_1300C_Pa_s": round(float(visc.predict(x)[0]), 3),
        "basicity_R": round(float(R), 3),
        "composition_normalized_wt_pct": {c: round(comp[c], 2) for c in COMP_COLS},
    }


# ════════════════════════════════════════════════════════════════════════════
# 工具 2: 相图代理预测
# ════════════════════════════════════════════════════════════════════════════
def _liq_curve(Y, pnames):
    idx = [i for i, p in enumerate(pnames) if "IONIC_LIQ" in p]
    return Y[:, idx].sum(axis=1) if idx else np.zeros(Y.shape[0])


def _t_at_frac(liq, frac):
    i = np.where(liq >= frac)[0]
    if len(i) == 0:
        return None
    k = i[0]
    if k == 0:
        return float(TEMPS[0])
    return float(TEMPS[k - 1] + (frac - liq[k - 1]) / (liq[k] - liq[k - 1]) * (TEMPS[k] - TEMPS[k - 1]))


def _t_complete(liq, reach=0.80):
    """完全液化温度：液相达 80% 后曲线进入平台的膝点（到首末连线距离最大处）。"""
    i0 = np.where(liq >= reach)[0]
    if len(i0) == 0:
        return None
    s = i0[0]
    x, y = TEMPS[s:], liq[s:]
    if len(y) < 4:
        return float(x[-1])
    cx, cy = float(x[-1]) - x[0], float(y[-1]) - y[0]
    d = np.abs(cx * (y - y[0]) - cy * (x - x[0])) / np.hypot(cx, cy)
    return float(x[int(np.argmax(d))])


def predict_phase_diagram(composition: dict) -> dict:
    """成分 → 液相50%温度 / 完全液化温度 / 峰值液相 / 主要结晶相。"""
    feat, comp, R = _feat13(composition)
    sur, scaler, pnames = _surrogate()
    X = np.hstack([TEMPS.reshape(-1, 1), np.tile(feat, (len(TEMPS), 1))])
    Y = np.clip(sur.predict(scaler.transform(X)), 0, 1)
    s = Y.sum(axis=1, keepdims=True)
    Y = Y / np.where(s > 0, s, 1)
    liq = _liq_curve(Y, pnames)
    peaks = Y.max(axis=0)
    top = [i for i in np.argsort(-peaks) if peaks[i] > 0.02][:8]
    t50, tc = _t_at_frac(liq, 0.5), _t_complete(liq)
    return {
        "liquid_50pct_T_C": (round(t50) if t50 else None),
        "complete_liquefaction_T_C": (round(tc) if tc else None),
        "peak_liquid_fraction": round(float(liq.max()), 3),
        "basicity_R": round(float(R), 3),
        "top_solid_phases": [
            {"phase": pnames[i], "peak_fraction": round(float(peaks[i]), 3)}
            for i in top if "IONIC_LIQ" not in pnames[i]
        ],
    }


# ════════════════════════════════════════════════════════════════════════════
# 工具 3: 反向设计 (优化)
# ════════════════════════════════════════════════════════════════════════════
def inverse_design(target_T_hem_C: float = None,
                   target_viscosity: float = None,
                   target_basicity_R: float = None) -> dict:
    """给定目标性质 → 差分进化搜索成分配方（归一化到 100%）。至少给 1 个目标。"""
    them, visc = _forward()

    def obj(raw):
        raw = np.asarray(raw, dtype=float)
        if raw.sum() <= 0:
            return 1e6
        comp = raw / raw.sum() * 100
        R = comp[2] / comp[0] if comp[0] > 0 else 0.0
        x = np.concatenate([comp, [R]]).reshape(1, -1)
        loss = 0.0
        if target_T_hem_C is not None:
            loss += ((them.predict(x)[0] - target_T_hem_C) / 50.0) ** 2
        if target_viscosity is not None:
            loss += ((visc.predict(x)[0] - target_viscosity) / 0.3) ** 2
        if target_basicity_R is not None:
            loss += ((R - target_basicity_R) / 0.3) ** 2
        return float(loss)

    res = differential_evolution(obj, _BOUNDS, seed=42, maxiter=60,
                                 popsize=15, tol=1e-3, polish=True)
    comp = np.asarray(res.x, dtype=float)
    comp = comp / comp.sum() * 100
    R = comp[2] / comp[0] if comp[0] > 0 else 0.0
    x = np.concatenate([comp, [R]]).reshape(1, -1)
    return {
        "composition_wt_pct": {c: round(float(comp[i]), 2) for i, c in enumerate(COMP_COLS)},
        "predicted": {
            "T_hem_C": round(float(them.predict(x)[0]), 1),
            "viscosity_1300C_Pa_s": round(float(visc.predict(x)[0]), 3),
            "basicity_R": round(float(R), 3),
        },
        "optimization_final_loss": round(float(res.fun), 4),
    }


# ════════════════════════════════════════════════════════════════════════════
# 工具 4: 相知识检索 (RAG)
# ════════════════════════════════════════════════════════════════════════════
def lookup_phase(query: str, k: int = 3) -> dict:
    """自然语言检索保护渣相 / 概念知识（embedding 或 TF-IDF）。"""
    from src.agent_rag import retrieve
    docs = retrieve(query, k=k)
    return {"query": query, "results": [{"score": round(d["score"], 3),
            "category": d["cat"], "knowledge": d["text"]} for d in docs]}


# ════════════════════════════════════════════════════════════════════════════
# OpenAI function-calling 工具清单 (JSON Schema)
# ════════════════════════════════════════════════════════════════════════════
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "predict_properties",
            "description": "由保护渣化学成分预测半球熔化温度 T_hem (°C)、1300°C 黏度 (Pa·s)、碱度 R (CaO/SiO2)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "composition": {
                        "type": "object",
                        "description": "氧化物质量百分比；缺失的氧化物按 0 处理。会自动归一化到 100%。",
                        "properties": {c: {"type": "number"} for c in COMP_COLS},
                        "required": ["SiO2", "CaO"],
                    }},
                "required": ["composition"],
            }},
    },
    {
        "type": "function",
        "function": {
            "name": "predict_phase_diagram",
            "description": "由保护渣化学成分预测相图关键指标：液相50%温度、完全液化温度、峰值液相分数、主要结晶相。",
            "parameters": {
                "type": "object",
                "properties": {
                    "composition": {
                        "type": "object",
                        "description": "氧化物质量百分比。",
                        "properties": {c: {"type": "number"} for c in COMP_COLS},
                        "required": ["SiO2", "CaO"],
                    }},
                "required": ["composition"],
            }},
    },
    {
        "type": "function",
        "function": {
            "name": "inverse_design",
            "description": "反向设计：给定目标性质（熔点/黏度/碱度，至少一个），搜索满足目标的保护渣化学成分配方。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_T_hem_C": {"type": "number", "description": "目标半球熔化温度 (°C)，如 1050"},
                    "target_viscosity": {"type": "number", "description": "目标 1300°C 黏度 (Pa·s)，如 0.2"},
                    "target_basicity_R": {"type": "number", "description": "目标碱度 R (CaO/SiO2)，如 0.9"},
                }},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_phase",
            "description": "检索保护渣领域知识：某结晶相的化学式/矿物名/作用机理，或基础概念（碱度、熔点、黏度、Thermo-Calc、代理模型等）。当用户问‘X 是什么 / X 的作用 / 解释某相’时调用。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "自然语言检索词"}},
                "required": ["query"],
            }},
    },
]

DISPATCH = {
    "predict_properties": predict_properties,
    "predict_phase_diagram": predict_phase_diagram,
    "inverse_design": inverse_design,
    "lookup_phase": lookup_phase,
}
