# -*- coding: utf-8 -*-
"""
保护渣相图预测 Web 应用 (Streamlit)
=====================================
输入保护渣化学成分 → 用相图代理模型 (MLP 14→256→256→128→77) 预测各相体积分数
随温度的变化，替代昂贵的 Thermo-Calc 计算。

启动: 双击 run_app.command  (或 .venv/bin/streamlit run app_phase_diagram.py)

性能要点 (遵循 Streamlit 官方建议):
  • @st.fragment 隔离「温度横截面」，滑温度滑块只重算局部，不重画整张相图
  • st.form 包住 12 个成分输入，输入时不触发 rerun，点「更新预测」才算
  • @st.cache_data / @st.cache_resource 缓存预测与模型
"""
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── matplotlib 中文字体 ──────────────────────────────────────────────────────
for _f in ["Heiti SC", "Hiragino Sans GB", "Arial Unicode MS", "PingFang SC"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams["font.sans-serif"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
COMP_COLS = ["SiO2", "Al2O3", "CaO", "MgO", "Na2O", "Li2O", "F", "B2O3",
             "TiO2", "MnO", "Fe2O3", "BaO"]
TEMPS = np.arange(800, 1501, 5)
DEFAULTS = {                                    # 典型保护渣成分 (wt%)
    "SiO2": 30.0, "Al2O3": 5.0, "CaO": 35.0, "MgO": 2.0,
    "Na2O": 8.0, "Li2O": 3.0, "F": 7.0, "B2O3": 2.0,
    "TiO2": 0.5, "MnO": 0.5, "Fe2O3": 2.0, "BaO": 0.0,
}

st.set_page_config(page_title="保护渣相图预测", page_icon="🔬", layout="wide")


# ── 可选密码保护（云端部署时在平台 Secrets 里设置 APP_PASSWORD）───────────────
def check_password():
    """若 Secrets 未配置 APP_PASSWORD 则不锁；配置后必须输入正确密码。"""
    import hmac
    try:
        configured = st.secrets.get("APP_PASSWORD", "")
    except (FileNotFoundError, KeyError):
        configured = ""
    if not configured:                         # 本地开发 / 未设密码
        return True
    if st.session_state.get("_authenticated"):
        return True

    st.title("🔐 保护渣相图预测")
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


# ── 加载模型 / 数据 (缓存) ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(ROOT, "models", "phase_surrogate.pkl"))
    scaler = joblib.load(os.path.join(ROOT, "models", "phase_surrogate_scaler.pkl"))
    phase_names = list(np.load(os.path.join(ROOT, "data", "phase_names_all.npy"),
                               allow_pickle=True))
    return model, scaler, phase_names


@st.cache_data
def load_presets():
    try:
        return pd.read_excel(os.path.join(ROOT, "data", "熔点粘度数据.xlsx"),
                             sheet_name="熔点")
    except Exception:
        return None


@st.cache_data
def predict_curve(comp13_tuple):
    """comp13_tuple: 13 维成分 → (nT, 77) 归一化相分率"""
    comp13 = np.array(comp13_tuple, dtype=float)
    X = np.hstack([TEMPS.reshape(-1, 1), np.tile(comp13, (len(TEMPS), 1))])
    Y = np.clip(model.predict(scaler.transform(X)), 0, 1)
    s = Y.sum(axis=1, keepdims=True)
    return Y / np.where(s > 0, s, 1)


model, scaler, phase_names = load_artifacts()
liq_idx = [i for i, p in enumerate(phase_names) if "IONIC_LIQ" in p]
preset_df = load_presets()


def t_at_liquid_fraction(liq_curve, frac):
    """液相体积分数达到 frac 的温度(相邻两点线性插值，精确定位)；达不到返回 None。"""
    idx = np.where(liq_curve >= frac)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(TEMPS[i])
    t0, t1 = float(TEMPS[i - 1]), float(TEMPS[i])
    f0, f1 = float(liq_curve[i - 1]), float(liq_curve[i])
    return float(t0 + (frac - f0) / (f1 - f0) * (t1 - t0))


def complete_liquid_temp(liq_curve, reach=0.80):
    """完全液化温度：液相达 80% 之后，曲线进入平台的「转折点」(膝点，
    即到首末连线距离最大处)。不依赖 93% 固定阈值——拟合曲线最高液相
    不到 93% 时仍能给出合理值。达不到 80% 返回 None。"""
    i0 = np.where(liq_curve >= reach)[0]
    if len(i0) == 0:
        return None
    s = i0[0]
    x, y = TEMPS[s:], liq_curve[s:]
    if len(y) < 4:
        return float(x[-1])
    x0, y0 = float(x[0]), float(y[0])
    cx, cy = float(x[-1]) - x0, float(y[-1]) - y0        # 弦向量
    dist = np.abs(cx * (y - y0) - cy * (x - x0)) / np.hypot(cx, cy)  # 各点到弦的距离
    return float(x[int(np.argmax(dist))])


import re

# ── 中间相说明 ──────────────────────────────────────────────────────────────
# Thermo-Calc 相名 → (化学式[普通数字], 中文名, 简介)。化学式里的数字会自动渲染成下标。
# 含 "·" 的是水泥式记法(C=CaO,A=Al₂O₃,F=CaF₂)，只把"字母后的数字"下标，乘数 11、7 不动。
PHASE_INFO = {
    "IONIC_LIQ#3":         ("",                    "液相",       "熔融态保护渣（液态离子熔体），完全液化后即此相"),
    "WOLLASTONITE":        ("CaSiO3",              "硅灰石",     "偏硅酸钙 CaO·SiO₂，保护渣中最常见的结晶硅酸盐"),
    "PSEUDO_WOLLASTONITE": ("CaSiO3",              "假硅灰石",   "硅灰石的高温变体（成分同为 CaSiO₃）"),
    "ANORTHITE":           ("CaAl2Si2O8",          "钙长石",     "钙质长石 CaO·Al₂O₃·2SiO₂，高熔点骨架相"),
    "ANORTHITE#2":         ("CaAl2Si2O8",          "钙长石",     "钙长石的另一亚晶格变体"),
    "ALBITE_LOW":          ("NaAlSi3O8",           "钠长石",     "钠质长石 Na₂O·Al₂O₃·6SiO₂，低温变体"),
    "MULLITE":             ("Al6Si2O13",           "莫来石",     "3Al₂O₃·2SiO₂，铝硅酸盐耐火骨架相"),
    "NEPHELINE_G":         ("NaAlSiO4",            "霞石",       "钠铝硅酸盐 Na₂O·Al₂O₃·2SiO₂，G 亚晶格变体"),
    "NEPHELINE_B":         ("NaAlSiO4",            "霞石",       "钠铝硅酸盐，B 亚晶格变体"),
    "NEPHELINE":           ("NaAlSiO4",            "霞石",       "钠铝硅酸盐"),
    "ALPHA_SPINEL":        ("MgAl2O4",             "尖晶石",     "镁铝尖晶石 MgO·Al₂O₃，高熔点"),
    "CAF2_S1":             ("CaF2",                "萤石",       "氟化钙 CaF₂，保护渣中氟的主要赋存相，强助熔"),
    "CA2NA2SI3O9":         ("Ca2Na2Si3O9",         "钠钙硅酸盐", "含 Na、Ca 的硅酸盐（数据库相）"),
    "MG2B2O5":             ("Mg2B2O5",             "硼酸镁",     "镁的硼酸盐"),
    "CA2B2O5_S2":          ("Ca2B2O5",             "硼酸钙",     "钙的硼酸盐，S2 亚晶格变体"),
    "CA2B2O5":             ("Ca2B2O5",             "硼酸钙",     "钙的硼酸盐"),
    "CA11B2SI4O22_LT":     ("Ca11B2Si4O22",        "钙硼硅酸盐", "含 B、Si 的复杂钙酸盐，低温变体"),
    "C11A7F":              ("11CaO·7Al2O3·CaF2",   "含氟铝酸钙", "水泥记法 C₁₁A₇F = 11CaO·7Al₂O₃·CaF₂，铝酸盐相"),
    "C1A1":                ("CaO·Al2O3",           "铝酸一钙",   "水泥记法 CA = CaO·Al₂O₃，铝酸钙"),
}
_SUB_U = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_ELEM_MAP = {"CA": "Ca", "NA": "Na", "SI": "Si", "MG": "Mg", "AL": "Al", "FE": "Fe",
             "TI": "Ti", "MN": "Mn", "BA": "Ba", "LI": "Li", "CL": "Cl", "ZR": "Zr",
             "ZN": "Zn", "CR": "Cr", "SR": "Sr",
             "O": "O", "F": "F", "B": "B", "C": "C", "N": "N", "H": "H", "S": "S", "P": "P", "K": "K"}


def _fmt_formula(s, mathtext=False):
    """化学式普通数字 → 下标。mathtext=True 给 matplotlib($_{2}$)，否则 unicode 下标。"""
    if not s:
        return ""
    cement = "·" in s
    pat = r"([A-Za-z])(\d+)" if cement else r"(\d+)"

    def repl(m):
        num = m.group(2) if cement else m.group(1)
        sub = f"$_{{{num}}}$" if mathtext else num.translate(_SUB_U)
        return (m.group(1) + sub) if cement else sub

    return re.sub(pat, repl, s)


def parse_formula(name):
    """Thermo-Calc 组成型相名 → 普通数字化学式(如 CA2B2O5_S2 → Ca2B2O5)；无法解析返回 None。"""
    base = re.split(r"[_#]", name)[0].upper()
    out, i = "", 0
    while i < len(base):
        if base[i:i + 2] in _ELEM_MAP:
            out += _ELEM_MAP[base[i:i + 2]]; i += 2
        elif base[i] in _ELEM_MAP:
            out += _ELEM_MAP[base[i]]; i += 1
        else:
            return None
        num = ""
        while i < len(base) and base[i].isdigit():
            num += base[i]; i += 1
        out += num
    return out or None


def phase_label(name, mathtext=False):
    """统一显示标签：化学式（中文名）；无中文名仅化学式；都无则原名。"""
    info = PHASE_INFO.get(name)
    if info:
        formula, mineral, _ = info
        f = _fmt_formula(formula, mathtext)
        if f and mineral:
            return f"{f}（{mineral}）"
        return mineral or f or name
    parsed = parse_formula(name)
    return _fmt_formula(parsed, mathtext) if parsed else name


def phase_desc(name):
    info = PHASE_INFO.get(name)
    if info:
        return info[2]
    return ("自动解析化学式（Thermo-Calc 数据库相，无通用矿物名）"
            if parse_formula(name) else "Thermo-Calc 数据库相")


# ════════════════════════════════════════════════════════════════════════════
# 初始化 session_state 默认值 (官方推荐: setdefault，不与 widget 的 value= 混用)
# ════════════════════════════════════════════════════════════════════════════
for c in COMP_COLS:
    st.session_state.setdefault(f"comp_{c}", float(DEFAULTS[c]))
st.session_state.setdefault("_normalize", True)

# ════════════════════════════════════════════════════════════════════════════
# 侧栏: 预设载入 + 成分表单
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🔧 成分输入")

# 预设样本 (表单外，用 button 设 session_state)
if preset_df is not None:
    with st.sidebar.expander("📂 从数据库载入样本"):
        ids = preset_df["整理编号"].astype(str).tolist()
        sel = st.selectbox("选择样本编号", ["（自定义）"] + ids, key="_preset_sel")
        if st.button("⬇ 载入该样本成分", use_container_width=True):
            if sel != "（自定义）":
                row = preset_df[preset_df["整理编号"].astype(str) == sel].iloc[0]
                for c in COMP_COLS:
                    v = pd.to_numeric(row.get(c, 0), errors="coerce")
                    st.session_state[f"comp_{c}"] = float(0.0 if np.isnan(v) else v)
                st.toast(f"已载入样本 {sel}，点击「更新预测」查看结果", icon="✅")

# 成分表单: 输入时不触发 rerun，提交后才算 → 彻底消除「输成分时整页变灰」
with st.sidebar.form("comp_form"):
    st.markdown("#### 氧化物组成 (wt%)")
    for c in COMP_COLS:
        st.number_input(c, min_value=0.0, max_value=100.0, step=0.5,
                        format="%.2f", key=f"comp_{c}")
    st.checkbox("归一化到 100%", key="_normalize",
                help="按比例缩放使成分总和=100%，与训练数据分布一致")
    st.form_submit_button("🔄 更新预测", use_container_width=True, type="primary")

st.sidebar.markdown("---")

# ── 从 session_state 取成分并计算 R ──────────────────────────────────────────
raw = {c: float(st.session_state[f"comp_{c}"]) for c in COMP_COLS}
normalize = bool(st.session_state["_normalize"])
total = sum(raw.values())
comp = {c: (raw[c] / total * 100 if (normalize and total > 0) else raw[c]) for c in COMP_COLS}
R = comp["CaO"] / comp["SiO2"] if comp["SiO2"] > 0 else 0.0

st.sidebar.metric("成分总和", f"{sum(comp.values()):.1f} %")
st.sidebar.metric("碱度 R (CaO/SiO₂)", f"{R:.3f}")
with st.sidebar.expander("📋 成分明细（你输入的 → 归一化后）"):
    detail = pd.DataFrame({
        "氧化物": COMP_COLS,
        "你输入的": [f"{raw[c]:.2f}" for c in COMP_COLS],
        "归一化后": [f"{comp[c]:.2f}" for c in COMP_COLS],
    })
    st.dataframe(detail, width="stretch", hide_index=True)
    if normalize and total > 0:
        st.caption(f"原始总和 {total:.1f}% → 按比例缩放到 100%（与训练数据一致，模型吃的是这组数）")
    else:
        st.caption("未归一化：原值直接喂给模型")
with st.sidebar.expander("ℹ 关于模型"):
    st.write("相图代理模型 MLP [温度+13维成分] → 77 相体积分数。"
             "513 个 Thermo-Calc 相图训练，held-out 主要相 R² 0.80–0.93。"
             "液相线温度 = 液相分数首次 ≥ 93% 的温度。")

# ════════════════════════════════════════════════════════════════════════════
# 预测 (缓存; 结果存 session_state 供 fragment 复用，避免重算)
# ════════════════════════════════════════════════════════════════════════════
comp13 = tuple(comp[c] for c in COMP_COLS) + (R,)
try:
    with st.spinner("正在用代理模型计算相图…"):
        curve = predict_curve(comp13)
except Exception as e:
    st.error(f"预测失败：{e}")
    st.stop()
st.session_state["_curve"] = curve
liq_curve = curve[:, liq_idx].sum(axis=1)
t50 = t_at_liquid_fraction(liq_curve, 0.50)            # 液相 50% 温度
t_complete = complete_liquid_temp(liq_curve)            # 完全液化温度(80%后的膝点)
peak_liq = float(liq_curve.max())

# ════════════════════════════════════════════════════════════════════════════
# 主区域
# ════════════════════════════════════════════════════════════════════════════
st.title("🔬 保护渣相图预测")
st.caption("基于相图代理模型（MLP），输入化学成分即可秒级预测各相体积分数随温度的变化，替代昂贵的 Thermo-Calc 计算。")

c1, c2, c3, c4 = st.columns(4)
c1.metric("液相 50% 温度", f"{t50:.0f} °C" if t50 else "未达 50%",
          help="液相体积分数达到 50% 对应的温度（熔化进程的中点）")
c2.metric("完全液化温度", f"{t_complete:.0f} °C" if t_complete else "—",
          help="液相达 80% 之后，曲线进入平台的转折点（膝点，即到首末连线距离最大处）。"
               "恒有定义，拟合曲线最高液相偏低时也能给出合理值")
c3.metric("峰值液相", f"{peak_liq * 100:.1f} %",
          help="该成分在 800–1500°C 内能达到的最高液相分数（配合「完全液化温度」理解）")
c4.metric("碱度 R", f"{R:.3f}")

# 选主要相 (按峰值) + 确保液相在内
peaks = curve.max(axis=0)
top_idx = [i for i in np.argsort(-peaks) if peaks[i] > 0.02][:8]
if liq_idx and not any(i in top_idx for i in liq_idx):
    top_idx.append(liq_idx[0])

# ── 主相图 (仅成分变化时重画) ──
fig, ax = plt.subplots(figsize=(11, 5))
cmap = plt.cm.tab10
# 画各固相(单独) + 液相(总和) —— 标记 t50/完全液化 用的就是这条「液相总和」曲线，
# 保证竖线与图上曲线严格对应(原来只画 IONIC_LIQ#3 单相，会和基于总和的标记略错位)
j = 0
for pi in top_idx:
    if "IONIC_LIQ" in phase_names[pi]:
        continue
    ax.plot(TEMPS, curve[:, pi], lw=1.8, color=cmap(j % 10),
            label=phase_label(phase_names[pi], mathtext=True))
    j += 1
ax.plot(TEMPS, liq_curve, lw=2.8, color="#d62728", label="液相（总和）")
ax.axhline(0.5, color="#1f77b4", ls=":", lw=0.8, alpha=0.4)   # 50% 水平参考线
# 在图上标两个参考温度: 液相 50% 与 完全液化
for tv, lab, col, y in [(t50, "f=50%", "#1f77b4", 0.96),
                        (t_complete, "完全液化", "#2ca02c", 0.90)]:
    if tv:
        ax.axvline(tv, ls="--", color=col, lw=1.2, alpha=0.8)
        ax.text(tv + 3, y, f"{lab} {tv:.0f}°C", fontsize=8, color=col)
ax.set_xlabel("温度 [°C]")
ax.set_ylabel("体积分数（归一化）")
ax.set_xlim(800, 1500)
ax.set_ylim(-0.03, 1.02)
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
st.pyplot(fig, width="stretch")
plt.close(fig)

# ── 中间相说明（图中出现的相）──
with st.expander("📖 图中中间相的化学式与说明", expanded=True):
    info_rows = []
    for pi in top_idx:
        nm = phase_names[pi]
        info = PHASE_INFO.get(nm)
        parsed = parse_formula(nm)
        formula = _fmt_formula(info[0]) if info else (_fmt_formula(parsed) if parsed else "—")
        mineral = info[1] if info else "—"
        info_rows.append({"Thermo-Calc 名": nm, "化学式": formula,
                          "中文名": mineral, "说明": phase_desc(nm)})
    st.dataframe(pd.DataFrame(info_rows), width="stretch", hide_index=True)
    st.caption("后缀：_S1/_S2=亚晶格变体，_LT=低温变体，#2/#3=编号变体。"
               "液相 = 熔融态保护渣。带中文名的是常见矿物相，其余为 Thermo-Calc 数据库化合物"
               "（化学式由相名自动解析）。")


# ════════════════════════════════════════════════════════════════════════════
# 温度横截面 — 用 @st.fragment 隔离: 滑温度滑块只重算本块，不重画相图
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 指定温度下的相组成")


@st.fragment
def cross_section():
    curve = st.session_state.get("_curve")
    if curve is None:
        st.info("请先在左侧输入成分并点击「更新预测」。")
        return
    t_sel = st.slider("温度", 800, 1500, 1100, step=5, format="%d °C", key="_t_sel")
    i_sel = int(np.argmin(abs(TEMPS - t_sel)))
    fracs = sorted(
        [(phase_names[k], float(curve[i_sel, k]))
         for k in range(len(phase_names)) if curve[i_sel, k] > 0.01],
        key=lambda x: -x[1])
    colA, colB = st.columns([1, 1.1])
    with colA:
        if fracs:
            names = [phase_label(f[0], mathtext=True) for f in fracs][::-1]
            vals = [f[1] for f in fracs][::-1]
            fig2, ax2 = plt.subplots(figsize=(6, max(3, 0.42 * len(fracs) + 1.2)))
            bars = ax2.barh(names, vals, color="#2E6DA4")
            ax2.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
            ax2.set_xlabel("体积分数")
            ax2.set_title(f"{t_sel} °C 相组成")
            ax2.set_xlim(0, max(vals) * 1.25 + 0.01)
            ax2.grid(alpha=0.3, axis="x")
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("该温度下无显著相（>1%）")
    with colB:
        if fracs:
            df_show = pd.DataFrame([(phase_label(f[0]), f[1]) for f in fracs],
                                   columns=["相", "体积分数"])
            df_show["体积分数"] = df_show["体积分数"].map(lambda x: f"{x:.4f}")
            st.dataframe(df_show, width="stretch", hide_index=True)
        else:
            st.write("—")


cross_section()


# ════════════════════════════════════════════════════════════════════════════
# 导出相图数据 — 宽表: 每行一个温度，每列一个相
# ════════════════════════════════════════════════════════════════════════════
# 选显著相(峰值>1%)；IONIC_LIQ 各相合并为「液相（总和）」单独一列
sig_idx = [i for i in range(len(phase_names))
           if curve[:, i].max() > 0.01 and "IONIC_LIQ" not in phase_names[i]]
wide = {"温度(°C)": TEMPS.astype(int), "液相（总和）": np.round(liq_curve, 4)}
_seen = {}
for i in sig_idx:
    lab = phase_label(phase_names[i])
    if lab in _seen:                      # 同名相(如两个钙长石变体)用 Thermo-Calc 原名消歧
        lab = f"{lab}（{phase_names[i]}）"
    _seen[lab] = True
    wide[lab] = np.round(curve[:, i], 4)
wide_df = pd.DataFrame(wide)

st.markdown("### 💾 导出相图数据（宽表：每行一个温度，每列一个相）")
import io as _io
_buf = _io.BytesIO(); wide_df.to_excel(_buf, index=False)   # Excel 原生格式，列一定分得开
st.download_button("⬇ 下载 Excel (.xlsx)", _buf.getvalue(),
                   file_name="phase_diagram_prediction.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                   help="Excel 原生格式，列一定分得开（CSV 在某些中文版 Excel 里会被挤进一列）")
st.download_button("⬇ 下载 CSV", wide_df.to_csv(index=False).encode("utf-8-sig"),
                   file_name="phase_diagram_prediction.csv", mime="text/csv")
with st.expander(f"预览宽表（{len(wide_df)} 行 × {len(wide_df.columns)} 列）"):
    st.dataframe(wide_df, width="stretch", height=260)
cap_label = "归一化后" if (normalize and total > 0) else "原始输入"
st.caption(f"喂给模型的成分（{cap_label}）: " + ", ".join(f"{c}={comp[c]:.1f}" for c in COMP_COLS))
