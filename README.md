---
title: 保护渣相图预测 & 智能设计 Agent
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.59"
app_file: app_phase_diagram.py
pinned: false
---

# 保护渣相图预测 & 智能设计 Agent 🔬🤖

一个 Streamlit 多页应用（同一 URL，侧栏切换），共用一套模型后端：

| 页面 | 入口 | 作用 |
|------|------|------|
| **相图预测**（表单式，首页） | `app_phase_diagram.py` | 输入 12 种氧化物 → MLP 相图代理秒级预测各相体积分数随温度的变化、液相 50% 温度、完全液化温度等，替代昂贵的 Thermo-Calc。 |
| **🤖 设计 Agent**（对话式） | `pages/2_🤖_设计_Agent.py` | 自然语言驱动：成分性质预测 / 相图推理 / **反向配方设计** / 领域知识问答。LangGraph + LLM function-calling 多步工具调用 + RAG。 |

## 运行（本地）
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
streamlit run app_phase_diagram.py     # 侧栏自动出现「🤖 设计 Agent」入口
```
> 依赖版本上限已固定（`pandas<3`、`pyarrow<20`），避免 Apple Silicon 上的段错误。
> Agent 的 LLM 默认走「离线规则路由」（不调 LLM 也能演示工具链）；配置 `OPENAI_API_KEY` 后切换为真 function-calling。

## 部署到 Streamlit Community Cloud
1. **主文件**：Cloud Dashboard → New app → 主文件选 `app_phase_diagram.py`。`pages/` 会被自动识别为多页，无需另建 app。
2. **Python 版本**：Advanced Settings → 选 **3.12**（`pyarrow<20` 无 3.14 wheel）。
3. **Secrets**（app 的 Settings → Secrets，会同时注入为环境变量）：
   ```toml
   APP_PASSWORD = "你的强密码"                 # 公网必设，两个页面共用
   # Agent 页需要（不设则离线模式）：
   OPENAI_API_KEY  = "sk-..."                 # DeepSeek / OpenAI 等
   OPENAI_BASE_URL = "https://api.deepseek.com/v1"
   AGENT_LLM_MODEL = "deepseek-chat"
   ```
   密钥只进 Secrets，**不进代码仓库**（`secrets.toml.example` 仅作示例）。

## 文件
- `app_phase_diagram.py` — 相图预测首页（多页应用的 main file）
- `pages/2_🤖_设计_Agent.py` — Agent 对话页
- `_auth.py` — 共享密码门
- `src/` — Agent 核心（`agent.py` 编排 / `agent_tools.py` 工具 / `agent_rag.py` 知识检索）
- `models/` — 相图代理（phase_surrogate.pkl / scaler）+ 正向模型（forward_them / forward_visc）
- `data/phase_names_all.npy` — 77 个相名
- `requirements.txt` — 依赖（精简 + 版本锁定）
- `.streamlit/` — headless / 主题 / secrets 示例
