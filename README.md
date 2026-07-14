---
title: 保护渣相图预测
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.59"
app_file: app_phase_diagram.py
pinned: false
---

# 保护渣相图预测 🔬

输入保护渣化学成分（12 种氧化物）→ 用相图代理模型（MLP）秒级预测各相体积分数
随温度的变化曲线、液相 50% 温度、完全液化温度等，替代昂贵的 Thermo-Calc 计算。

## 运行（本地）
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
streamlit run app_phase_diagram.py
```
依赖版本上限已固定（`pandas<3`、`pyarrow<20`），避免 Apple Silicon 上的段错误。

## 访问密码
应用支持密码保护：在部署平台的 **Secrets / 环境变量** 里设置
```
APP_PASSWORD = "你的强密码"
```
设置后，访问者必须输入该密码才能使用；不设置则不锁（本地开发）。
密码存放在平台 Secrets，**不进代码仓库**。

## 文件
- `app_phase_diagram.py` — 程序本体
- `models/` — 相图代理模型（phase_surrogate.pkl / scaler）
- `data/phase_names_all.npy` — 77 个相名
- `requirements.txt` — 依赖（精简 + 版本锁定）
- `.streamlit/config.toml` — headless / 主题
