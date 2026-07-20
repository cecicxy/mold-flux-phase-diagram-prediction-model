# -*- coding: utf-8 -*-
"""
领域知识 RAG 检索 (Retrieval-Augmented Generation 里的 R)
=========================================================
语料: 保护渣基础概念 + 主要结晶相的化学式 / 矿物名 / 作用机理。
向量化: 优先 OpenAI text-embedding-3-small (带磁盘缓存 data/rag_index.npz)；
        未配置 OPENAI_API_KEY 时自动退化为 sklearn TF-IDF + 余弦相似度。
检索: 余弦相似度 top-k。

作用: Agent 在解释预测 / 设计结果时检索引用，降低 LLM 在专业术语上的幻觉。

用法:
    from src.agent_rag import retrieve
    docs = retrieve("霞石是什么", k=3)
"""
import os
import sys
import hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

# ── 知识语料 (id, 分类, 正文) ────────────────────────────────────────────────
CORPUS = [
    ("intro", "基础", "保护渣（连铸保护渣）覆盖在连铸结晶器钢液表面，起绝热保温、防止二次氧化、吸收夹杂物、润滑坯壳、控制传热等作用。化学成分以 SiO2-CaO 为主体，配入 Al2O3、Na2O、Li2O、F、B2O3、MgO 等。"),
    ("R", "基础", "碱度 R = CaO/SiO2（质量比），是保护渣最关键的结构参数。R 偏高→碱性、熔化温度偏低、黏度下降；R 偏低→酸性、熔化温度升高。典型保护渣 R 在 0.8–1.2。"),
    ("them", "基础", "半球熔化温度 T_hem：试样熔化至半球状（高度降为一半）时的温度，工程上代表保护渣的熔点，典型 950–1150°C。Na2O、Li2O、F 显著降低 T_hem；Al2O3 升高 T_hem。"),
    ("visc", "基础", "黏度 η：保护渣在 1300°C 的动力黏度（Pa·s），影响润滑与传热，典型 0.1–0.5 Pa·s。Na2O、Li2O、F 降低黏度；SiO2、Al2O3 升高黏度。"),
    ("tc", "方法", "Thermo-Calc：商业热力学平衡计算软件，给定成分与温度算出各相体积分数。单次计算昂贵（分钟级），本项目训练 MLP 代理模型近似它（毫秒级）。"),
    ("sur", "方法", "相图代理模型：MLP 14→256→256→128→77，输入 [温度, 13 维成分(12 氧化物+R)]，输出 77 个相体积分数。在 513 张 Thermo-Calc 相图上训练，held-out 主要相 R² 0.80–0.93。"),
    ("inv", "方法", "反向设计：给定目标性质（如 T_hem≤1050°C、黏度≈0.2 Pa·s、R≈0.9），用差分进化在成分空间搜索满足目标的保护渣化学成分配方。"),
    ("shap", "方法", "正向模型关键成分（SHAP）：Na2O(~25%) > Li2O(~17%) > Al2O3(~13%) > MgO、CaO。Na2O、Li2O 是最强的助熔、降黏组分。"),
    ("liq", "相", "液相 IONIC_LIQ：熔融态保护渣（液态离子熔体）。完全液化后即此相；液相分数随温度升高而增加。"),
    ("wol", "相", "硅灰石 WOLLASTONITE，化学式 CaSiO3 (CaO·SiO2)：保护渣中最常见的结晶硅酸盐，熔体冷却析出的主要初晶相之一。"),
    ("ano", "相", "钙长石 ANORTHITE，化学式 CaAl2Si2O8 (CaO·Al2O3·2SiO2)：高熔点（约 1550°C）骨架相，Al2O3 高时易出现，提高保护渣凝固温度。"),
    ("mul", "相", "莫来石 MULLITE，化学式 Al6Si2O13 (3Al2O3·2SiO2)：铝硅酸盐耐火骨架相，高 Al2O3 体系出现，熔点高（约 1850°C）。"),
    ("neph", "相", "霞石 NEPHELINE，化学式 NaAlSiO4 (Na2O·Al2O3·2SiO2)：钠铝硅酸盐，Na2O 与 Al2O3 同时较高时出现，降低熔化温度。"),
    ("caf2", "相", "萤石 CaF2 (CAF2)：保护渣中氟的主要赋存相，强助熔，显著降低熔化温度与黏度；过多会侵蚀耐火材料。"),
    ("alb", "相", "钠长石 ALBITE，化学式 NaAlSi3O8 (Na2O·Al2O3·6SiO2)：钠质长石，Na2O 高且 SiO2 充足时出现。"),
    ("spi", "相", "尖晶石 MgAl2O4 (SPINEL)：镁铝尖晶石，MgO 与 Al2O3 同时存在时出现，高熔点（约 2135°C）固相。"),
    ("casib", "相", "硼酸盐（如 Ca2B2O5、Mg2B2O5）：B2O3 引入的含硼相，助熔、细化析晶；过多降低化学稳定性。"),
    ("cusa", "相", "钙铝酸盐（如 CaO·Al2O3、11CaO·7Al2O3·CaF2）：高 CaO、高 Al2O3 体系的水泥式相，影响凝固行为与熔化温度。"),
]

_INDEX = None


def _corpus_key():
    return hashlib.md5("|".join(c[2] for c in CORPUS).encode()).hexdigest()[:10]


def _has_openai():
    return bool(os.environ.get("OPENAI_API_KEY"))


def _build_openai():
    from openai import OpenAI
    client = OpenAI()
    texts = [c[2] for c in CORPUS]
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def _build_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    # 字符级 n-gram：中文无空格分词，char_wb 能匹配 "霞石"、"CaF2" 等子串
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    M = vec.fit_transform([c[2] for c in CORPUS]).toarray().astype(np.float32)
    return M, vec


def _ensure():
    global _INDEX
    if _INDEX is not None:
        return
    cache = os.path.join(ROOT, "data", "rag_index.npz")
    if _has_openai():
        try:
            vecs = None
            if os.path.exists(cache):
                d = np.load(cache, allow_pickle=True)
                if str(d["key"]) == _corpus_key():
                    vecs = d["vecs"]
            if vecs is None:
                vecs = _build_openai()
                np.savez(cache, key=_corpus_key(), vecs=vecs)
            _INDEX = ("oai", vecs)
            return
        except Exception as e:  # 网络/配额失败 → 退回 TF-IDF
            print(f"[RAG] OpenAI embedding 失败，退回 TF-IDF：{e}")
    M, vec = _build_tfidf()
    _INDEX = ("tfidf", M, vec)


def _embed_query(text):
    if _INDEX[0] == "oai":
        from openai import OpenAI
        v = OpenAI().embeddings.create(model="text-embedding-3-small", input=[text]).data[0].embedding
        return np.array(v, dtype=np.float32)
    return _INDEX[2].transform([text]).toarray().astype(np.float32)[0]


def retrieve(query: str, k: int = 3):
    """余弦相似度 top-k 检索。返回 list[dict(id, cat, text, score)]。"""
    _ensure()
    M = _INDEX[1]
    q = _embed_query(query)
    sims = M @ q / (np.linalg.norm(M, axis=1) * np.linalg.norm(q) + 1e-9)
    idx = np.argsort(-sims)[:k]
    return [{"id": CORPUS[i][0], "cat": CORPUS[i][1],
             "text": CORPUS[i][2], "score": float(sims[i])} for i in idx]


def mode() -> str:
    """当前检索模式（'openai-embedding' / 'tfidf-fallback'），供 UI 展示。"""
    _ensure()
    return "openai-embedding" if _INDEX[0] == "oai" else "tfidf-fallback"


if __name__ == "__main__":
    for q in ["霞石是什么", "怎么降低熔点", "碱度什么意思", "CaF2 的作用"]:
        print("\nQ:", q)
        for d in retrieve(q, k=3):
            print(f"  [{d['score']:.3f}] {d['text'][:60]}")
