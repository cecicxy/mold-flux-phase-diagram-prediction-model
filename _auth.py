# -*- coding: utf-8 -*-
"""共享密码门：多页应用里每个页面 import check_password() 即可。

Secrets 未配置 APP_PASSWORD 时不锁（本地开发）；配置后必须输入正确密码。
session_state 跨页面共享，所以一次登录覆盖所有页面。
"""
import hmac
import streamlit as st


def check_password() -> bool:
    try:
        configured = st.secrets.get("APP_PASSWORD", "")
    except (FileNotFoundError, KeyError):
        configured = ""
    if not configured:
        return True
    if st.session_state.get("_authenticated"):
        return True

    st.title("🔐 保护渣智能设计")
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
