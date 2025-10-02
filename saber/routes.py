import streamlit as st


routes = (
    st.Page(
        "views/chat.py",
        title="Chat",
        icon="💬",
    ),
    st.Page(
        "views/settings.py",
        title="Settings",
        icon="⚙️",
    ),
)
