"""Chatbot page — natural-language query interface for inspection logs."""
import streamlit as st
from datetime import datetime

from chatbot.chatbot_engine import InspectionChatbot
from utils.config import THEME


def render_chatbot():
    st.markdown('<div class="page-title">🤖 AI Inspection Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Query inspection history using natural language — 100% offline, database-backed'
        '</div>',
        unsafe_allow_html=True,
    )

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = InspectionChatbot()
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hello! I'm the **Inspection Log Chatbot**. "
                    "I can answer questions about your steel billet inspection records.\n\n"
                    + InspectionChatbot.HELP_TEXT
                ),
            }
        ]

    bot: InspectionChatbot = st.session_state.chatbot

    # ── Chat history ──────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────
    user_input = st.chat_input("Ask about inspection records…")

    if user_input:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Querying inspection database..."):
                result = bot.query(user_input)
                response = result.get("message", "I couldn't process that query.")

            st.markdown(response)

        st.session_state.chat_messages.append({"role": "assistant", "content": response})

    # ── Quick actions sidebar ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")

    qa1, qa2, qa3 = st.columns(3)

    with qa1:
        if st.button("📊 Today's Summary", use_container_width=True):
            result = bot.query("summarise today's inspections")
            st.session_state.chat_messages.append({"role": "user", "content": "Summarise today's inspections"})
            st.session_state.chat_messages.append({"role": "assistant", "content": result["message"]})
            st.rerun()

    with qa2:
        if st.button("⚠️ Show Defects", use_container_width=True):
            result = bot.query("show billets with defects")
            st.session_state.chat_messages.append({"role": "user", "content": "Show billets with defects"})
            st.session_state.chat_messages.append({"role": "assistant", "content": result["message"]})
            st.rerun()

    with qa3:
        if st.button("❓ Unreadable IDs", use_container_width=True):
            result = bot.query("show billets with unreadable ids")
            st.session_state.chat_messages.append({"role": "user", "content": "Show billets with unreadable IDs"})
            st.session_state.chat_messages.append({"role": "assistant", "content": result["message"]})
            st.rerun()

    qa4, qa5, qa6 = st.columns(3)

    with qa4:
        if st.button("🔔 Recent Alerts", use_container_width=True):
            result = bot.query("show recent alerts")
            st.session_state.chat_messages.append({"role": "user", "content": "Show recent alerts"})
            st.session_state.chat_messages.append({"role": "assistant", "content": result["message"]})
            st.rerun()

    with qa5:
        if st.button("📈 Defect Rate", use_container_width=True):
            result = bot.query("what is the defect rate")
            st.session_state.chat_messages.append({"role": "user", "content": "What is the defect rate?"})
            st.session_state.chat_messages.append({"role": "assistant", "content": result["message"]})
            st.rerun()

    with qa6:
        if st.button("🚩 Review Flags", use_container_width=True):
            result = bot.query("which batches need quality review")
            st.session_state.chat_messages.append({"role": "user", "content": "Which batches need quality review?"})
            st.session_state.chat_messages.append({"role": "assistant", "content": result["message"]})
            st.rerun()

    # ── Batch flagging tool ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚩 Flag Batch for Quality Review")
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        flag_batch = st.text_input("Batch ID", placeholder="e.g. B-1234", key="flag_batch_id")
    with fc2:
        flag_reason = st.text_input("Reason", placeholder="e.g. Repeated surface defects", key="flag_reason")

    if st.button("🚩 Flag Batch", use_container_width=True):
        if flag_batch and flag_reason:
            result = bot.flag_batch_for_quality_review(flag_batch, flag_reason)
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": result["message"],
            })
            st.rerun()
        else:
            st.warning("Please provide both batch ID and reason.")

    # ── Clear chat ────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_messages = [st.session_state.chat_messages[0]]
        st.rerun()
