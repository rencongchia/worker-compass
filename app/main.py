"""
WorkerCompass — Chatbot UI with session management.

Run locally: streamlit run app/main.py
Run via Docker: docker compose up
"""

import logging
import uuid

import streamlit as st

from app.agent import WorkerCompassAgent
from app.config import load_config
from app.prompts import LANGUAGES, PLACEHOLDERS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

st.set_page_config(
    page_title="WorkerCompass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Larger, properly centred icon buttons (pen, trash) in the chat session list */
    section[data-testid="stSidebar"] [data-testid="stButton"] button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-height: 2.4rem;
        min-width: 2.4rem;
        padding: 0.25rem !important;
        font-size: 1.1rem;
        line-height: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _new_session(name: str | None = None) -> str:
    sid = str(uuid.uuid4())[:8]
    idx = len(st.session_state.sessions) + 1
    st.session_state.sessions[sid] = {
        "name": name or f"Chat {idx}",
        "messages": [],   # list of {role, content, response?}
        "feedback": {},   # msg_idx → True/False
    }
    st.session_state.active_session = sid
    return sid


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

if "agent" not in st.session_state:
    cfg = load_config()
    st.session_state.agent = WorkerCompassAgent(cfg)

if "sessions" not in st.session_state:
    st.session_state.sessions = {}
    _new_session("Chat 1")

if "active_session" not in st.session_state:
    st.session_state.active_session = next(iter(st.session_state.sessions))

if "language" not in st.session_state:
    st.session_state.language = "en"

if "renaming" not in st.session_state:
    st.session_state.renaming = None  # session id currently being renamed

# ---------------------------------------------------------------------------
# Sidebar — language + session management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧭 WorkerCompass")
    st.caption("Employment rights · Singapore migrant workers")

    st.divider()

    # Language selector
    lang_options = list(LANGUAGES.keys())
    lang_labels = [f"{LANGUAGES[c]['display']} ({LANGUAGES[c]['name']})" for c in lang_options]
    sel_idx = st.selectbox(
        "Language / மொழி / ভাষা / ဘာသာစကား",
        options=range(len(lang_options)),
        format_func=lambda i: lang_labels[i],
        index=lang_options.index(st.session_state.language),
        key="lang_select",
    )
    st.session_state.language = lang_options[sel_idx]

    st.divider()

    if st.button("＋ New chat", use_container_width=True, type="primary"):
        _new_session()
        st.rerun()

    st.markdown("**Chats**")

    for sid, sess in list(st.session_state.sessions.items()):
        is_active = sid == st.session_state.active_session

        if st.session_state.renaming == sid:
            new_name = st.text_input(
                "Rename chat",
                value=sess["name"],
                key=f"ri_{sid}",
                label_visibility="collapsed",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save", key=f"rs_{sid}", use_container_width=True):
                    st.session_state.sessions[sid]["name"] = new_name.strip() or sess["name"]
                    st.session_state.renaming = None
                    st.rerun()
            with c2:
                if st.button("Cancel", key=f"rc_{sid}", use_container_width=True):
                    st.session_state.renaming = None
                    st.rerun()
        else:
            c_name, c_ren, c_del = st.columns([6, 1, 1], vertical_alignment="center")
            with c_name:
                label = f"**{sess['name']}**" if is_active else sess["name"]
                if st.button(label, key=f"sb_{sid}", use_container_width=True):
                    st.session_state.active_session = sid
                    st.rerun()
            with c_ren:
                if st.button("✏️", key=f"rn_{sid}", help="Rename"):
                    st.session_state.renaming = sid
                    st.rerun()
            with c_del:
                if st.button("🗑", key=f"dl_{sid}", help="Delete"):
                    del st.session_state.sessions[sid]
                    remaining = list(st.session_state.sessions.keys())
                    if remaining:
                        st.session_state.active_session = remaining[-1]
                    else:
                        _new_session("Chat 1")
                    st.rerun()

    st.divider()
    st.caption("MOM: **6438 5122** · TWC2: twc2.org.sg · MWC: **6536 2692**")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

active = st.session_state.sessions[st.session_state.active_session]
messages = active["messages"]
feedback = active["feedback"]
language = st.session_state.language

st.title("🧭 WorkerCompass")
st.caption(
    "Employment rights information for migrant workers in Singapore · "
    "Grounded in Singapore legislation and MOM guidance"
)

# Empty-state prompt
if not messages:
    st.info(
        "Ask a question about your employment rights in Singapore. "
        "You can ask in English, Bengali (বাংলা), Tamil (தமிழ்), or Burmese (မြန်မာ).",
        icon="💬",
    )

# Render conversation history
for idx, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
            continue

        resp = msg.get("response")

        if resp is None:
            st.write(msg["content"])
        elif resp.refused:
            st.warning(resp.answer, icon="ℹ️")
        else:
            st.markdown(resp.answer)

            if resp.freshness_warning:
                st.warning(resp.freshness_warning, icon="⚠️")

            if resp.citations:
                _sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
                with st.expander(f"📄 Sources ({len(resp.citations)} documents)"):
                    for i, cit in enumerate(resp.citations, 1):
                        sup = str(i).translate(_sup)
                        st.markdown(
                            f"**{sup} {cit['act_name']}**"
                            + (f" — {cit['section_title']}" if cit["section_title"] else "")
                        )
                        if cit["url"]:
                            st.markdown(f"🔗 [{cit['url']}]({cit['url']})")
                        with st.expander("Show excerpt", expanded=False):
                            st.caption(cit["snippet"] + " …")
                        if cit.get("corpus_snapshot_date"):
                            st.caption(f"Last retrieved: {cit['corpus_snapshot_date']}")
                        st.divider()

        # Feedback buttons (assistant messages only)
        fb_key = f"fb_{idx}"
        sid = st.session_state.active_session
        if fb_key not in feedback:
            fc1, fc2, _ = st.columns([1, 1, 8])
            with fc1:
                if st.button("👍", key=f"up_{sid}_{idx}", help="Helpful"):
                    active["feedback"][fb_key] = True
                    st.rerun()
            with fc2:
                if st.button("👎", key=f"dn_{sid}_{idx}", help="Not helpful"):
                    active["feedback"][fb_key] = False
                    st.rerun()
        else:
            if feedback[fb_key]:
                st.caption("✅ Thanks for your feedback!")
            else:
                st.caption("❌ Thanks. MOM 6438 5122 | twc2.org.sg | MWC 6536 2692")

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

query = st.chat_input(PLACEHOLDERS[language])

if query and query.strip():
    active["messages"].append({"role": "user", "content": query.strip()})

    # Auto-name session from first message
    if len(active["messages"]) == 1:
        active["name"] = query.strip()[:45] + ("…" if len(query.strip()) > 45 else "")

    # Build conversation history for multi-turn context (last 2 non-refused turns)
    history = []
    msgs = active["messages"][:-1]
    i = 0
    while i < len(msgs) - 1:
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            prev_resp = msgs[i + 1].get("response")
            if prev_resp and not prev_resp.refused:
                history.append({"query": msgs[i]["content"], "answer": prev_resp.answer})
            i += 2
        else:
            i += 1
    history = history[-2:]

    with st.spinner("Searching legal documents …"):
        try:
            response = st.session_state.agent.run(
                query.strip(), language, history=history or None
            )
            active["messages"].append({
                "role": "assistant",
                "content": response.answer,
                "response": response,
            })
        except Exception as exc:
            active["messages"].append({
                "role": "assistant",
                "content": f"Error: {exc}",
                "response": None,
            })

    st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "WorkerCompass is a research prototype. It is not legal advice. "
    "Information is grounded in Singapore legislation and MOM guidance as of the corpus snapshot date. "
    "For your specific situation, contact MOM (6438 5122), TWC2 (twc2.org.sg), or MWC (6536 2692)."
)
