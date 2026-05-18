"""
app.py
──────
Streamlit web interface for the Local AI Smart Router.

Run with:
    streamlit run app.py
"""

import time
import streamlit as st
from router import Router, RoutedResponse
from classifier import classify
from router import pick_model
from config import MODELS
import ollama

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Local AI Router",
    page_icon="🤖",
    layout="wide",
)

# ── Category colours ──────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "factual":   "#00bcd4",
    "reasoning": "#ffc107",
    "code":      "#4caf50",
    "creative":  "#e91e63",
    "edge":      "#f44336",
}

MODEL_COLORS = {
    "llama3.2:3b": "#2196f3",
    "phi4-mini":   "#4caf50",
    "mistral:7b":  "#ff9800",
}

MEMORY_WINDOW = 5

# ── Session state ─────────────────────────────────────────────────────────────

if "messages"      not in st.session_state:
    st.session_state.messages      = []   # display messages
if "memory"        not in st.session_state:
    st.session_state.memory        = []   # raw {role, content} for model context
if "router"        not in st.session_state:
    st.session_state.router        = None
if "stats"         not in st.session_state:
    st.session_state.stats         = []   # list of RoutedResponse metadata
if "prefer_speed"  not in st.session_state:
    st.session_state.prefer_speed  = False


# ── Load router once ──────────────────────────────────────────────────────────

@st.cache_resource
def load_router(prefer_speed: bool) -> Router:
    return Router(prefer_speed=prefer_speed)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    mode = st.radio(
        "Routing mode",
        ["Quality (default)", "Speed"],
        index=0,
    )
    prefer_speed = mode == "Speed"

    if prefer_speed != st.session_state.prefer_speed:
        st.session_state.prefer_speed = prefer_speed
        st.session_state.router       = None   # force reload

    show_routing = st.toggle("Show routing details", value=True)
    show_memory  = st.toggle("Show memory indicator", value=True)

    st.divider()

    st.subheader("Models")
    for tag, info in MODELS.items():
        color = MODEL_COLORS.get(tag, "#888")
        st.markdown(
            f'<span style="color:{color}">●</span> **{info["label"]}** ({info["params_b"]}B)',
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.memory   = []
        st.session_state.stats    = []
        st.rerun()

    # ── Session stats ─────────────────────────────────────────────────────────
    if st.session_state.stats:
        st.subheader("Session Stats")
        total = len(st.session_state.stats)
        st.metric("Total queries", total)

        by_model = {}
        for s in st.session_state.stats:
            by_model.setdefault(s["model"], []).append(s["tps"])

        for model, tps_list in by_model.items():
            label = MODELS.get(model, {}).get("label", model)
            avg   = sum(tps_list) / len(tps_list)
            st.markdown(f"**{label}**: {len(tps_list)} queries · {avg:.1f} tok/s avg")

        by_cat = {}
        for s in st.session_state.stats:
            by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
        st.markdown("**By category:**")
        for cat, count in sorted(by_cat.items()):
            color = CATEGORY_COLORS.get(cat, "#888")
            st.markdown(
                f'<span style="color:{color}">■</span> {cat}: {count}',
                unsafe_allow_html=True,
            )


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🤖 Local AI Router")
st.caption("Routes each question to the best local model based on benchmark data. No internet required.")

# Load router
if st.session_state.router is None:
    with st.spinner("Loading router and benchmark data..."):
        st.session_state.router = load_router(st.session_state.prefer_speed)

router = st.session_state.router

# ── Chat history display ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]

            if show_routing:
                cat   = meta["category"]
                model = meta["model"]
                label = MODELS.get(model, {}).get("label", model)
                cat_color   = CATEGORY_COLORS.get(cat,   "#888")
                model_color = MODEL_COLORS.get(model, "#888")

                cols = st.columns([1, 1, 1, 1])
                cols[0].markdown(
                    f'<span style="color:{cat_color}">◆ {cat}</span>',
                    unsafe_allow_html=True,
                )
                cols[1].markdown(
                    f'<span style="color:{model_color}">⚡ {label}</span>',
                    unsafe_allow_html=True,
                )
                cols[2].markdown(f"⏱ {meta['latency']:.2f}s")
                cols[3].markdown(f"🔤 {meta['tps']:.0f} tok/s")

            if show_memory and "memory_exchanges" in meta:
                st.caption(f"Memory: {meta['memory_exchanges']}/{MEMORY_WINDOW} exchanges")

        st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────────────────────

if question := st.chat_input("Ask anything..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Route and respond
    with st.chat_message("assistant"):
        # Classify
        classification = classify(question)
        decision       = pick_model(
            classification.category,
            router.routing_table,
            prefer_speed=router.prefer_speed,
        )

        label       = MODELS.get(decision.chosen_model, {}).get("label", decision.chosen_model)
        cat         = classification.category
        cat_color   = CATEGORY_COLORS.get(cat, "#888")
        model_color = MODEL_COLORS.get(decision.chosen_model, "#888")

        if show_routing:
            cols = st.columns([1, 1, 1, 1])
            cols[0].markdown(
                f'<span style="color:{cat_color}">◆ {cat}</span>',
                unsafe_allow_html=True,
            )
            cols[1].markdown(
                f'<span style="color:{model_color}">⚡ {label}</span>',
                unsafe_allow_html=True,
            )
            cols[2].markdown("⏱ thinking...")
            cols[3].markdown("🔤 ...")

        # Build messages with memory
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful local AI assistant. "
                    "You have memory of recent conversation. "
                    "Be concise and accurate."
                ),
            }
        ]
        messages.extend(st.session_state.memory[-(MEMORY_WINDOW * 2):])
        messages.append({"role": "user", "content": question})

        # Stream the response
        client    = ollama.Client()
        t0        = time.perf_counter()
        full_text = []
        placeholder = st.empty()

        stream = client.chat(
            model=decision.chosen_model,
            messages=messages,
            options={"temperature": 0.7},
            stream=True,
        )

        token_count = 0
        for chunk in stream:
            content = chunk["message"]["content"]
            full_text.append(content)
            placeholder.markdown("".join(full_text) + "▌")
            if chunk.get("done"):
                token_count = chunk.get("eval_count", 0)

        elapsed     = time.perf_counter() - t0
        answer      = "".join(full_text)
        tps         = token_count / elapsed if elapsed > 0 else 0
        placeholder.markdown(answer)

        # Update routing display with real numbers
        if show_routing:
            cols[2].markdown(f"⏱ {elapsed:.2f}s")
            cols[3].markdown(f"🔤 {tps:.0f} tok/s")

        exchanges = len(st.session_state.memory) // 2 + 1
        if show_memory:
            st.caption(f"Memory: {min(exchanges, MEMORY_WINDOW)}/{MEMORY_WINDOW} exchanges")

        # Save to memory and history
        st.session_state.memory.append({"role": "user",      "content": question})
        st.session_state.memory.append({"role": "assistant", "content": answer})

        meta = {
            "category": cat,
            "model":    decision.chosen_model,
            "latency":  round(elapsed, 3),
            "tps":      round(tps, 1),
            "memory_exchanges": min(exchanges, MEMORY_WINDOW),
        }

        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "meta":    meta,
        })
        st.session_state.stats.append(meta)
