"""
Streamlit UI for the Ramayana Knowledge Assistant.

A persona-based chatbot where users pick Lord Rama, Lakshmana, or
Hanuman and ask questions grounded in the Ramayana knowledge base
built by rag_utils/. Run with:

    streamlit run app.py
"""

import logging
import time

import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai

from config import (
    AI_PROVIDERS,
    CACHE_TTL_SECONDS,
    FALLBACK_QUESTIONS,
    GOOGLE_API_KEY,
    MAX_AUDIO_SIZE_MB,
    MAX_QUESTION_LENGTH,
    OPENAI_API_KEY,
    PERSONAS,
    RATE_LIMIT_MAX_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)
from rag_utils.rag_pipeline import RamayanaRAGPipeline

# Real error details go here (server-side only) instead of on screen —
# raw exceptions can leak file paths, library internals, or provider
# endpoint details that users shouldn't see.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ramayana_assistant")

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ramayana Knowledge Assistant",
    page_icon="🏹",
    layout="centered",
)

PERSONA_EMOJI = {
    "Lord Rama": "🏹",
    "Lakshmana": "🛡️",
    "Hanuman": "🐒",
}

PERSONA_TAGLINE = {
    "Lord Rama": "Calm, dharma-guided wisdom",
    "Lakshmana": "Loyal, disciplined, direct",
    "Hanuman": "Devoted, energetic, encouraging",
}


# ---------------------------------------------------------------------------
# Backend loading (cached so it only initializes once per server session).
# A TTL is set so cached pipelines — which hold a reference to whichever
# API key built them — don't stay resident in server memory forever.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load_pipeline(api_key: str = None, provider: str = "Gemini", model: str = None):
    # Cached per distinct (api_key, provider, model) combination —
    # switching any of these gets its own pipeline instance automatically.
    return RamayanaRAGPipeline(api_key=api_key, provider=provider, model=model)


@st.cache_resource(show_spinner=False)
def get_transcription_model(api_key: str):
    """A lightweight Gemini client used only for speech-to-text.
    Voice input always uses Gemini regardless of the chosen chat
    provider, since that's the only transcription backend built."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(AI_PROVIDERS["Gemini"]["chat_models"][0])


def transcribe_audio(audio_bytes: bytes, api_key: str) -> str:
    """
    Send a recorded audio clip to Gemini and return the spoken words
    as plain text, so voice questions can be handled exactly like
    typed ones by the rest of the app.
    """
    model = get_transcription_model(api_key)
    response = model.generate_content(
        [
            {"mime_type": "audio/wav", "data": audio_bytes},
            "Transcribe exactly what is spoken in this audio. "
            "Output only the transcribed words, with no extra commentary.",
        ]
    )
    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages_by_persona" not in st.session_state:
    st.session_state.messages_by_persona = {}  # {persona_name: [list of {role, content, ...}]}

if "persona" not in st.session_state:
    st.session_state.persona = None  # no persona chosen yet — show selection screen

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

if "provider" not in st.session_state:
    st.session_state.provider = "Gemini"

if "model" not in st.session_state:
    st.session_state.model = AI_PROVIDERS["Gemini"]["chat_models"][0]

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "request_timestamps" not in st.session_state:
    st.session_state.request_timestamps = []  # for simple per-session rate limiting


def check_rate_limit() -> bool:
    """
    Returns True if this session is within its request quota. Keeps
    only timestamps from within the current rolling window, so this
    naturally resets over time without needing a background job.
    """
    now = time.time()
    st.session_state.request_timestamps = [
        t for t in st.session_state.request_timestamps
        if now - t < RATE_LIMIT_WINDOW_SECONDS
    ]
    if len(st.session_state.request_timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        return False
    st.session_state.request_timestamps.append(now)
    return True


# ---------------------------------------------------------------------------
# Sidebar — persona selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏹 Ramayana Assistant")
    st.caption("A persona-based knowledge assistant grounded in the Ramayana.")

    st.subheader("⚙️ Model Configuration")

    provider_choice = st.selectbox(
        "AI Provider",
        options=list(AI_PROVIDERS.keys()),
        index=list(AI_PROVIDERS.keys()).index(st.session_state.provider),
    )
    if provider_choice != st.session_state.provider:
        st.session_state.provider = provider_choice
        # Reset model to the new provider's default when switching
        st.session_state.model = AI_PROVIDERS[provider_choice]["chat_models"][0]

    provider_info = AI_PROVIDERS[st.session_state.provider]

    model_choice = st.selectbox(
        f"{st.session_state.provider} Model",
        options=provider_info["chat_models"],
        index=provider_info["chat_models"].index(st.session_state.model)
        if st.session_state.model in provider_info["chat_models"]
        else 0,
    )
    st.session_state.model = model_choice

    show_key = st.checkbox("👁 Show key", value=False)
    st.session_state.user_api_key = st.text_input(
        f"{st.session_state.provider} API Key",
        value=st.session_state.user_api_key,
        type="default" if show_key else "password",
        placeholder=provider_info["key_placeholder"],
        label_visibility="collapsed",
        help=f"Get a free key at {provider_info['get_key_url']}. "
             "Your key is used only in this session and never stored.",
    )
    st.caption(
        f"Optional — leave blank to use the app's default key, if the "
        f"owner has configured one. [Get a free key ↗]({provider_info['get_key_url']})"
    )

    st.divider()

    if st.session_state.persona is not None:
        st.subheader("Choose your guide")
        persona_choice = st.radio(
            label="Persona",
            options=list(PERSONAS.keys()),
            format_func=lambda p: f"{PERSONA_EMOJI.get(p, '')}  {p}",
            index=list(PERSONAS.keys()).index(st.session_state.persona),
            label_visibility="collapsed",
        )

        if persona_choice != st.session_state.persona:
            st.session_state.persona = persona_choice

        st.caption(PERSONA_TAGLINE.get(st.session_state.persona, ""))

        st.divider()
        if st.button("🗑️ Clear this conversation", use_container_width=True):
            st.session_state.messages_by_persona[st.session_state.persona] = []
            st.rerun()

    st.divider()
    st.caption(
        "Answers are generated using Gemini, grounded in retrieved "
        "passages from the Ramayana knowledge base. Each persona "
        "responds in its own distinct voice."
    )


# ---------------------------------------------------------------------------
# API key gate — nothing else shows until a key is entered
# ---------------------------------------------------------------------------
active_provider = st.session_state.provider
default_key = GOOGLE_API_KEY if active_provider == "Gemini" else OPENAI_API_KEY
effective_key = st.session_state.user_api_key.strip() or default_key
provider_info = AI_PROVIDERS[active_provider]

if not effective_key:
    st.title("🏹 Ramayana Knowledge Assistant")
    st.info(
        f"👋 To get started, enter your {active_provider} API key in the "
        f"sidebar and press Enter. It's free — get one at "
        f"[{provider_info['get_key_url']}]({provider_info['get_key_url']})."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Persona selection screen (shown until the user picks one)
# ---------------------------------------------------------------------------
if st.session_state.persona is None:
    st.title("🏹 Ramayana Knowledge Assistant")
    st.markdown("### Choose who you'd like to speak with:")
    st.write("")

    cols = st.columns(len(PERSONAS))
    for col, name in zip(cols, PERSONAS.keys()):
        with col:
            st.markdown(f"## {PERSONA_EMOJI.get(name, '')}")
            st.markdown(f"**{name}**")
            st.caption(PERSONA_TAGLINE.get(name, ""))
            if st.button(f"Speak with {name}", key=f"choose_{name}", use_container_width=True):
                st.session_state.persona = name
                st.rerun()

    st.stop()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
active_persona = st.session_state.persona
active_model = st.session_state.model
st.title(f"{PERSONA_EMOJI.get(active_persona, '')} Speaking with {active_persona}")
st.caption(PERSONA_TAGLINE.get(active_persona, ""))

# Each persona keeps its own separate conversation thread — switching
# personas doesn't carry chat history between them.
if active_persona not in st.session_state.messages_by_persona:
    st.session_state.messages_by_persona[active_persona] = []
active_messages = st.session_state.messages_by_persona[active_persona]

# Load the backend (shows a spinner only on first load / first index build)
try:
    with st.spinner("Loading knowledge base... (first run may take a while)"):
        pipeline = load_pipeline(api_key=effective_key, provider=active_provider, model=active_model)
except Exception as exc:
    logger.exception("Backend initialization failed")
    st.error(
        "Unable to initialize the knowledge base. This usually means the "
        "API key is invalid, or the data/ folder has no documents. Please "
        "check your key and try again."
    )
    st.stop()

# Render chat history
for i, msg in enumerate(active_messages):
    avatar = PERSONA_EMOJI.get(msg.get("persona"), None) if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📖 Sources used"):
                seen = set()
                for s in msg["sources"]:
                    if s["source"] not in seen:
                        st.markdown(f"**{s['source']}**")
                        st.caption(s["content"][:300] + "...")
                        seen.add(s["source"])
        if msg["role"] == "assistant" and msg.get("fallback_suggestions"):
            suggestions = msg["fallback_suggestions"]
            cols = st.columns(len(suggestions))
            for j, (col, sug) in enumerate(zip(cols, suggestions)):
                with col:
                    if st.button(sug, key=f"fallback_{i}_{j}"):
                        st.session_state.pending_question = sug
                        st.rerun()

# Voice input — lets people who can't type ask by speaking instead.
# Transcription always uses Gemini, so it needs a Gemini key specifically —
# not necessarily the same key used for chat generation if OpenAI is selected.
gemini_key_for_voice = (
    st.session_state.user_api_key.strip() if active_provider == "Gemini" else ""
) or GOOGLE_API_KEY

voice_question = None
if gemini_key_for_voice:
    st.markdown("**🎙️ Ask by voice**")
    audio_value = st.audio_input("Record your question", label_visibility="collapsed")

    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        audio_size_mb = len(audio_bytes) / (1024 * 1024)

        if audio_size_mb > MAX_AUDIO_SIZE_MB:
            st.warning(
                f"That recording is {audio_size_mb:.1f} MB, which is over "
                f"the {MAX_AUDIO_SIZE_MB} MB limit. Please record a shorter question."
            )
        else:
            audio_hash = hash(audio_bytes)
            if st.session_state.last_audio_hash != audio_hash:
                st.session_state.last_audio_hash = audio_hash
                with st.spinner("Transcribing your question..."):
                    try:
                        voice_question = transcribe_audio(audio_bytes, gemini_key_for_voice)
                        if voice_question:
                            st.success(f'Heard: "{voice_question}"')
                        else:
                            st.warning("Couldn't make out any speech in that recording — please try again.")
                    except Exception as exc:
                        logger.exception("Voice transcription failed")
                        st.error("Could not transcribe that recording. Please try again.")
else:
    st.caption("🎙️ Voice input needs a Gemini API key (used for speech-to-text only).")

# Chat input
typed_question = st.chat_input(f"Ask {active_persona} a question about the Ramayana...")
question = st.session_state.pop("pending_question", None) or typed_question or voice_question

# Disclaimer inserted directly into the same container as the chat input,
# so it automatically inherits the exact same responsive width — no
# guessed pixel values, it truly matches the message bar at any screen size.
components.html(
    """
    <script>
    function attachDisclaimer() {
        const doc = window.parent.document;
        const chatInput = doc.querySelector('[data-testid="stChatInput"]');
        if (!chatInput) { return false; }
        const container = chatInput.closest('[data-testid="stBottomBlockContainer"]') || chatInput.parentElement;
        if (!container) { return false; }

        const existing = container.querySelector('.ai-disclaimer-injected');
        if (existing) { existing.remove(); }

        const div = doc.createElement('div');
        div.className = 'ai-disclaimer-injected';
        div.style.fontSize = '0.7rem';
        div.style.color = '#808495';
        div.style.textAlign = 'center';
        div.style.padding = '0.3rem 0 0.4rem 0';
        div.style.width = '100%';
        div.style.boxSizing = 'border-box';
        div.innerText = "⚠️ This chatbot uses AI and may occasionally make mistakes, misinterpret context, or provide incomplete information. Please verify important facts independently.";
        container.appendChild(div);
        return true;
    }

    if (!attachDisclaimer()) {
        let tries = 0;
        const interval = setInterval(() => {
            tries += 1;
            if (attachDisclaimer() || tries > 20) clearInterval(interval);
        }, 200);
    }
    </script>
    """,
    height=0,
)

if question:
    if len(question) > MAX_QUESTION_LENGTH:
        st.warning(
            f"That question is a bit long ({len(question)} characters). "
            f"Please keep it under {MAX_QUESTION_LENGTH} characters."
        )
    elif not check_rate_limit():
        st.warning(
            f"You've asked a lot of questions in a short time! Please wait "
            f"a moment before asking another (limit: {RATE_LIMIT_MAX_REQUESTS} "
            f"per {RATE_LIMIT_WINDOW_SECONDS} seconds)."
        )
    else:
        active_messages.append({"role": "user", "content": question})

        with st.spinner(f"{active_persona} is reflecting..."):
            try:
                result = pipeline.answer(question, persona=active_persona)

                if not result.get("understood", True):
                    # The chatbot couldn't ground an answer in the knowledge
                    # base — store a friendly fallback message + suggestions;
                    # the history loop above renders the buttons on rerun.
                    fallback_msg = (
                        f"Hmm, I'm not sure how to answer that one! 🤔 "
                        f"Here are some questions {active_persona} loves to answer:"
                    )
                    active_messages.append(
                        {
                            "role": "assistant",
                            "content": fallback_msg,
                            "persona": active_persona,
                            "sources": [],
                            "fallback_suggestions": FALLBACK_QUESTIONS.get(active_persona, []),
                        }
                    )
                else:
                    active_messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "persona": active_persona,
                            "sources": result["sources"],
                        }
                    )
            except Exception as exc:
                logger.exception("Answer generation failed")
                active_messages.append(
                    {
                        "role": "assistant",
                        "content": "Something went wrong generating a response. Please try again.",
                        "persona": active_persona,
                    }
                )

        # Rerun so the new message(s) render through the same history loop
        # as everything else — keeps the voice input bar in a fixed spot
        # instead of the newest message appearing after it.
        st.rerun()