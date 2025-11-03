import io
import os
import uuid
import base64
from pathlib import Path
from typing import List

import streamlit as st
from openai import OpenAI

# -----------------------
# Config
# -----------------------
DEFAULT_MODEL = "gpt-4o-mini-tts"
ALLOWED_VOICES = [
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "coral", "verse", "ballad", "ash", "sage", "marin", "cedar"
]
MAX_CHARS_PER_CHUNK = 4000
OUTPUT_DIR = Path("tts_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="TTS", page_icon="🔊", layout="centered")
st.title("🔊 Text to Speech")

# -----------------------
# Helpers
# -----------------------
def chunk_text(text: str, max_len: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """Split text into roughly max_len-sized chunks on whitespace."""
    chunks, buf = [], []
    length = 0
    for part in text.split():
        add_len = len(part) + (1 if buf else 0)
        if length + add_len > max_len:
            if buf:
                chunks.append(" ".join(buf))
            buf, length = [part], len(part)
        else:
            buf.append(part)
            length += add_len
    if buf:
        chunks.append(" ".join(buf))
    return chunks

@st.cache_resource(show_spinner=False)
def get_client():
    key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not key:
        st.error("Missing OPENAI_API_KEY (set in Secrets or environment).")
        st.stop()
    return OpenAI(api_key=key)

def _tts_once(client, model: str, voice: str, text: str) -> bytes:
    """
    Call OpenAI Speech API. Supports SDKs that use `format` or `response_format`.
    Returns raw MP3 bytes.
    """
    try:
        r = client.audio.speech.create(model=model, voice=voice, input=text, format="mp3")
    except TypeError:
        r = client.audio.speech.create(model=model, voice=voice, input=text, response_format="mp3")

    # Normalize to bytes across SDK variants
    if hasattr(r, "content") and r.content is not None:
        return r.content
    if hasattr(r, "read"):
        return r.read()
    if isinstance(r, (bytes, bytearray)):
        return bytes(r)
    return getattr(r, "data", b"")

def synthesize_tts(chunks: List[str], voice: str, model: str) -> bytes:
    client = get_client()
    out = io.BytesIO()
    for chunk in chunks:
        out.write(_tts_once(client, model, voice, chunk))
    return out.getvalue()

def render_audio(audio_bytes: bytes, file_name: str):
    """
    Primary player: Streamlit audio with proper MIME for iOS.
    Fallback: raw HTML5 <audio> tag (more tolerant on mobile).
    Always shows a download button.
    """
    ok = False
    try:
        # Safari expects standards-compliant MIME
        st.audio(io.BytesIO(audio_bytes), format="audio/mpeg")
        ok = True
    except Exception as e:
        st.caption(f"Player error: {e}")

    if not ok:
        b64 = base64.b64encode(audio_bytes).decode()
        st.markdown(
            f"""
            <audio controls style="width:100%">
              <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
              Your browser does not support the audio element.
            </audio>
            """,
            unsafe_allow_html=True,
        )

    st.download_button("Download MP3", audio_bytes, file_name=file_name, mime="audio/mpeg")

# -----------------------
# Input Section (Text OR .txt file)
# -----------------------
st.subheader("Input Text")
uploaded_file = st.file_uploader("Upload a .txt file (optional)", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    st.success("✅ Text file loaded.")
    st.text_area("Preview File Content", text, height=200)
else:
    text = st.text_area("Or enter text manually:", placeholder="Type or paste text here...", height=200)

# Voice / Model selection
col1, col2 = st.columns(2)
with col1:
    voice = st.selectbox("Voice", ALLOWED_VOICES, index=0)
with col2:
    model = st.text_input("Model", value=DEFAULT_MODEL)

# Generate
gen = st.button("Generate", type="primary", disabled=not text.strip())

if gen:
    with st.spinner("Synthesizing..."):
        try:
            chunks = chunk_text(text)
            st.caption(f"Chunked into {len(chunks)} piece(s)")
            audio_bytes = synthesize_tts(chunks, voice, model)
            size = len(audio_bytes)
            st.caption(f"Generated {size} bytes")
            if size == 0:
                st.error("No audio returned. Check API key, model name, or logs.")
            else:
                fname = f"{voice}-{uuid.uuid4().hex[:8]}.mp3"
                (OUTPUT_DIR / fname).write_bytes(audio_bytes)
                st.success("Done!")
                render_audio(audio_bytes, fname)
        except Exception as e:
            st.error(f"Error while generating audio: {e}")

st.caption("Set OPENAI_API_KEY in Secrets (Streamlit Cloud) or as an environment variable.")
