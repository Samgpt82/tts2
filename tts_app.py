import io
import os
import uuid
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

def chunk_text(text: str, max_len: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    chunks, buf = [], []
    length = 0
    for part in text.split():
        if length + len(part) + 1 > max_len:
            chunks.append(" ".join(buf))
            buf, length = [part], len(part)
        else:
            buf.append(part)
            length += len(part) + 1
    if buf:
        chunks.append(" ".join(buf))
    return chunks

@st.cache_resource(show_spinner=False)
def get_client():
    # works with Streamlit Secrets or env var
    key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not key:
        st.error("Missing OPENAI_API_KEY")
        st.stop()
    return OpenAI(api_key=key)

def _tts_once(client, model: str, voice: str, text: str) -> bytes:
    """Call OpenAI Speech API, compatible with SDKs that use `format` or `response_format`."""
    try:
        # newer SDKs
        r = client.audio.speech.create(model=model, voice=voice, input=text, format="mp3")
    except TypeError:
        # some SDKs use response_format instead
        r = client.audio.speech.create(model=model, voice=voice, input=text, response_format="mp3")

    # normalize to bytes across SDK variants
    if hasattr(r, "content") and r.content is not None:
        return r.content
    if hasattr(r, "read"):
        return r.read()
    if isinstance(r, (bytes, bytearray)):
        return bytes(r)
    return getattr(r, "data", b"")

def synthesize_tts(chunks, voice: str, model: str) -> bytes:
    client = get_client()
    out = io.BytesIO()
    for chunk in chunks:
        out.write(_tts_once(client, model, voice, chunk))
    return out.getvalue()

# -----------------------
# UI
# -----------------------
text = st.text_area("Text", placeholder="Type or paste text to speak...", height=200)
col1, col2 = st.columns(2)
with col1:
    voice = st.selectbox("Voice", ALLOWED_VOICES, index=0)
with col2:
    model = st.text_input("Model", value=DEFAULT_MODEL)

if st.button("Generate", type="primary", disabled=not text.strip()):
    with st.spinner("Synthesizing..."):
        chunks = chunk_text(text)
        audio_bytes = synthesize_tts(chunks, voice, model)
        fname = f"{voice}-{uuid.uuid4().hex[:8]}.mp3"
        (OUTPUT_DIR / fname).write_bytes(audio_bytes)

    st.success("Done!")
    st.audio(io.BytesIO(audio_bytes), format="audio/mp3")
    st.download_button("Download MP3", audio_bytes, file_name=fname, mime="audio/mpeg")

st.caption("Set OPENAI_API_KEY in Secrets (Streamlit Cloud) or as an environment variable.")
