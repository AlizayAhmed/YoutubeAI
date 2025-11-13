import os
import re
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import numpy as np

# ----------------- Optional libraries -----------------
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

try:
    from pytube import YouTube
except Exception:
    YouTube = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

try:
    from groq_gradio import Groq
except Exception:
    Groq = None

# ----------------- Config -----------------
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 100
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TRANSCRIBE_WHISPER_MODEL = "tiny"
TOP_K = 5
STORAGE_DIR = Path("./stored_indexes")
STORAGE_DIR.mkdir(exist_ok=True)

# ----------------- Utilities -----------------
def url_to_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

def download_audio(youtube_url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if YouTube is None:
        raise RuntimeError("pytube not installed")
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    out_file = audio_stream.download(output_path=dest_dir)
    return Path(out_file)

def transcribe_audio_whisper(audio_path: Path) -> str:
    if WhisperModel is None:
        raise RuntimeError("faster-whisper not installed.")
    import torch
    use_cuda = torch.cuda.is_available()
    model = WhisperModel(
        TRANSCRIBE_WHISPER_MODEL,
        device="cuda" if use_cuda else "cpu",
        compute_type="int8" if not use_cuda else "float16",
    )
    segments, _ = model.transcribe(str(audio_path), beam_size=5)
    transcript = " ".join([seg.text.strip() for seg in segments])
    return re.sub(r"\s+", " ", transcript).strip()

def get_youtube_transcript(video_url: str) -> str:
    """Use YouTubeTranscriptApi if available (fastest for short videos)"""
    if YouTubeTranscriptApi is None:
        raise RuntimeError("youtube-transcript-api not installed")
    video_id = video_url.split("v=")[-1]
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = " ".join([t['text'] for t in transcript_list])
    return re.sub(r"\s+", " ", transcript).strip()

def chunk_text(text: str, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS) -> List[str]:
    if not text:
        return []
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end].strip())
        i = end - overlap
        if i < 0: i = 0
        if i >= n: break
    return [c for c in chunks if c]

def build_embeddings(chunks: List[str]):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    return embeddings

def create_faiss_index(embeddings: np.ndarray):
    if faiss is None:
        raise RuntimeError("faiss not installed")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index_and_meta(idx, chunks: List[str], index_path: Path):
    faiss.write_index(idx, str(index_path / "index.faiss"))
    np.save(index_path / "chunks.npy", np.array(chunks, dtype=object), allow_pickle=True)

def load_index_and_meta(index_path: Path):
    idx = faiss.read_index(str(index_path / "index.faiss"))
    chunks = list(np.load(index_path / "chunks.npy", allow_pickle=True))
    return idx, chunks

def retrieve_top_k(query: str, idx, chunks: List[str], embedder, k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    D, I = idx.search(q_emb, k)
    return [chunks[pos] for pos in I[0] if pos < len(chunks)]

# ----------------- Groq Wrapper -----------------
def groq_generate(prompt: str, model_name="llama-3.3-70b-versatile", max_tokens=512) -> str:
    if Groq is None:
        raise RuntimeError("groq-gradio not installed.")
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Groq API key not set in Streamlit secrets.")
    client = Groq(api_key=api_key)
    resp = client.generate(model=model_name, prompt=prompt, max_tokens=max_tokens)
    if isinstance(resp, dict):
        return resp.get("text") or resp.get("output") or str(resp)
    return str(resp)

# ----------------- Pipelines -----------------
def process_video_pipeline(youtube_url: str, use_transcript_api=True):
    transcript, summary, status, next_step = "", "", "Starting...", ""
    try:
        if use_transcript_api:
            transcript = get_youtube_transcript(youtube_url)
            status = "Transcript fetched via YouTube API."
        else:
            tmpdir = Path(tempfile.mkdtemp(prefix="yt_audio_"))
            audio_path = download_audio(youtube_url, tmpdir)
            transcript = transcribe_audio_whisper(audio_path)
            status = "Transcript generated via Whisper."

        # Groq summary
        try:
            summary_prompt = f"You are a helpful assistant. Summarize this transcript:\n\n{transcript}\n\nSummary:"
            summary = groq_generate(summary_prompt, max_tokens=256)
        except Exception:
            summary = transcript[:2000]

        # Chunking & embedding
        chunks = chunk_text(transcript)
        embeddings = build_embeddings(chunks)
        idx = create_faiss_index(embeddings)

        vid_id = url_to_id(youtube_url)
        index_path = STORAGE_DIR / vid_id
        if index_path.exists(): shutil.rmtree(index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        save_index_and_meta(idx, chunks, index_path)

        next_step = "You can now ask questions about this video."
        status = f"Processing complete. {len(chunks)} chunks created."
        return transcript, summary, status, next_step
    except Exception as e:
        return "", "", f"Error: {e}", ""

def chat_with_mentor(youtube_url: str, user_question: str, chat_history=None):
    if chat_history is None: chat_history = []
    try:
        vid_id = url_to_id(youtube_url)
        index_path = STORAGE_DIR / vid_id
        if not index_path.exists():
            return chat_history, "No index found. Process the video first."

        idx, chunks = load_index_and_meta(index_path)
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        top_contexts = retrieve_top_k(user_question, idx, chunks, embedder, k=TOP_K)
        context_text = "\n\n---\n\n".join(top_contexts)
        prompt = f"You are a helpful AI mentor. Use the context below to answer the question.\n\nContext:\n{context_text}\n\nQuestion: {user_question}\nAnswer:"

        try:
            answer = groq_generate(prompt, max_tokens=400)
        except Exception:
            answer = f"Fallback: Context retrieved:\n{context_text[:1500]}...\nPlease set GROQ_API_KEY for full AI answer."

        chat_history.append((user_question, answer))
        return chat_history, "Answer generated."
    except Exception as e:
        return chat_history, f"Error: {e}"

# ----------------- Streamlit UI -----------------
st.title("🎓 YouTube AI Mentor (RAG System)")

tab1, tab2 = st.tabs(["Process Video", "Chat with AI Mentor"])

with tab1:
    url_input = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    use_transcript_api = st.checkbox("Use YouTubeTranscriptAPI (faster for short videos)", value=True)
    if st.button("Process Video"):
        transcript, summary, status, next_step = process_video_pipeline(url_input, use_transcript_api)
        st.text_area("Transcript", transcript, height=200)
        st.text_area("Summary", summary, height=100)
        st.success(status)
        st.info(next_step)

with tab2:
    url_input_chat = st.text_input("YouTube Video URL for Chat")
    question_input = st.text_input("Your question")
    chat_history = st.session_state.get("chat_history", [])
    if st.button("Ask"):
        chat_history, status = chat_with_mentor(url_input_chat, question_input, chat_history)
        st.session_state["chat_history"] = chat_history
        for q, a in chat_history:
            st.markdown(f"**Q:** {q}\n\n**A:** {a}\n---")
        st.success(status)
