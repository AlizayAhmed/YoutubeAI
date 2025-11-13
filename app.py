# app_streamlit.py

import os
import re
import hashlib
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import numpy as np
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

# Embeddings & FAISS
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

# Optional Groq
try:
    from groq_gradio import Groq
except Exception:
    Groq = None

# ---------------- Config ----------------
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 100
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
STORAGE_DIR = Path("./stored_indexes")
STORAGE_DIR.mkdir(exist_ok=True)

# ---------------- Utilities ----------------
def url_to_id(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return h

def download_audio(youtube_url: str, dest_dir: Path) -> Path:
    """Download audio using pytube."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    file_path = stream.download(output_path=str(dest_dir))
    return Path(file_path)

def get_transcript(youtube_url: str) -> str:
    """Fetch transcript using youtube-transcript-api (fallback to pytube captions if needed)."""
    try:
        video_id = YouTube(youtube_url).video_id
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([t["text"] for t in transcript_list])
        return re.sub(r"\s+", " ", transcript).strip()
    except Exception as e:
        return f"Transcript not available: {e}"

def chunk_text(text: str, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS) -> List[str]:
    if not text:
        return []
    chunks = []
    i, n = 0, len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end].strip())
        i = end - overlap
        if i < 0:
            i = 0
        if i >= n:
            break
    return [c for c in chunks if c]

def build_embeddings(chunks: List[str]):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed.")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    return embeddings

def create_faiss_index(embeddings: np.ndarray):
    if faiss is None:
        raise RuntimeError("faiss not installed.")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index_and_meta(idx: faiss.IndexFlatL2, chunks: List[str], index_path: Path):
    faiss.write_index(idx, str(index_path / "index.faiss"))
    np.save(index_path / "chunks.npy", np.array(chunks, dtype=object), allow_pickle=True)

def load_index_and_meta(index_path: Path) -> Tuple[faiss.IndexFlatL2, List[str]]:
    idx = faiss.read_index(str(index_path / "index.faiss"))
    chunks = list(np.load(index_path / "chunks.npy", allow_pickle=True))
    return idx, chunks

def retrieve_top_k(query: str, idx: faiss.IndexFlatL2, chunks: List[str], embedder: SentenceTransformer, k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    D, I = idx.search(q_emb, k)
    results = [chunks[pos] for pos in I[0] if pos < len(chunks)]
    return results

def groq_generate(prompt: str, model_name: str = "llama-3.3-70b-versatile", max_tokens: int = 512) -> str:
    if Groq is None:
        return "Groq not installed or API unavailable."
    api_key = "YOUR_GROQ_API_KEY_HERE"  # hardcoded
    client = Groq(api_key=api_key)
    resp = client.generate(model=model_name, prompt=prompt, max_tokens=max_tokens)
    if isinstance(resp, dict):
        return resp.get("text", "") or resp.get("output", "") or str(resp)
    return str(resp)

# ---------------- Streamlit UI ----------------
st.title("🎓 YouTube AI Mentor (RAG System)")

tab1, tab2 = st.tabs(["Process Video", "Chat with AI Mentor"])

with tab1:
    youtube_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    process_btn = st.button("Process Video")
    transcript_box = st.text_area("Transcript", height=200)
    summary_box = st.text_area("Summary", height=100)
    status_box = st.text_area("Status", height=80)

    if process_btn and youtube_url:
        status_box.text("Downloading audio...")
        tmpdir = Path(tempfile.mkdtemp(prefix="yt_audio_"))
        audio_path = download_audio(youtube_url, tmpdir)
        status_box.text("Audio downloaded. Fetching transcript...")
        transcript = get_transcript(youtube_url)
        transcript_box.text(transcript[:20000])
        status_box.text("Transcript fetched. Chunking...")
        chunks = chunk_text(transcript)
        status_box.text(f"{len(chunks)} chunks created. Embedding...")
        embeddings = build_embeddings(chunks)
        status_box.text("Embedding done. Creating FAISS index...")
        idx = create_faiss_index(embeddings)
        vid_id = url_to_id(youtube_url)
        index_path = STORAGE_DIR / vid_id
        index_path.mkdir(parents=True, exist_ok=True)
        save_index_and_meta(idx, chunks, index_path)
        status_box.text(f"Index saved at {index_path}. You can now chat with AI Mentor.")
        summary_box.text(transcript[:2000] + "...")

with tab2:
    youtube_url_chat = st.text_input("YouTube Video URL (processed)", placeholder="https://www.youtube.com/watch?v=...")
    user_question = st.text_input("Your question")
    ask_btn = st.button("Ask AI Mentor")
    chat_history = st.container()

    if ask_btn and youtube_url_chat and user_question:
        vid_id = url_to_id(youtube_url_chat)
        index_path = STORAGE_DIR / vid_id
        if not index_path.exists():
            st.warning("No index found. Process the video first.")
        else:
            idx, chunks = load_index_and_meta(index_path)
            embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
            top_contexts = retrieve_top_k(user_question, idx, chunks, embedder)
            context_text = "\n\n---\n\n".join(top_contexts)
            prompt = (
                "You are a helpful AI mentor. Use the retrieved transcript/context to answer user's question.\n\n"
                f"Retrieved context:\n{context_text}\n\n"
                f"User question: {user_question}\n\nAnswer:"
            )
            answer = groq_generate(prompt, max_tokens=400)
            st.markdown(f"**Answer:** {answer}")
