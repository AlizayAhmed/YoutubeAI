# app.py

"""
YouTube AI Mentor (RAG) prototype using:
- yt-dlp for audio download
- faster-whisper for transcription (free/local)
- sentence-transformers for embeddings (free)
- faiss for vector store
- groq_gradio for Groq Llama calls (optional; requires GROQ_API_KEY or hard-coded API key)
- streamlit for UI

Usage:
  pip install -r requirements.txt
  # set GROQ_API_KEY env var or adjust the hard-coded key below
  streamlit run app.py
"""

import os
import re
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# Transcription
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

# Download
import yt_dlp

# Embeddings & vector DB
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

# Groq (optional)
try:
    from groq_gradio import Groq
except Exception:
    Groq = None

# ---------- Config ----------
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 100
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TRANSCRIBE_WHISPER_MODEL = "tiny"
TOP_K = 5
STORAGE_DIR = Path("./stored_indexes")
STORAGE_DIR.mkdir(exist_ok=True)

# ---------- Utilities ----------
def url_to_id(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return h

def download_audio(youtube_url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(info)
    return Path(filename)

def transcribe_audio_whisper(audio_path: Path) -> str:
    if WhisperModel is None:
        raise RuntimeError("faster-whisper not installed.")
    use_cuda = False
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False
    model_size = TRANSCRIBE_WHISPER_MODEL
    if not use_cuda:
        if model_size not in ("tiny", "base", "small"):
            model_size = "small"
    preferred_compute = "float16" if use_cuda else "int8"
    try:
        model = WhisperModel(model_size, device="cuda" if use_cuda else "cpu", compute_type=preferred_compute)
    except Exception:
        model = WhisperModel(model_size, device="cuda" if use_cuda else "cpu")
    segments, _info = model.transcribe(str(audio_path), beam_size=5)
    texts = [seg.text.strip() for seg in segments]
    transcript = " ".join(texts)
    transcript = re.sub(r"\s+", " ", transcript).strip()
    return transcript

def chunk_text(text: str, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS) -> List[str]:
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunk = text[i:end]
        chunks.append(chunk.strip())
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

def save_index_and_meta(idx, chunks: List[str], index_path: Path):
    faiss.write_index(idx, str(index_path / "index.faiss"))
    np.save(index_path / "chunks.npy", np.array(chunks, dtype=object), allow_pickle=True)

def load_index_and_meta(index_path: Path) -> Tuple:
    idx = faiss.read_index(str(index_path / "index.faiss"))
    chunks = list(np.load(index_path / "chunks.npy", allow_pickle=True))
    return idx, chunks

def retrieve_top_k(query: str, idx, chunks: List[str], embedder: SentenceTransformer, k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    D, I = idx.search(q_emb, k)
    results = []
    for pos in I[0]:
        if pos < len(chunks):
            results.append(chunks[pos])
    return results

def groq_generate(prompt: str, model_name: str = "llama-3.3-70b-versatile", max_tokens: int = 512) -> str:
    if Groq is None:
        raise RuntimeError("groq-gradio not installed.")
    # Retrieve API key from environment or hardcode
    import os
    api_key = os.environ["GROQ_API_KEY"]

    if not api_key:
        raise RuntimeError("Groq API key not found.")
    client = Groq(api_key=api_key)
    resp = client.generate(model=model_name, prompt=prompt, max_tokens=max_tokens)
    if isinstance(resp, dict):
        return resp.get("text", "") or resp.get("output", "") or str(resp)
    return str(resp)

# ---------- Main pipeline functions ----------
def process_video_pipeline(youtube_url: str):
    status = "Starting processing..."
    next_step = ""
    transcript = ""
    summary = ""
    try:
        if not youtube_url or youtube_url.strip() == "":
            raise ValueError("Please provide a YouTube video URL.")
        status = "Downloading audio..."
        tmpdir = Path(tempfile.mkdtemp(prefix="yt_audio_"))
        audio_path = download_audio(youtube_url, tmpdir)
        status = f"Downloaded audio to {audio_path.name}. Transcribing..."
        transcript = transcribe_audio_whisper(audio_path)
        status = "Transcription complete."
        try:
            status += " Generating summary with Groq..."
            summary_prompt = (
                "You are a helpful assistant. Provide a concise summary of the following transcript:\n\n"
                f"{transcript}\n\nSummary:"
            )
            summary = groq_generate(summary_prompt, max_tokens=256)
            status += " Summary generated."
        except Exception as e:
            status += f" Groq summary not available ({e}). Using fallback."
            summary = transcript[:2000]
        status += " Chunking transcript..."
        chunks = chunk_text(transcript)
        status += f" {len(chunks)} chunks created. Embedding..."
        embeddings = build_embeddings(chunks)
        status += " Embeddings created. Building FAISS index..."
        idx = create_faiss_index(embeddings)
        status += " FAISS index created."
        vid_id = url_to_id(youtube_url)
        index_path = STORAGE_DIR / vid_id
        if index_path.exists():
            shutil.rmtree(index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        save_index_and_meta(idx, chunks, index_path)
        status += f" Index saved at {index_path}."
        next_step = "You can now ask questions about this video in the chat section."
        return transcript, summary, status, next_step
    except Exception as ex:
        return "", "", f"Error: {str(ex)}", "Processing failed."

def chat_with_mentor(youtube_url: str, user_question: str, chat_history: List[Tuple[str, str]] = None):
    if chat_history is None:
        chat_history = []
    status = ""
    if not youtube_url or not user_question:
        return chat_history, "Please provide both a YouTube URL and a question."
    try:
        vid_id = url_to_id(youtube_url)
        index_path = STORAGE_DIR / vid_id
        if not index_path.exists():
            return chat_history, "No index found. Process the video first."
        idx, chunks = load_index_and_meta(index_path)
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        top_contexts = retrieve_top_k(user_question, idx, chunks, embedder, k=TOP_K)
        context_text = "\n\n---\n\n".join(top_contexts)
        prompt = (
            "You are a helpful AI mentor. Use the retrieved transcript/context to answer the user's question. "
            "When you don't know, say you don't know. Keep answers concise but thorough.\n\n"
            "Retrieved context:\n"
            f"{context_text}\n\nUser question: {user_question}\n\nAnswer:"
        )
        try:
            answer = groq_generate(prompt, max_tokens=400)
        except Exception as e:
            answer = (
                "NOTE: Groq Llama not available or failed. Returning fallback.\n\n"
                f"Top retrieved context:\n{context_text[:1500]}...\n\nSuggested answer: The context above may help."
            )
            status = f"Warning: Groq generation failed ({e})."
        else:
            status = "Answer generated successfully."
        chat_history.append((user_question, answer))
        return chat_history, status
    except Exception as ex:
        return chat_history, f"Error during chat: {str(ex)}"

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="YouTube AI Mentor (RAG)", layout="wide")
    st.title("🎓 YouTube AI Mentor (RAG System using Groq + FAISS)")

    tab1, tab2 = st.tabs(["Process Video", "Chat with AI Mentor"])

    with tab1:
        url_input = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        if st.button("Process Video"):
            transcript, summary, status_msg, next_msg = process_video_pipeline(url_input)
            st.text_area("Transcript (copyable)", transcript, height=200)
            st.text_area("Summary", summary, height=100)
            st.success(status_msg)
            st.info(next_msg)

    with tab2:
        url_input_chat = st.text_input("YouTube Video URL (must be processed first)", placeholder="https://www.youtube.com/watch?v=...")
        question_input = st.text_input("Your question", placeholder="Ask something about the video...")
        if st.button("Ask"):
            # maintain chat history in session state
            if "history" not in st.session_state:
                st.session_state.history = []
            new_history, status_msg = chat_with_mentor(url_input_chat, question_input, st.session_state.history)
            st.session_state.history = new_history
            for user_q, bot_a in new_history:
                st.write("**You:**", user_q)
                st.write("**AI Mentor:**", bot_a)
                st.write("---")
            st.success(status_msg)

if __name__ == "__main__":
    main()
