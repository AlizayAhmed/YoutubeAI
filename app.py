import os
import re
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download / transcript
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

# Groq (optional)
try:
    from groq_gradio import Groq
except Exception:
    Groq = None

# ---------- Config ----------
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 100
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
STORAGE_DIR = Path("./stored_indexes")
STORAGE_DIR.mkdir(exist_ok=True)


# ---------- Utilities ----------
def url_to_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def download_audio(youtube_url: str) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="yt_audio_"))
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    out_file = stream.download(output_path=str(tmpdir))
    return Path(out_file)


def get_transcript(youtube_url: str) -> str:
    video_id = YouTube(youtube_url).video_id
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception:
        return ""
    text = " ".join([x["text"] for x in transcript_list])
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
        if i < 0: i = 0
        if i >= n: break
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
    # Save faiss index and chunks metadata (simple .npy)
    faiss.write_index(idx, str(index_path / "index.faiss"))
    np.save(index_path / "chunks.npy", np.array(chunks, dtype=object), allow_pickle=True)



def load_index_and_meta(index_path: Path):
    idx = faiss.read_index(str(index_path / "index.faiss"))
    chunks = list(np.load(index_path / "chunks.npy", allow_pickle=True))
    return idx, chunks



def retrieve_top_k(query: str, idx: faiss.IndexFlatL2, chunks: List[str], embedder: SentenceTransformer, k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    D, I = idx.search(q_emb, k)
    results = []
    for pos in I[0]:
        if pos < len(chunks):
            results.append(chunks[pos])
    return results


# ---------- Groq Wrapper ----------
def groq_generate(prompt: str, model_name: str = "llama-3.3-70b-versatile", max_tokens: int = 512) -> str:
    if Groq is None:
        return "Groq not installed or API unavailable."
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        return "Groq API key not found in Streamlit secrets."
    client = Groq(api_key=api_key)
    resp = client.generate(model=model_name, prompt=prompt, max_tokens=max_tokens)
    if isinstance(resp, dict):
        return resp.get("text", "") or resp.get("output", "") or str(resp)
    return str(resp)


# ---------- Pipeline ----------
def process_video(youtube_url: str):
    status = "Starting..."
    transcript = ""
    summary = ""
    try:
        status = "Fetching transcript..."
        transcript = get_transcript(youtube_url)
        if not transcript:
            status += " Transcript not found, downloading audio..."
            audio_path = download_audio(youtube_url)
            status += f" Audio downloaded at {audio_path}, fallback to audio transcription required."

        status += " Chunking..."
        chunks = chunk_text(transcript)
        status += f" {len(chunks)} chunks created. Embedding..."
        embeddings = build_embeddings(chunks)
        idx = create_faiss_index(embeddings)
        vid_id = url_to_id(youtube_url)
        index_path = STORAGE_DIR / vid_id
        if index_path.exists():
            shutil.rmtree(index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        save_index_and_meta(idx, chunks, index_path)
        status += f" Index saved at {index_path}."

        # Groq summary
        try:
            summary_prompt = f"Summarize the following transcript:\n\n{transcript}\n\nSummary:"
            summary = groq_generate(summary_prompt, max_tokens=256)
            status += " Summary generated via Groq."
        except Exception:
            summary = transcript[:2000]

        return transcript, summary, status
    except Exception as e:
        return "", "", f"Error: {e}"


def chat_with_mentor(youtube_url: str, question: str):
    vid_id = url_to_id(youtube_url)
    index_path = STORAGE_DIR / vid_id
    if not index_path.exists():
        return "Video not processed. Please process it first."

    idx, chunks = load_index_and_meta(index_path)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    top_chunks = retrieve_top_k(question, idx, chunks, embedder)
    context_text = "\n\n---\n\n".join(top_chunks)
    prompt = (
        f"You are a helpful AI mentor.\nUse the following context to answer the user's question.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
    )
    answer = groq_generate(prompt, max_tokens=400)
    return answer


# ---------- Streamlit UI ----------
st.set_page_config(page_title="YouTube AI Mentor", layout="wide")
st.title("🎓 YouTube AI Mentor (RAG with Groq + FAISS)")

tab1, tab2 = st.tabs(["Process Video", "Chat with AI Mentor"])

with tab1:
    url_input = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    process_btn = st.button("Process Video")
    transcript_box = st.text_area("Transcript", height=250)
    summary_box = st.text_area("Summary", height=150)
    status_box = st.text_area("Status", height=100)

    if process_btn and url_input:
        transcript, summary, status = process_video(url_input)
        transcript_box = st.text_area("Transcript", value=transcript, height=250)
        summary_box = st.text_area("Summary", value=summary, height=150)
        status_box = st.text_area("Status", value=status, height=100)

with tab2:
    url_chat = st.text_input("YouTube Video URL (must be processed first)", placeholder="https://www.youtube.com/watch?v=...")
    question_input = st.text_input("Your Question")
    ask_btn = st.button("Ask")
    chat_output = st.text_area("AI Mentor Answer", height=300)

    if ask_btn and url_chat and question_input:
        answer = chat_with_mentor(url_chat, question_input)
        chat_output = st.text_area("AI Mentor Answer", value=answer, height=300)
