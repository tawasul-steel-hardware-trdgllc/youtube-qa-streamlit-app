import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from requests import Session
from streamlit.errors import StreamlitSecretNotFoundError
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

class TranscriptSession(Session):
    def __init__(self, verify_ssl=True):
        super().__init__()
        self.verify = verify_ssl

    def request(self, method, url, **kwargs):
        kwargs["verify"] = self.verify
        return super().request(method, url, **kwargs)

def get_config_value(key):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except StreamlitSecretNotFoundError:
        pass

    return os.getenv(key)


def get_openai_api_key():
    return get_config_value("OPENAI_API_KEY")


def create_youtube_transcript_api():
    http_proxy = get_config_value("YOUTUBE_PROXY_HTTP")
    https_proxy = get_config_value("YOUTUBE_PROXY_HTTPS")
    verify_ssl = str(get_config_value("YOUTUBE_VERIFY_SSL") or "true").lower() != "false"

    http_client = Session()
    http_client.verify = verify_ssl
    http_client = TranscriptSession(verify_ssl=verify_ssl)

    proxies = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy

    if proxies:
        http_client.proxies.update(proxies)

    return YouTubeTranscriptApi(http_client=http_client)


def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)

    if parsed_url.hostname in {"youtu.be", "www.youtu.be"}:
        return parsed_url.path.lstrip("/")

    if parsed_url.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]
        if parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/embed/")[1].split("/")[0]
        if parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/shorts/")[1].split("/")[0]

    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", youtube_url)
    return match.group(1) if match else None


def fetch_transcript(video_id, language):
    languages = [language, "en"] if language != "en" else ["en"]

    try:
        transcript = create_youtube_transcript_api().fetch(video_id, languages=languages)
    except TypeError:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)

    transcript_parts = []
    for item in transcript:
        if isinstance(item, dict):
            text = item.get("text")
        else:
            text = getattr(item, "text", None)

        if text:
            transcript_parts.append(text)

    return " ".join(transcript_parts)


def chunk_text(text, max_words=350, overlap_words=60):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = max(0, end - overlap_words)

    return chunks


def embed_texts(client, texts):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def cosine_similarity(query_embedding, document_embeddings):
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    document_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
    return np.dot(document_norms, query_norm)


def find_relevant_chunks(client, question, chunks, chunk_embeddings, top_k):
    query_embedding = embed_texts(client, [question])[0]
    scores = cosine_similarity(query_embedding, chunk_embeddings)
    best_indexes = np.argsort(scores)[-top_k:][::-1]
    return [(chunks[index], float(scores[index])) for index in best_indexes]


def answer_question(client, question, relevant_chunks):
    context = "\n\n---\n\n".join(chunk for chunk, _score in relevant_chunks)

    prompt = f"""You are answering questions about a YouTube video.

Use only the transcript context below. If the answer is not in the context, say that the video transcript does not provide enough information.

Question:
{question}

Transcript context:
{context}
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def reset_video_state():
    for key in ["video_id", "transcript", "chunks", "chunk_embeddings", "loaded_url"]:
        st.session_state.pop(key, None)


st.set_page_config(page_title="Chat With YouTube", page_icon="▶", layout="wide")

st.title("Chat With YouTube")

api_key = get_openai_api_key()

if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to your .env file locally or Streamlit Secrets when deployed.")
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.header("Settings")
    language = st.text_input("Transcript language", value="en")
    top_k = st.slider("Transcript chunks to use", min_value=1, max_value=5, value=3)
    max_words = st.slider("Words per chunk", min_value=150, max_value=700, value=350, step=50)
    overlap_words = st.slider("Chunk overlap", min_value=0, max_value=150, value=60, step=10)

    if st.button("Clear loaded video"):
        reset_video_state()
        st.rerun()

youtube_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

load_clicked = st.button("Load video transcript", type="primary")

if load_clicked:
    reset_video_state()
    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("Please enter a valid YouTube URL.")
        st.stop()

    try:
        with st.spinner("Fetching transcript..."):
            transcript = fetch_transcript(video_id, language)

        with st.spinner("Chunking and embedding transcript..."):
            chunks = chunk_text(transcript, max_words=max_words, overlap_words=overlap_words)
            chunk_embeddings = embed_texts(client, chunks)

        st.session_state["video_id"] = video_id
        st.session_state["loaded_url"] = youtube_url
        st.session_state["transcript"] = transcript
        st.session_state["chunks"] = chunks
        st.session_state["chunk_embeddings"] = chunk_embeddings

        st.success(f"Loaded transcript with {len(chunks)} chunks.")

    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript was found for this video in the selected language.")
    except Exception as exc:
        st.error(f"Could not load this video: {exc}")

if "chunks" in st.session_state:
    st.caption(f"Loaded video ID: {st.session_state['video_id']}")

    with st.expander("Preview transcript"):
        st.write(st.session_state["transcript"][:4000])

    question = st.text_area(
        "Ask a question about the video",
        placeholder="What are the main points in this video?",
        height=100,
    )

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please type a question first.")
            st.stop()

        with st.spinner("Searching transcript and generating answer..."):
            relevant_chunks = find_relevant_chunks(
                client=client,
                question=question,
                chunks=st.session_state["chunks"],
                chunk_embeddings=st.session_state["chunk_embeddings"],
                top_k=top_k,
            )
            answer = answer_question(client, question, relevant_chunks)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Transcript chunks used"):
            for index, (chunk, score) in enumerate(relevant_chunks, 1):
                st.markdown(f"**Chunk {index} - similarity score: {score:.3f}**")
                st.write(chunk)
else:
    st.info("Enter a YouTube URL and load the transcript to start chatting.")
