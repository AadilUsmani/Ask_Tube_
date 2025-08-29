# app/services.py
import os
import logging
import datetime
import traceback
import uuid
from typing import List, Tuple, Dict, Any, Optional

# YouTube transcript API shim (keeps compatibility)
try:
    import youtube_transcript_api as yta
except Exception:
    yta = None

if yta is not None:
    if not hasattr(yta, "FetchedTranscript"):
        class FetchedTranscript(list): pass
        setattr(yta, "FetchedTranscript", FetchedTranscript)
    YT = yta.YouTubeTranscriptApi
    if not hasattr(YT, "list_transcripts") and hasattr(YT, "list"):
        def _list_transcripts(video_id, *args, **kwargs):
            return YT().list(video_id, *args, **kwargs)
        setattr(YT, "list_transcripts", staticmethod(_list_transcripts))

from dotenv import load_dotenv
load_dotenv()

# LangChain / Pinecone imports
from langchain_community.document_loaders import YoutubeLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from pytube import YouTube
import tempfile
import os
# new deps
import yt_dlp
import whisper
import tempfile

# LangChain vectorstore wrapper for Pinecone
from langchain_community.vectorstores import Pinecone as LC_Pinecone
import pinecone

# Local helpers & LLM factories (assume app/llm.py and app/utils.py exist)
from .llm import get_answer_llm, get_cleaner_llm, get_embedder
from .utils import (
    md5,
    truncate_to_words,
    decide_word_limit_from_query,
    extract_video_id,
    chunk_id_for,
    extract_text_from_invoke_result,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants (match original)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
SUMMARIZE_MAX_CHARS = int(os.getenv("SUMMARIZE_MAX_CHARS", "10000"))

# Pinecone config (from .env) - Fixed to match your .env file
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # Your .env uses PINECONE_ENV
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not PINECONE_API_KEY or not PINECONE_ENV or not PINECONE_INDEX:
    logger.warning("Pinecone env vars are not fully set. Make sure PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX are set in .env")

# init pinecone client (will raise if config missing)
pinecone_index = None

def _init_pinecone_once():
    global pinecone_index
    try:
        if "pinecone_initialized" not in globals():
            # Initialize Pinecone with v2.2.4 API
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
            logger.info("Pinecone initialized with environment: %s", PINECONE_ENV)

            # Check if index exists
            try:
                existing_indexes = pinecone.list_indexes()
                logger.info("Found existing indexes: %s", existing_indexes)
                
                if PINECONE_INDEX not in existing_indexes:
                    logger.info("Creating new index: %s", PINECONE_INDEX)
                    # Create index if it doesn't exist
                    pinecone.create_index(
                        name=PINECONE_INDEX,
                        dimension=1536,  # update if using different embedding model
                        metric="cosine"
                    )
                    logger.info("Index %s created successfully", PINECONE_INDEX)
                else:
                    logger.info("Index %s already exists", PINECONE_INDEX)
                    
            except Exception as e:
                logger.warning("Could not list/create index, will try to connect directly: %s", e)
                # Try to connect to the index directly
                pass

            # Get the index
            pinecone_index = pinecone.Index(PINECONE_INDEX)
            logger.info("Successfully connected to index: %s", PINECONE_INDEX)

            globals()["pinecone_initialized"] = True
            logger.info("Pinecone initialization completed successfully")

        return pinecone_index

    except Exception as e:
        logger.exception("Failed to initialize Pinecone: %s", str(e))
        logger.error("Please check:")
        logger.error("1. PINECONE_API_KEY is correct")
        logger.error("2. PINECONE_ENV is correct (try: gcp-starter, us-west1-gcp, us-east1-aws)")
        logger.error("3. Network connectivity to Pinecone servers")
        logger.error("4. Your Pinecone account status")
        raise

# Module-level cache that replaces Streamlit session_state
VIDEO_CACHE: Dict[str, Dict[str, Any]] = {}

# Safe YoutubeLoader subclass (title fallback)
class SafeYoutubeLoader(YoutubeLoader):
    def _get_video_info(self):
        return {"title": f"Video: {self.video_id}", "description": "", "author": "Unknown"}
def fetch_youtube_transcript(video_id: str, lang_codes=None) -> Optional[str]:
    """Robust transcript fetch using list_transcripts() API."""
    lang_codes = lang_codes or ["en", "en-US", "en-GB"]
    
    # Note: YouTubeTranscriptApi doesn't support cookies directly
    # We'll rely on yt-dlp with cookies for authenticated access
    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)

        # Prefer manually created, then auto-generated
        for getter in (
            lambda: listing.find_manually_created_transcript(lang_codes),
            lambda: listing.find_generated_transcript(lang_codes),
        ):
            try:
                t = getter()
                pieces = t.fetch()  # list of dicts
                return "\n".join(p.get("text", "") for p in pieces if p.get("text", "").strip())
            except Exception:
                pass

        # Last resort: first available transcript
        for t in listing:
            pieces = t.fetch()
            return "\n".join(p.get("text", "") for p in pieces if p.get("text", "").strip())

        return None
    except (NoTranscriptFound, TranscriptsDisabled):
        logger.warning("No transcript available for %s", video_id)
        return None
    except Exception as e:
        logger.error("Transcript API failed for %s: %s", video_id, e)
        return None


def transcribe_with_whisper(video_url: str) -> str:
    """Download audio via yt-dlp (robust) and transcribe with Whisper."""
    tmpdir = tempfile.mkdtemp()
    outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
    
    # Add cookie support for YouTube authentication
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "noplaylist": True,
        "nocheckcertificate": True,
    }
    
    # Add cookies if available
    cookies_content = os.getenv("YOUTUBE_COOKIES_FILE")
    if cookies_content:
        # Create temporary cookies file from environment variable content
        import tempfile
        temp_cookies_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_cookies_file.write(cookies_content)
        temp_cookies_file.close()
        ydl_opts["cookiefile"] = temp_cookies_file.name
        logger.info("Using cookies from environment variable (temp file: %s)", temp_cookies_file.name)
    else:
        logger.info("No cookies found in environment variable, using default yt-dlp options")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            audio_path = ydl.prepare_filename(info)  # e.g., .m4a/.webm

        model = whisper.load_model("base")  # "small"/"medium" for higher accuracy
        result = model.transcribe(audio_path)
        try:
            os.remove(audio_path)
        except Exception:
            pass
        return (result.get("text") or "").strip()
    except Exception as e:
        logger.error("Whisper/YT-DLP failed for %s: %s", video_url, e)
        return ""
    finally:
        # Clean up temporary cookies file if it was created
        if 'temp_cookies_file' in locals():
            try:
                os.unlink(temp_cookies_file.name)
                logger.debug("Cleaned up temporary cookies file: %s", temp_cookies_file.name)
            except Exception as e:
                logger.debug("Failed to clean up temporary cookies file: %s", e)

# ---------------------------
# Low-level loader & chunking
# ---------------------------
def load_split_chunks(video_url: str) -> Tuple[List[Document], str]:
    # Try to use cookies for YouTube loader
    cookies_content = os.getenv("YOUTUBE_COOKIES_FILE")
    
    try:
        if cookies_content:
            # Create temporary cookies file from environment variable content
            import tempfile
            temp_cookies_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
            temp_cookies_file.write(cookies_content)
            temp_cookies_file.close()
            
            # Use cookies with YouTube loader
            loader = SafeYoutubeLoader.from_youtube_url(
                video_url, 
                add_video_info=False,
                cookies=temp_cookies_file.name
            )
            logger.info("Using cookies with YouTube loader from environment variable (temp file: %s)", temp_cookies_file.name)
        else:
            # Fallback to default loader
            loader = SafeYoutubeLoader.from_youtube_url(video_url, add_video_info=False)
            logger.info("Using default YouTube loader (no cookies)")
        
        docs: List[Document] = []
        raw = loader.load()
        if raw:
            docs = raw if isinstance(raw, list) else [raw]
    except Exception as e:
        logger.warning("YoutubeLoader failed for %s, will try direct API: %s", video_url, e)
    finally:
        # Clean up temporary cookies file if it was created
        if 'temp_cookies_file' in locals():
            try:
                os.unlink(temp_cookies_file.name)
                logger.debug("Cleaned up temporary cookies file: %s", temp_cookies_file.name)
            except Exception as e:
                logger.debug("Failed to clean up temporary cookies file: %s", e)

    # ✅ fallback to youtube_transcript_api if loader failed
    if not docs:
        vid = extract_video_id(video_url)
        raw_text = fetch_youtube_transcript(vid)
        if raw_text:
            docs = [Document(page_content=raw_text, metadata={"source": video_url})]

    # ✅ final fallback to Whisper if everything else failed
    if not docs:
        raw_text = transcribe_with_whisper(video_url)
        if raw_text:
            docs = [Document(page_content=raw_text, metadata={"source": video_url})]
        else:
            raise RuntimeError(f"Could not load transcript for {video_url}")
    # coerce non-Document
    if not isinstance(docs[0], Document):
        combined = ""
        if isinstance(docs, str):
            combined = docs
        elif isinstance(docs, list) and isinstance(docs[0], dict) and "text" in docs[0]:
            combined = "\n\n".join(p.get("text", "") for p in docs)
        else:
            combined = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
        docs = [Document(page_content=combined, metadata={"source": video_url})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(docs)

    for i, d in enumerate(split_docs):
        d.metadata = dict(d.metadata or {})
        d.metadata["chunk_index"] = i
        d.metadata["source"] = video_url

    # title fallback
    title = None
    try:
        title = docs[0].metadata.get("title")
    except Exception:
        title = None
    if not title:
        vid = extract_video_id(video_url)
        title = f"Video: {vid}" if vid else "YouTube Video"

    return split_docs, title

# ---------------------------
# Cache helpers (no Streamlit)
# ---------------------------
def ensure_video_cached(video_url: str) -> Dict[str, Any]:
    if video_url in VIDEO_CACHE:
        return VIDEO_CACHE[video_url]
    split_docs, title = load_split_chunks(video_url)
    VIDEO_CACHE[video_url] = {
        "raw_db": None,            # will hold LC_Pinecone wrapper or FAISS fallback
        "raw_chunks": split_docs,
        "cleaned_chunks": {},      # chunk_id -> cleaned_text
        "clean_db": None,
        "title": title,
        "last_processed_time": None,
        "chat_sessions": {},       # session_id -> list of messages
    }
    return VIDEO_CACHE[video_url]

def build_raw_db_if_missing(video_url: str) -> None:
    cache = ensure_video_cached(video_url)
    if cache.get("raw_db") is not None:
        return
    
    embedder = get_embedder()
    docs = cache["raw_chunks"]
    
    # Try Pinecone first
    try:
        _init_pinecone_once()
        # LangChain wrapper will upsert docs into Pinecone index_name
        cache["raw_db"] = LC_Pinecone.from_documents(docs, embedder, index_name=PINECONE_INDEX)
        logger.info("Built Pinecone-backed index for video %s (chunks=%d)", video_url, len(docs))
        return
    except Exception as e:
        logger.warning("Pinecone indexing failed for %s: %s", video_url, e)
        logger.info("Falling back to FAISS in-memory index")
    
    # Fallback to FAISS
    try:
        from langchain_community.vectorstores import FAISS
        cache["raw_db"] = FAISS.from_documents(docs, embedder)
        logger.info("Built FAISS fallback index for video %s (chunks=%d)", video_url, len(docs))
    except Exception as e:
        logger.error("Both Pinecone and FAISS failed for %s: %s", video_url, e)
        raise RuntimeError(f"Could not create vector index for {video_url}: {e}")

# ---------------------------
# Cleaning & optional re-embedding
# ---------------------------
def clean_chunk_lazy(video_url: str, chunk_doc: Document, reembed: bool = False) -> str:
    cache = ensure_video_cached(video_url)
    idx = int(chunk_doc.metadata.get("chunk_index", 0))
    cid = chunk_id_for(video_url, idx, chunk_doc.page_content)
    if cid in cache["cleaned_chunks"]:
        return cache["cleaned_chunks"][cid]

    cleaning_prompt = PromptTemplate(
        input_variables=["chunk"],
        template=(
            "You are a careful transcript copy editor.\n"
            "Fix grammar, punctuation, casing, and obvious ASR typos in this transcript chunk.\n"
            "Do NOT add facts or change meaning. Return only the corrected text.\n\n"
            "{chunk}"
        ),
    )
    chain = LLMChain(llm=get_cleaner_llm(), prompt=cleaning_prompt)
    try:
        res = chain.invoke({"chunk": chunk_doc.page_content})
        cleaned = extract_text_from_invoke_result(res).strip()
        if not cleaned:
            cleaned = chunk_doc.page_content
    except Exception:
        logger.exception("Cleaner LLM failed for %s chunk %s", video_url, idx)
        cleaned = chunk_doc.page_content

    cache["cleaned_chunks"][cid] = cleaned

    if reembed:
        try:
            new_doc = Document(page_content=cleaned, metadata=chunk_doc.metadata)
            if cache.get("clean_db") is None:
                try:
                    cache["clean_db"] = LC_Pinecone.from_documents([new_doc], get_embedder(), index_name=PINECONE_INDEX)
                except Exception:
                    from langchain_community.vectorstores import FAISS
                    cache["clean_db"] = FAISS.from_documents([new_doc], get_embedder())
            else:
                cache["clean_db"].add_documents([new_doc], get_embedder())
        except Exception:
            logger.exception("Re-embed cleaned chunk failed (non-fatal).")

    return cleaned

# ---------------------------
# Full clean for download
# ---------------------------
def clean_all_chunks_for_download(video_url: str) -> str:
    cache_entry = ensure_video_cached(video_url)
    raw_chunks = cache_entry["raw_chunks"]
    cleaned_parts: List[str] = []
    for chunk_doc in raw_chunks:
        try:
            cleaned_text = clean_chunk_lazy(video_url, chunk_doc, reembed=False)
        except Exception:
            logger.exception("Chunk cleaning failed, using raw content")
            cleaned_text = chunk_doc.page_content
        cleaned_parts.append(cleaned_text)

    full_transcript = "\n\n".join(cleaned_parts)
    cache_entry["last_processed_time"] = datetime.datetime.utcnow().isoformat()
    title = cache_entry.get("title", f"Video: {extract_video_id(video_url)}")
    header = (
        "CLEANED TRANSCRIPT\n"
        f"Title: {title}\n"
        f"URL: {video_url}\n"
        f"Generated: {cache_entry.get('last_processed_time', 'Unknown')}\n"
        + "=" * 50 + "\n\n"
    )
    return header + full_transcript

def generate_transcript_filename(video_url: str) -> str:
    vid = extract_video_id(video_url)
    return f"transcript_{vid}_cleaned.txt" if vid else "transcript_cleaned.txt"

# ---------------------------
# Chat/session support
# ---------------------------
def _make_session_id() -> str:
    return uuid.uuid4().hex

def get_or_create_session(video_url: str, session_id: Optional[str] = None) -> str:
    cache = ensure_video_cached(video_url)
    sessions = cache.setdefault("chat_sessions", {})
    if session_id and session_id in sessions:
        return session_id
    sid = session_id or _make_session_id()
    sessions.setdefault(sid, [])
    return sid

def append_chat_message(video_url: str, session_id: str, role: str, content: str) -> None:
    cache = ensure_video_cached(video_url)
    sessions = cache.setdefault("chat_sessions", {})
    hist = sessions.setdefault(session_id, [])
    hist.append({"role": role, "content": content, "ts": datetime.datetime.utcnow().isoformat()})
    # trim to keep tokens reasonable
    MAX_MSGS = 200
    if len(hist) > MAX_MSGS:
        del hist[:-100]

def get_chat_history(video_url: str, session_id: str):
    cache = ensure_video_cached(video_url)
    return cache.get("chat_sessions", {}).get(session_id, [])

# ---------------------------
# Retrieval & QA (RAG) with session memory
# ---------------------------
def retrieve_and_answer(
    video_url: str,
    query: str,
    session_id: Optional[str] = None,
    k: int = 4,
    length_choice: str = "Auto (smart)",
    fix_transcript: bool = True,
    reembed_cleaned: bool = False,
) -> Tuple[str, List[Document], str]:
    cache_entry = ensure_video_cached(video_url)

    # build index lazily
    if cache_entry.get("raw_db") is None:
        build_raw_db_if_missing(video_url)

    use_db = cache_entry.get("raw_db")
    # prefer clean_db when reembed requested and present
    if reembed_cleaned and cache_entry.get("clean_db"):
        use_db = cache_entry.get("clean_db")

    if use_db is None:
        raise RuntimeError("Vector store not available for this video (embedding failed).")

    # grammar-check query with the cleaner LLM (best-effort)
    cleaner_prompt = PromptTemplate(
        input_variables=["text"],
        template="Fix grammar/spelling in this short user query. Return only the cleaned query:\n\n{text}"
    )
    cleaner_chain = LLMChain(llm=get_cleaner_llm(), prompt=cleaner_prompt)
    try:
        clean_res = cleaner_chain.invoke({"text": query})
        clean_query = extract_text_from_invoke_result(clean_res)
    except Exception:
        clean_query = query

    retrieved = use_db.similarity_search(clean_query, k=k)

    docs_for_answer: List[Document] = []
    for doc in retrieved:
        if fix_transcript:
            cleaned = clean_chunk_lazy(video_url, doc, reembed=reembed_cleaned)
            docs_for_answer.append(Document(page_content=cleaned, metadata=doc.metadata))
        else:
            docs_for_answer.append(doc)

    docs_context = " ".join(d.page_content for d in docs_for_answer)
    if len(docs_context) > 12000:
        docs_context = docs_context[:12000] + "..."

    # conversation context (last few messages)
    sid = get_or_create_session(video_url, session_id)
    history = get_chat_history(video_url, sid)
    last_msgs = history[-6:] if history else []
    conv_history = "\n".join(f"{m['role']}: {m['content']}" for m in last_msgs)

    max_words = decide_word_limit_from_query(query, length_choice)

    qa_prompt = PromptTemplate(
        input_variables=["conv_history", "question", "docs", "max_words"],
        template=(
            "You are a concise assistant that answers using ONLY the provided transcript excerpts and recent conversation context.\n"
            "Be factual and avoid fabrication.\n\n"
            "Conversation history:\n{conv_history}\n\n"
            "Question: {question}\n\n"
            "Transcript excerpts:\n{docs}\n\n"
            "Answer (max {max_words} words):"
        )
    )

    qa_chain = LLMChain(llm=get_answer_llm(), prompt=qa_prompt)
    try:
        res = qa_chain.invoke({
            "conv_history": conv_history,
            "question": clean_query,
            "docs": docs_context,
            "max_words": str(max_words)
        })
        raw_answer = extract_text_from_invoke_result(res)
    except Exception:
        logger.exception("LLM error while answering for %s", video_url)
        raise RuntimeError("LLM request failed. See server logs.")

    final = truncate_to_words(raw_answer, max_words)
    append_chat_message(video_url, sid, "user", clean_query)
    append_chat_message(video_url, sid, "assistant", final)

    return final.replace("\n", " "), docs_for_answer, sid

# ---------------------------
# Summarizer
# ---------------------------
def summarize_video(video_url: str, max_words: int = 50) -> str:
    cache_entry = ensure_video_cached(video_url)
    raw_chunks = cache_entry["raw_chunks"]
    full_text = " ".join(d.page_content for d in raw_chunks)
    if len(full_text) > SUMMARIZE_MAX_CHARS:
        part = SUMMARIZE_MAX_CHARS // 3
        a = full_text[:part]
        mid = len(full_text) // 2
        b = full_text[mid - part // 2: mid + part // 2]
        c = full_text[-part:]
        full_text = "\n\n".join([a, b, c])

    summ_prompt = PromptTemplate(
        input_variables=["transcript", "max_w"],
        template="Summarize the following transcript in UNDER {max_w} words. Be factual and concise.\n\n{transcript}\n\nSummary:"
    )
    summ_chain = LLMChain(llm=get_answer_llm(), prompt=summ_prompt)
    try:
        res = summ_chain.invoke({"transcript": full_text, "max_w": str(max_words)})
        raw_summary = extract_text_from_invoke_result(res)
    except Exception:
        logger.exception("Summarizer failed for %s", video_url)
        raw_summary = "I couldn't produce a summary at this time."

    # clean pass
    clean_prompt = PromptTemplate(input_variables=["text"],
                                 template="Fix grammar/spelling in this short summary. Return only the cleaned summary:\n\n{text}")
    clean_chain = LLMChain(llm=get_cleaner_llm(), prompt=clean_prompt)
    try:
        res2 = clean_chain.invoke({"text": raw_summary})
        cleaned_summary = extract_text_from_invoke_result(res2)
    except Exception:
        cleaned_summary = raw_summary

    return truncate_to_words(cleaned_summary, max_words)

# ---------------------------
# Helper indexing wrapper
# ---------------------------
def index_video(video_url: str, build_index: bool = True) -> Dict[str, Any]:
    cache_entry = ensure_video_cached(video_url)
    if build_index and cache_entry.get("raw_db") is None:
        build_raw_db_if_missing(video_url)
    return {
        "title": cache_entry.get("title"),
        "chunks": len(cache_entry.get("raw_chunks", [])),
        "cleaned_cached": len(cache_entry.get("cleaned_chunks", {})),
        "has_index": cache_entry.get("raw_db") is not None,
    }

# ---------------------------
# Grammar check small wrapper
# ---------------------------
def check_query_grammar(query: str) -> str:
    clean_prompt = PromptTemplate(input_variables=["text"],
        template="Fix grammar/spelling in this short user query. Return only the cleaned query:\n\n{text}")
    chain = LLMChain(llm=get_cleaner_llm(), prompt=clean_prompt)
    try:
        res = chain.invoke({"text": query})
        return extract_text_from_invoke_result(res)
    except Exception:
        logger.exception("Grammar check failed")
        return query
