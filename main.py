# main.py
import io
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import services

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="YouTube Video Assistant (RAG)", version="1.0")

# Request models
class IndexRequest(BaseModel):
    video_url: str
    build_index: Optional[bool] = True

class QueryRequest(BaseModel):
    video_url: str
    query: str
    session_id: Optional[str] = None
    k: Optional[int] = 4
    length_choice: Optional[str] = "Auto (smart)"
    fix_transcript: Optional[bool] = True
    reembed_cleaned: Optional[bool] = False

class SummarizeRequest(BaseModel):
    video_url: str
    max_words: Optional[int] = 50

class CheckGrammarRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/index")
def index_video(req: IndexRequest):
    try:
        meta = services.index_video(req.video_url, build_index=req.build_index)
        return meta
    except Exception as e:
        logger.exception("Indexing error for %s", req.video_url)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query")
def query_video(req: QueryRequest):
    try:
        answer, docs, session_id = services.retrieve_and_answer(
            req.video_url,
            req.query,
            session_id=req.session_id,
            k=req.k,
            length_choice=req.length_choice,
            fix_transcript=req.fix_transcript,
            reembed_cleaned=req.reembed_cleaned,
        )
        sources = [{"chunk_index": d.metadata.get("chunk_index"), "snippet": (d.page_content or "")[:400]} for d in docs]
        return {"answer": answer, "sources": sources, "session_id": session_id}
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    try:
        summary = services.summarize_video(req.video_url, max_words=req.max_words)
        return {"summary": summary}
    except Exception as e:
        logger.exception("Summarize failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcript")
def download_transcript(video_url: str = Query(..., description="YouTube video URL")):
    try:
        text = services.clean_all_chunks_for_download(video_url)
        filename = services.generate_transcript_filename(video_url)
        return StreamingResponse(io.BytesIO(text.encode("utf-8")), media_type="text/plain",
                                 headers={"Content-Disposition": f"attachment; filename={filename}"})
    except Exception as e:
        logger.exception("Transcript download failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check-grammar")
def check_grammar(req: CheckGrammarRequest):
    try:
        corrected = services.check_query_grammar(req.query)
        return {"corrected_query": corrected}
    except Exception as e:
        logger.exception("Grammar check failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def status(video_url: Optional[str] = None):
    try:
        if video_url:
            if video_url not in services.VIDEO_CACHE:
                raise HTTPException(status_code=404, detail="Video not cached")
            entry = services.VIDEO_CACHE[video_url]
            return {
                "video_url": video_url,
                "title": entry.get("title"),
                "chunks": len(entry.get("raw_chunks", [])),
                "cleaned_cached": len(entry.get("cleaned_chunks", {})),
                "has_index": entry.get("raw_db") is not None,
                "last_processed_time": entry.get("last_processed_time"),
            }
        else:
            return {"cached_videos": list(services.VIDEO_CACHE.keys())}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Status check failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
