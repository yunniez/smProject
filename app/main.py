from fastapi import FastAPI, Depends, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from app.youtube import search_videos
from app.database import SessionLocal, init_db, Video
from app.insight import generate_insight
from app.melon import get_melon_chart
from app.crawler import crawl_youtube_comments, save_comments
from app.database import SessionLocal, init_db, Video, Comment  # Comment 추가
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app = FastAPI(title="FandomLens", description="K-POP Fandom Intelligence API")
templates = Jinja2Templates(directory="app/templates")

@app.on_event("startup")
def startup():
    init_db()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/videos")
def get_videos(query: str = "aespa", max_results: int = 10):
    videos = search_videos(query, max_results)
    return {"videos": videos}

@app.post("/videos/collect")
def collect_videos(query: str = "aespa", max_results: int = 10, db: Session = Depends(get_db)):
    videos = search_videos(query, max_results)
    saved = 0
    for v in videos:
        existing = db.query(Video).filter(Video.video_id == v["video_id"]).first()
        if not existing:
            video = Video(**v)
            db.add(video)
            saved += 1
    db.commit()
    return {"saved": saved, "total": len(videos)}

@app.get("/videos/insight")
def get_insight(query: str = "aespa", max_results: int = 10, db: Session = Depends(get_db)):
    videos = db.query(Video).filter(Video.title.ilike(f"%{query}%")).limit(max_results).all()
    if not videos:
        return {"insight": "데이터가 없습니다. /videos/collect 먼저 실행해주세요."}
    insight = generate_insight(videos)
    return {"insight": insight}

@app.get("/chart/melon")
def melon_chart(limit: int = 10):
    chart = get_melon_chart(limit)
    return {"chart": chart}

@app.post("/comments/collect")
def collect_comments(video_id: str, max_comments: int = 100):
    comments = crawl_youtube_comments(video_id, max_comments)
    saved_count = save_comments(video_id, comments)
    return {
        "video_id": video_id,
        "crawled": len(comments),
        "saved": saved_count
    }

@app.get("/comments/{video_id}")
def get_comments(video_id: str):
    """저장된 댓글 조회"""
    db = SessionLocal()
    try:
        comments = db.query(Comment).filter_by(video_id=video_id).all()
        return {"video_id": video_id, "count": len(comments), "comments": comments}
    finally:
        db.close()

@app.post("/pipeline/run")
def run_pipeline(artist: str, max_videos: int = 5, max_comments: int = 100):
    db = SessionLocal()
    try:
        videos = search_videos(artist, max_videos)
        for v in videos:
            existing = db.query(Video).filter(Video.video_id == v["video_id"]).first()
            if not existing:
                db.add(Video(**v))
        db.commit()

        total_comments = 0
        for video in videos:
            comments = crawl_youtube_comments(video["video_id"], max_comments)
            saved = save_comments(video["video_id"], comments)
            total_comments += saved

        return {
            "artist": artist,
            "videos_collected": len(videos),
            "total_comments_saved": total_comments
        }
    finally:
        db.close()