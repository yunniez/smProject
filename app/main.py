from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from app.youtube import search_videos
from app.database import SessionLocal, init_db, Video

app = FastAPI()

@app.on_event("startup")
def startup():
    init_db()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "SM Fan Insight API"}

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