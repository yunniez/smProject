from fastapi import FastAPI
from app.youtube import search_videos

app = FastAPI()

@app.get("/")
def root():
    return {"message": "SM Fan Insight API"}

@app.get("/videos")
def get_videos(query: str = "aespa", max_results: int = 10):
    videos = search_videos(query, max_results)
    return {"videos": videos}