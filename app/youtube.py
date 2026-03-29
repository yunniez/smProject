from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_youtube_client():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def search_videos(query: str, max_results: int = 10):
    youtube = get_youtube_client()
    response = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=max_results
    ).execute()

    videos = []
    for item in response["items"]:
        if item["id"].get("videoId") is None:
            continue
        videos.append({
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "published_at": item["snippet"]["publishedAt"],
            "description": item["snippet"]["description"]
        })
    return videos