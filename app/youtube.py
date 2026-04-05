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

def get_video_comments(video_id: str, max_results: int = 20):
    youtube = get_youtube_client()
    response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        order="relevance"
    ).execute()

    comments = []
    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]
        comments.append({
            "video_id": video_id,
            "author": comment["authorDisplayName"],
            "text": comment["textDisplay"],
            "like_count": comment["likeCount"],
            "published_at": comment["publishedAt"],
            "language": comment.get("language", "unknown")
        })
    return comments