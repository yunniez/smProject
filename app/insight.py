import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def generate_insight(videos: list) -> str:
    video_list = "\n".join([
        f"- {v.title} ({v.channel}, {v.published_at[:10]})"
        for v in videos
    ])

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""다음은 유튜브에서 수집한 K-POP 영상 데이터입니다.
이 데이터를 분석해서 팬덤 트렌드와 인사이트를 한국어로 요약해주세요.

{video_list}

다음 항목을 포함해주세요:
1. 최근 활동 트렌드
2. 주요 채널 분포
3. 글로벌 팬덤 관점에서의 시사점
"""
            }
        ]
    )
    return message.content[0].text