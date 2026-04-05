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
                "content": f"""당신은 SM Entertainment 소속 아티스트의 글로벌 팬덤을 분석하는 데이터 애널리스트입니다.
아래는 YouTube에서 수집한 SM 아티스트 관련 영상 데이터입니다.

{video_list}

다음 항목을 SM Entertainment의 비즈니스 관점에서 한국어로 분석해주세요:

1. **최근 활동 트렌드**: 어떤 아티스트/콘텐츠가 주목받고 있는지
2. **채널 분포 분석**: 공식 채널 vs 팬 채널 비율, 주요 업로더
3. **글로벌 팬덤 시사점**: 해외 팬덤 반응 및 확산 가능성
4. **콘텐츠 전략 제언**: SM이 활용할 수 있는 인사이트
"""
            }
        ]
    )
    return message.content[0].text


def analyze_sentiment(video_id: str, comments: list) -> dict:
    """댓글 감성 분석 - 긍정/부정/중립 분류"""
    
    comment_text = "\n".join([
        f"- {c.text}" for c in comments[:50]  # 최대 50개
    ])

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""당신은 K-POP 팬덤 감성 분석 전문가입니다.
아래는 YouTube 영상(ID: {video_id})에 달린 팬 댓글입니다.

{comment_text}

다음을 JSON 형식으로 분석해주세요. JSON 외에 다른 텍스트는 출력하지 마세요:
{{
    "positive_ratio": 긍정 비율(0~100 숫자),
    "negative_ratio": 부정 비율(0~100 숫자),
    "neutral_ratio": 중립 비율(0~100 숫자),
    "dominant_emotion": "주요 감정 키워드 (예: 감동, 기대, 응원, 실망 등)",
    "summary": "전체 팬 반응 요약 2~3문장",
    "top_positive": "대표 긍정 반응 1문장",
    "top_negative": "대표 부정 반응 1문장 (없으면 null)",
    "fandom_keywords": ["핵심 키워드1", "핵심 키워드2", "핵심 키워드3"]
}}
"""
            }
        ]
    )
    
    import json
    raw = message.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)