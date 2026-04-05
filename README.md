# 🎯 FandomLens

> K-POP 아티스트 팬덤 데이터를 자동 수집·분석하고, Claude AI가 감성 인사이트를 생성하는 엔드투엔드 데이터 파이프라인

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![Claude API](https://img.shields.io/badge/Claude-API-8B5CF6?style=flat-square)](https://anthropic.com)
[![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/features/actions)

---

## 📌 프로젝트 개요

FandomLens는 SM Entertainment 아티스트를 중심으로 YouTube 영상 데이터와 팬 댓글을 **자동 수집 → PostgreSQL 적재 → Claude AI 감성 분석**까지 이어지는 데이터 파이프라인입니다.

엔터테인먼트 기업의 **자체 AI 엔진 학습용 데이터 수집 자동화** 니즈를 직접 구현한 프로젝트로, 글로벌 팬덤 트렌드 인사이트 도출을 목표로 합니다.

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| **Language** | Python 3.12 |
| **Backend** | FastAPI 0.135 |
| **Database** | PostgreSQL + SQLAlchemy ORM |
| **크롤링** | Playwright (동적 렌더링 대응) + BeautifulSoup |
| **데이터 수집** | YouTube Data API v3 |
| **AI 분석** | Anthropic Claude API |
| **인프라** | AWS EC2 (Ubuntu 22.04) |
| **배포 자동화** | GitHub Actions CI/CD |
| **문서화** | Swagger UI (FastAPI 내장) |

---

## ✨ 주요 기능

### 1. YouTube 영상 데이터 자동 수집
- YouTube Data API v3 키워드 기반 영상 검색
- 영상 제목, 채널명, 게시일, 설명 등 메타데이터 수집
- `video_id` 기준 upsert로 중복 데이터 방지

### 2. Playwright 기반 댓글 크롤링
- YouTube 동적 렌더링 대응 (JavaScript 렌더링 후 댓글 파싱)
- 인기순 정렬 기반 상위 댓글 수집
- 무한 스크롤 자동 처리로 대량 댓글 수집
- 작성자, 댓글 내용, 좋아요 수, 작성일 수집
- 불필요한 리소스(이미지, 광고) 차단으로 크롤링 속도 최적화

### 3. 엔드투엔드 파이프라인 자동화
- `/pipeline/run` 엔드포인트 한 번 호출로 영상 수집 → 댓글 크롤링 → 감성 분석 → DB 저장 일괄 처리
- 아티스트명 파라미터로 다양한 아티스트 데이터 수집 가능

### 4. Claude AI 팬덤 감성 분석
- 수집된 댓글을 Claude API에 전달
- 긍정/부정/중립 비율, 주요 감정 키워드, 팬덤 반응 요약 자동 생성
- `dominant_emotion` DB 저장으로 댓글별 감성 추적

### 5. AI 팬덤 인사이트 리포트
- SM Entertainment 비즈니스 관점의 트렌드 분석
- 채널 분포, 글로벌 팬덤 시사점, 콘텐츠 전략 제언 자동 생성

### 6. 멜론 차트 수집
- BeautifulSoup 기반 실시간 멜론 차트 크롤링

---

## 🏗️ 아키텍처

```
[Client / Swagger UI]
        │
        ▼
[FastAPI Server - AWS EC2]
        │
   ┌────┴────────┐
   │             │
   ▼             ▼
[YouTube      [Playwright]
 Data API]     크롤러
   │             │
   └──────┬──────┘
          │
          ▼
    [PostgreSQL]
    ├── videos
    └── comments (+ sentiment)
          │
          ▼
    [Claude API]
    ├── 감성 분석
    └── 인사이트 리포트
```

---

## 📁 프로젝트 구조

```
smProject/
├── .github/
│   └── workflows/
│       └── deploy.yml       # GitHub Actions CI/CD
├── app/
│   ├── main.py              # FastAPI 앱 진입점
│   ├── youtube.py           # YouTube Data API 연동
│   ├── crawler.py           # Playwright 댓글 크롤러
│   ├── database.py          # SQLAlchemy 모델 및 DB 연결
│   ├── insight.py           # Claude API 감성 분석 및 인사이트
│   ├── melon.py             # 멜론 차트 크롤러
│   └── templates/
│       └── index.html       # 대시보드 UI
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔌 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/` | 대시보드 UI |
| `GET` | `/videos` | YouTube 영상 검색 |
| `POST` | `/videos/collect` | 영상 수집 후 DB 저장 |
| `GET` | `/videos/insight` | AI 팬덤 인사이트 리포트 생성 |
| `POST` | `/comments/collect` | 댓글 크롤링 후 DB 저장 |
| `GET` | `/comments/{video_id}` | 저장된 댓글 조회 |
| `GET` | `/comments/analyze/{video_id}` | 댓글 감성 분석 |
| `POST` | `/pipeline/run` | 영상 수집 + 댓글 크롤링 + 감성 분석 일괄 실행 |
| `GET` | `/chart/melon` | 멜론 차트 수집 |

Swagger UI: `http://서버IP:8000/docs`

---

## ⚙️ 실행 방법

```bash
git clone https://github.com/yunniez/smProject.git
cd smProject

py -3.12 -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium
```

`.env` 파일 생성:

```env
YOUTUBE_API_KEY=your_youtube_api_key
DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/postgres
ANTHROPIC_API_KEY=your_anthropic_api_key
```

```bash
uvicorn app.main:app --reload
```

### AWS EC2 자동 배포

```bash
git push origin master
```

---

## 💡 설계 포인트

- **동적 크롤링 대응**: YouTube 댓글은 JavaScript로 렌더링되므로 Playwright 적용
- **파이프라인 레이어 분리**: 수집 → 적재 → AI 분석의 명확한 책임 분리
- **중복 방지**: `video_id` unique 제약 및 댓글 텍스트 기준 중복 체크로 멱등성 보장
- **자동 배포**: GitHub Actions 기반 CI/CD로 push 즉시 EC2 반영
- **감성 DB 저장**: 분석 결과를 댓글 단위로 저장해 추후 통계 분석 가능하도록 설계

---

## 📊 수집 데이터 규모

| 대상 | 수집 항목 | 건수 |
|------|----------|------|
| YouTube 영상 | 제목, 채널, 게시일, 설명 | 회당 최대 50건 |
| 댓글 | 작성자, 내용, 좋아요, 작성일, 감성 | 영상당 최대 30건 (인기순) |
| 멜론 차트 | 순위, 곡명, 아티스트 | 실시간 TOP 100 |

---

## 🔮 개선 방향

- [ ] APScheduler 기반 주기적 자동 수집 스케줄러
- [ ] 국가별 언어 분포 분석으로 글로벌 팬덤 지도 시각화
- [ ] HTTPS 적용 (Let's Encrypt)
- [ ] asyncio 병렬 처리로 크롤링 속도 최적화
- [ ] 감성 분석 결과 시각화 대시보드