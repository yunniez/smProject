# 🎯 FandomLens

> K-POP 아티스트 유튜브 데이터를 자동 수집·적재하고, AI가 팬덤 트렌드 인사이트를 생성하는 데이터 파이프라인 백엔드

---

## 📌 프로젝트 개요

FandomLens는 YouTube Data API를 통해 K-POP 아티스트 관련 영상 데이터를 **자동 수집 → PostgreSQL 적재 → Claude AI 분석**까지 이어지는 엔드투엔드 데이터 파이프라인

**자체 AI 서비스 백엔드 구축** 및 **글로벌 팬덤 데이터 활용** 전략에 직접 대응하는 구조로 설계

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| **Language** | Python 3.12 |
| **Backend Framework** | FastAPI |
| **Database** | PostgreSQL + SQLAlchemy ORM |
| **Data Collection** | YouTube Data API v3 |
| **AI Integration** | Anthropic Claude API |
| **API Documentation** | Swagger UI (FastAPI 내장) |
| **Version Control** | Git / GitHub |

---

## ✨ 주요 기능

### 1. 유튜브 영상 데이터 자동 수집
- YouTube Data API v3를 활용한 키워드 기반 영상 검색
- 영상 제목, 채널명, 게시일, 설명 등 메타데이터 수집
- 중복 데이터 방지 로직 적용 (video_id 기준 upsert)

### 2. PostgreSQL 자동 적재
- SQLAlchemy ORM 기반 데이터 모델링
- 서버 시작 시 테이블 자동 생성 (`create_all`)
- RESTful API 엔드포인트를 통한 수집 트리거

### 3. Claude AI 팬덤 인사이트 생성
- 수집된 영상 데이터를 Claude API에 전달
- 최근 활동 트렌드, 채널 분포, 글로벌 팬덤 시사점 자동 분석
- 마크다운 형식의 구조화된 리포트 반환

---

## 📁 프로젝트 구조

```
smProject/
├── app/
│   ├── main.py          # FastAPI 앱 진입점, 엔드포인트 정의
│   ├── youtube.py       # YouTube Data API 연동
│   ├── database.py      # SQLAlchemy 모델 및 DB 연결
│   └── insight.py       # Claude API 인사이트 생성
├── .env                 # 환경변수 (API 키, DB URL)
├── .gitignore
└── README.md
```

---

## 🔌 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/` | 헬스체크 |
| `GET` | `/videos` | YouTube 영상 검색 (DB 미저장) |
| `POST` | `/videos/collect` | 영상 데이터 수집 후 DB 적재 |
| `GET` | `/videos/insight` | DB 데이터 기반 AI 인사이트 생성 |

Swagger UI: `http://localhost:8000/docs`

---

## ⚙️ 실행 방법

### 1. 환경 설정
```bash
git clone https://github.com/yunniez/smProject.git
cd smProject
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. 환경변수 설정
`.env` 파일 생성:
```
YOUTUBE_API_KEY=your_youtube_api_key
DATABASE_URL=postgresql://postgres:password@localhost:5432/postgres
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 3. 서버 실행
```bash
uvicorn app.main:app --reload
```

---

## 💡 설계 포인트

- **데이터 파이프라인 구조**: 수집 → 적재 → AI 분석의 명확한 레이어 분리
- **중복 방지**: video_id unique 제약으로 멱등성 보장
- **확장성 고려**: 아티스트 쿼리 파라미터화로 다양한 아티스트 데이터 수집 가능
- **AI 연동 백엔드**: Claude API를 서비스 로직에 통합하는 구조 구현

---

## 🔮 개선 방향

- APScheduler를 활용한 주기적 자동 수집 스케줄러 추가
- 댓글 수집 및 감성 분석 기능 확장
- 국가별 언어 분포 분석으로 글로벌 팬덤 지도 시각화
- AWS EC2 + RDS 배포를 통한 운영 환경 구축