from playwright.sync_api import sync_playwright
from app.database import SessionLocal, Comment

def crawl_youtube_comments(video_id: str, max_comments: int = 50) -> list[dict]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    comments = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.route("**/*.{png,jpg,jpeg,gif,svg,mp4,webp}", lambda route: route.abort())
        page.goto(url, wait_until="domcontentloaded")

        page.wait_for_timeout(3000)
        page.evaluate("window.scrollBy(0, 600)")
        page.wait_for_timeout(3000)

        sort_button = page.query_selector("tp-yt-paper-dropdown-menu")
        if sort_button:
            sort_button.click()
            page.wait_for_timeout(1000)
            # 인기순 선택 (첫 번째 옵션)
            options = page.query_selector_all("tp-yt-paper-item")
            if options:
                options[0].click()
                page.wait_for_timeout(2000)

        prev_count = 0
        while len(comments) < max_comments:
            items = page.query_selector_all("ytd-comment-thread-renderer")

            for item in items[prev_count:]:
                try:
                    author_el = item.query_selector("#author-text")
                    text_el = item.query_selector("#content-text")
                    likes_el = item.query_selector("#vote-count-middle")
                    date_el = item.query_selector(".published-time-text")

                    author = author_el.inner_text().strip() if author_el else ""
                    text = text_el.inner_text().strip() if text_el else ""
                    likes_str = likes_el.inner_text().strip() if likes_el else "0"
                    published_at = date_el.inner_text().strip() if date_el else ""

                    likes = parse_likes(likes_str)

                    if text:
                        comments.append({
                            "video_id": video_id,
                            "author": author,
                            "text": text,
                            "likes": likes,
                            "published_at": published_at
                        })
                except Exception:
                    continue

            if len(comments) >= max_comments:
                break

            prev_count = len(items)
            page.evaluate("window.scrollBy(0, 1000)")
            page.wait_for_timeout(2000)

            new_items = page.query_selector_all("ytd-comment-thread-renderer")
            if len(new_items) == prev_count:
                break

        browser.close()

    return comments[:max_comments]


def parse_likes(likes_str: str) -> int:
    likes_str = likes_str.replace(",", "").strip()
    if not likes_str:
        return 0
    try:
        if "K" in likes_str:
            return int(float(likes_str.replace("K", "")) * 1000)
        elif "M" in likes_str:
            return int(float(likes_str.replace("M", "")) * 1000000)
        return int(likes_str)
    except ValueError:
        return 0


def save_comments(video_id: str, comments: list[dict]) -> int:
    db = SessionLocal()
    try:
        existing_texts = {c.text for c in db.query(Comment).filter_by(video_id=video_id).all()}
        new_comments = [
            Comment(**c) for c in comments
            if c["text"] not in existing_texts
        ]
        db.add_all(new_comments)
        db.commit()
        return len(new_comments)
    finally:
        db.close()