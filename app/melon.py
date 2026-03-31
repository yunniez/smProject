import requests
from bs4 import BeautifulSoup

def get_melon_chart(limit: int = 10):
    url = "https://www.melon.com/chart/index.htm"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    chart = []
    rows = soup.select("tr.lst50, tr.lst100")[:limit]

    for row in rows:
        rank = row.select_one(".rank").text.strip()
        title = row.select_one(".rank01 span a").text.strip()
        artist = row.select_one(".rank02 a").text.strip()

        chart.append({
            "rank": int(rank),
            "title": title,
            "artist": artist
        })

    return chart