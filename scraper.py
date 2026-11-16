import json
import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://www.tn.gov.in/"
TARGET_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, verify=False, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def extract_clean_text(soup: BeautifulSoup) -> str:
    # Remove scripts, styles, navbars, headers, and footers to avoid noise in the Knowledge Graph
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Try common content containers on tn.gov.in first, fallback to body
    container = (
        soup.find("div", id="content")
        or soup.find("div", class_="content")
        or soup.find("table")
        or soup.body
    )
    if not container:
        return ""

    lines = [line.strip() for line in container.get_text("\n").splitlines() if line.strip()]
    return "\n".join(lines)


def scrape_schemes():
    print(f"Fetching schemes list from {TARGET_URL}...")
    soup = fetch_soup(TARGET_URL)

    # Target only links pointing to scheme_details.php
    scheme_links = soup.select('a[href*="scheme_details.php"]')
    print(f"Found {len(scheme_links)} scheme links.\n")

    schemes = []
    seen_urls = set()

    for idx, link in enumerate(scheme_links, start=1):
        title = link.get_text(" ", strip=True)
        detail_url = urljoin(BASE_URL, link["href"])

        if not title or detail_url in seen_urls:
            continue
        seen_urls.add(detail_url)

        print(f"[{idx}/{len(scheme_links)}] Scraping: {title}")
        details_text = ""
        try:
            detail_soup = fetch_soup(detail_url)
            details_text = extract_clean_text(detail_soup)
            time.sleep(0.3)  # Polite delay so we don't get rate-limited
        except Exception as e:
            print(f"   -> Warning: Could not fetch details ({e})")

        schemes.append({
            "id": idx,
            "title": title,
            "url": detail_url,
            "details": details_text,
        })

    with open("tn_agri_schemes.json", "w", encoding="utf-8") as f:
        json.dump(schemes, f, indent=2, ensure_ascii=False)

    print(f"\nSuccess! Saved {len(schemes)} schemes to tn_agri_schemes.json")


if __name__ == "__main__":
    scrape_schemes()