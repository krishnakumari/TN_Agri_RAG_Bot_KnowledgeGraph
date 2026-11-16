import requests
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

resp = requests.get(url, headers=headers, verify=False, timeout=15)
print("Status Code:", resp.status_code)
print("HTML Length:", len(resp.text))

soup = BeautifulSoup(resp.text, "html.parser")
print("Page Title:", soup.title.string.strip() if soup.title else "No Title")

# Print all links that look related to schemes
scheme_links = [
    (a.get_text(strip=True), a["href"])
    for a in soup.find_all("a", href=True)
    if "scheme" in a["href"].lower() or len(a.get_text(strip=True)) > 25
]

print(f"\nFound {len(scheme_links)} candidate links:")
for text, href in scheme_links[:15]:
    print(f" - {text[:60]}  -->  {href}")

# Save raw HTML so we can inspect if needed
with open("debug_page.html", "w", encoding="utf-8") as f:
    f.write(resp.text)
print("\nSaved full HTML to debug_page.html")