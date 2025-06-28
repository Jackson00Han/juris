# part1_fetch_sitemap.py
import time
import xml.etree.ElementTree as ET
import requests

SITEMAP_URL = "https://www.retsinformation.dk/eli/sitemap.xml"
NAMESPACE = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
THROTTLE = 0.1
MAX_SHOW = 10  # 先看前 10 条

def fetch_sitemap_urls(url):
    resp = requests.get(url); resp.raise_for_status()
    root = ET.fromstring(resp.content)
    subs = root.findall("sm:sitemap/sm:loc", NAMESPACE)
    if subs:
        urls = []
        for tag in subs:
            urls += fetch_sitemap_urls(tag.text)
            time.sleep(THROTTLE)
        return urls
    return [loc.text for loc in root.findall("sm:url/sm:loc", NAMESPACE)]

if __name__ == "__main__":
    urls = fetch_sitemap_urls(SITEMAP_URL)
    print(f"共抓到 {len(urls)} 条 URL，先看前 {MAX_SHOW} 条：")
    for u in urls[:MAX_SHOW]:
        print(" ", u)
