#!/usr/bin/env python3
import os
import time
import re
import requests
import xml.etree.ElementTree as ET

# ─── 配置 ────────────────────────────────────────────────────────────────────────
SITEMAP_URL = "https://www.retsinformation.dk/eli/sitemap.xml"
NS          = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
THROTTLE    = 0.1       # 秒，防止太快
MAX_DOCS    = None      # 测试时可设为 5 或 10，正式拉全量设为 None
OUTPUT_DIR  = "lbk_texts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 只匹配 lovbekendtgørelse（合并法令）的 ELI URL
LBKH_REGEX  = re.compile(r"/eli/regel/lbkh/\d{4}/\d{2}/\d{2}/\d+$")
# XML 中 accessionsnummer 所在的命名空间
AKN_NS      = "{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}"

def fetch_all_sitemap_locs(url: str) -> list[str]:
    """递归抓取 sitemap，返回所有 <loc> URL 列表。"""
    resp = requests.get(url)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    # 如果是 sitemap-index，则继续递归
    sitemap_tags = root.findall("sm:sitemap/sm:loc", NS)
    if sitemap_tags:
        out = []
        for tag in sitemap_tags:
            out += fetch_all_sitemap_locs(tag.text)
            time.sleep(THROTTLE)
        return out

    # 否则是叶子页，直接取所有 <url><loc>
    return [u.text for u in root.findall("sm:url/sm:loc", NS)]

def extract_accn(eli_url: str) -> str|None:
    """给定 /eli/regel/lbkh/... URL，抓它的 /xml，解析 <accessionsnummer>。"""
    xml_url = eli_url.rstrip("/") + "/xml"
    r = requests.get(xml_url)
    if r.status_code != 200:
        return None
    doc = ET.fromstring(r.content)
    el = doc.find(f".//{AKN_NS}accessionsnummer")
    return el.text if el is not None else None

def main():
    print("1) 抓取所有 ELI sitemap URL…")
    all_locs = fetch_all_sitemap_locs(SITEMAP_URL)
    print(f"   → 共拿到 {len(all_locs)} 条 ELI URL")

    # 过滤出合并法令
    lbk_urls = [u for u in all_locs if LBKH_REGEX.search(u)]
    if MAX_DOCS:
        lbk_urls = lbk_urls[:MAX_DOCS]
    print(f"2) 匹配到 lovbekendtgørelse URL 共 {len(lbk_urls)} 条")

    # 依次下载纯文本
    for eli in lbk_urls:
        accn = extract_accn(eli)
        if not accn:
            print(f"   × 解析 accn 失败：{eli}")
            continue

        txt = requests.get(
            f"https://api.retsinformation.dk/api/v2/dokument/{accn}/ekstraher/tekst"
        ).text

        path = os.path.join(OUTPUT_DIR, f"{accn}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"✅ 保存纯文本：{path}")

        time.sleep(THROTTLE)

    print("🎉 完成纯文本下载。")

if __name__ == "__main__":
    main()
