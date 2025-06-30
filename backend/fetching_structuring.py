#!/usr/bin/env python3
import os
import re
import time
import json
import logging
import requests
import argparse
import xml.etree.ElementTree as ET
from lxml import etree
from requests.adapters import HTTPAdapter, Retry

# ─── Configuration ────────────────────────────────────────────────────────────

SITEMAP_ROOT = "https://www.retsinformation.dk/eli/sitemap.xml"
NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
LAW_URL_PATTERN = re.compile(
    r"^https://www\.retsinformation\.dk/eli/(lta|ltb|ltc)/\d{4}/\d+$"
)
USER_AGENT = "DanishLawScraper/0.1 (your.email@example.com)"

# ─── Session with Retries ─────────────────────────────────────────────────────

def make_session():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

# ─── 1) Sitemap Crawling ──────────────────────────────────────────────────────

def fetch_all_leaf_urls(session, sitemap_url):
    """Recursively fetch all <loc> URLs under the sitemap index."""
    resp = session.get(sitemap_url, timeout=10)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    sitemap_tags = root.findall("sm:sitemap/sm:loc", NS)
    if sitemap_tags:
        urls = []
        for tag in sitemap_tags:
            urls.extend(fetch_all_leaf_urls(session, tag.text))
            time.sleep(0.2)
        return urls
    return [loc.text for loc in root.findall("sm:url/sm:loc", NS)]

# ─── 2) Filter Law URLs ───────────────────────────────────────────────────────

def filter_law_urls(all_urls):
    """Keep only canonical Lovtidende A/B/C URLs."""
    return [u for u in all_urls if LAW_URL_PATTERN.match(u)]

# ─── 3) Fetch, Parse & Save ──────────────────────────────────────────────────

def fetch_and_save_law_json(session, eli_url, save_dir):
    """Download law XML, parse metadata & paragraphs (with chapters), save as JSON."""
    xml_url = eli_url.rstrip("/") + "/xml"
    resp = session.get(xml_url, timeout=10)
    resp.raise_for_status()

    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(resp.content, parser)

    def xp(expr, ctx=root):
        return ctx.xpath(expr)

    status = xp('string(//*[local-name()="Meta"]/*[local-name()="Status"])').strip()
    if status != "Valid":
        logging.info(f"Skipping non-valid law: {eli_url}")
        return False

    # 提取基本 meta 信息
    year = xp('string(//*[local-name()="Meta"]/*[local-name()="Year"])').strip()
    num  = xp('string(//*[local-name()="Meta"]/*[local-name()="Number"])').strip()

    law = {
        "id": f"{year}_{num}",
        "title": xp('string(//*[local-name()="Meta"]/*[local-name()="DocumentTitle"])').strip(),
        "year": year,
        "number": num,
        "status": status,
        "ministry": xp('string(//*[local-name()="Meta"]/*[local-name()="Ministry"])').strip(),
        "date_published": xp('string(//*[local-name()="Meta"]/*[local-name()="DiesEdicti"])').strip(),
        "signatures": xp('//*[local-name()="Meta"]/*[local-name()="Signature"]/text()'),
        "concerns": xp('//*[local-name()="Meta"]/*[local-name()="Ref_Text"]/text()'),
        "structured_text": []
    }

    # 查所有 Kapitel 节点
    kap_nodes = xp('//*[local-name()="Kapitel"]')
    structured = []

    if kap_nodes:
        # 有章节时，按章节遍历
        for kap in kap_nodes:
            chap_num = kap.xpath('string(*[local-name()="Explicatus"])').strip()
            chap_title = kap.xpath(
                'string(*[local-name()="Rubrica"]//*[local-name()="Char"])'
            ).strip() or ""
            chap_obj = {
                "chapter": chap_num,
                "title": chap_title,
                "paragraphs": []
            }

            for p in kap.xpath('.//*[local-name()="Paragraf"]'):
                para_num = p.xpath('string(*[local-name()="Explicatus"])').strip()
                para_obj = {"paragraph": para_num, "sections": []}

                for stk in p.xpath('.//*[local-name()="Stk"]'):
                    stk_num = stk.xpath(
                        'string(*[local-name()="Explicatus"])'
                    ).strip()
                    chars = stk.xpath('.//*[local-name()="Char"]/text()')
                    text = " ".join(c.strip() for c in chars if c.strip())
                    para_obj["sections"].append({
                        "section": stk_num,
                        "text": text
                    })

                chap_obj["paragraphs"].append(para_obj)

            structured.append(chap_obj)

    else:
        # 无章节时，回退到单一“默认章”
        default_chap = {"chapter": "", "title": "", "paragraphs": []}
        for p in xp('//*[local-name()="Paragraf"]'):
            para_num = p.xpath('string(*[local-name()="Explicatus"])').strip()
            para_obj = {"paragraph": para_num, "sections": []}

            for stk in p.xpath('.//*[local-name()="Stk"]'):
                stk_num = stk.xpath(
                    'string(*[local-name()="Explicatus"])'
                ).strip()
                chars = stk.xpath('.//*[local-name()="Char"]/text()')
                text = " ".join(c.strip() for c in chars if c.strip())
                para_obj["sections"].append({
                    "section": stk_num,
                    "text": text
                })

            default_chap["paragraphs"].append(para_obj)
        structured.append(default_chap)

    law["structured_text"] = structured

    # 保存 JSON
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{law['id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(law, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved: {path}")
    return True

# ─── Main / CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download and parse Danish laws from ELI into JSON."
    )
    parser.add_argument(
        "--out", "-o", default="laws_json",
        help="Directory to save JSON files"
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Only process the first N laws (for testing)"
    )
    parser.add_argument(
        "--sleep", "-s", type=float, default=0.1,
        help="Seconds to sleep between requests"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    session = make_session()
    logging.info("Fetching sitemap URLs…")
    all_urls = fetch_all_leaf_urls(session, SITEMAP_ROOT)
    logging.info(f"Found {len(all_urls):,} total URLs")

    law_urls = filter_law_urls(all_urls)
    logging.info(f"Filtered to {len(law_urls):,} law URLs")

    for idx, url in enumerate(law_urls[:10]):
        if args.limit and idx >= args.limit:
            break
        try:
            fetch_and_save_law_json(session, url, args.out)
        except Exception as e:
            logging.error(f"Error processing {url}: {e}")
        time.sleep(args.sleep)

if __name__ == "__main__":
    main()
