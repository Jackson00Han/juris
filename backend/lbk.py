#!/usr/bin/env python3
import time
import logging
import requests
import xml.etree.ElementTree as ET
import re
import sys
import os
import json

# ─── Configuration ─────────────────────────────────────────────────────────────
SITEMAP_URL        = "https://www.retsinformation.dk/eli/sitemap.xml"
NAMESPACE          = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
OUTPUT_DIR         = "lbk_bulk_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

THROTTLE           = 0.2    # seconds between HTTP calls
MAX_DOCS           = 5      # for test; set to None for full ingest
CONFIRM_THRESHOLD  = 1000   # prompt if > this many docs

# RIGHT: matches the canonical Retsinformation URLs (including LBK, decisions, etc.)
DOC_URL_PATTERN = re.compile(r"/eli/retsinfo/\d{4}/\d+$")


# ─── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lbk_harvester")

# ─── 1. Fetch sitemap URLs recursively ──────────────────────────────────────────
def fetch_sitemap_urls(url: str) -> list[str]:
    resp = requests.get(url)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    sitemap_tags = root.findall("sm:sitemap/sm:loc", NAMESPACE)
    if sitemap_tags:
        urls = []
        for tag in sitemap_tags:
            urls += fetch_sitemap_urls(tag.text)
            time.sleep(THROTTLE)
        return urls

    leaf = [loc.text for loc in root.findall("sm:url/sm:loc", NAMESPACE)]
    logger.info(f"  → Found {len(leaf)} URLs in leaf sitemap")
    return leaf

# ─── 2. Extract accn by calling the XML alias endpoint ─────────────────────────
def extract_accn_from_doc_url(doc_url: str) -> str | None:
    """
    Given a URL like https://.../eli/regel/lbkh/2015/1234,
    fetch {doc_url}/xml and parse the <accessionsnummer> element.
    """
    xml_url = doc_url.rstrip("/") + "/xml"
    resp = requests.get(xml_url)
    if resp.status_code != 200:
        return None
    root = ET.fromstring(resp.content)
    accn_el = root.find(".//{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}accessionsnummer")
    return accn_el.text if accn_el is not None else None

# ─── 3. Fetch meta + text and save if LBK & Gældende ───────────────────────────
def fetch_and_save(accn: str):
    base = "https://api.retsinformation.dk/api/v2/dokument"
    meta = requests.get(f"{base}/{accn}/metadata").json()
    # debug print first metadata for inspection
    if fetch_and_save.first:
        logger.info("Sample metadata:\n" + json.dumps(meta, indent=2, ensure_ascii=False))
        fetch_and_save.first = False

    if meta.get("type") != "LBK" or meta.get("status") != "Gældende":
        logger.info(f"Skipping {accn}: {meta.get('type')} / {meta.get('status')}")
        return False

    text = requests.get(f"{base}/{accn}/ekstraher/tekst").text
    with open(os.path.join(OUTPUT_DIR, f"{accn}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR, f"{accn}_text.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"Saved {accn}")
    return True

fetch_and_save.first = True

# ─── Main routine ───────────────────────────────────────────────────────────────
def main():
    logger.info("Fetching ELI sitemap URLs...")
    all_urls = fetch_sitemap_urls(SITEMAP_URL)
    logger.info(f"Total ELI URLs: {len(all_urls)}")

    # Filter to only consolidated-act doc URLs
    doc_urls = [u for u in all_urls if DOC_URL_PATTERN.search(u)]
    logger.info(f"Filtered document URLs: {len(doc_urls)}")

    # Extract accession numbers
    accns = []
    for url in doc_urls:
        accn = extract_accn_from_doc_url(url)
        if accn:
            accns.append(accn)
        time.sleep(THROTTLE)
    logger.info(f"Valid consolidated law accession numbers: {len(accns)}")

    # Cap for testing
    if MAX_DOCS:
        accns = accns[:MAX_DOCS]
    logger.info(f"Documents to attempt (capped): {len(accns)}")

    # Confirm if large
    if len(accns) > CONFIRM_THRESHOLD:
        ans = input(f"This will process {len(accns)} docs. Continue? (y/N) ")
        if ans.lower() != 'y':
            logger.info("Aborting per user input.")
            sys.exit(0)

    # Fetch & save
    saved = 0
    for accn in accns:
        try:
            if fetch_and_save(accn):
                saved += 1
        except Exception as e:
            logger.error(f"Error fetching {accn}: {e}")
        time.sleep(THROTTLE)

    logger.info(f"Finished: attempted {len(accns)}, saved {saved} laws.")

if __name__ == "__main__":
    main()
