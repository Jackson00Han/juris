# eli_mvp_pipeline.py
# ------------------
# A one-time bulk ingest script for Danish laws from Retsinformation (ELI sitemap).
# It:
#  1) fetches all ELI URLs via sitemap index
#  2) downloads metadata + plain text for each law
#   3) chunks by section (§ / Artikel)
#  4) embeds and indexes via FAISS
#  5) (Placeholder) stores raw and chunk records in database



#!/usr/bin/env python3
import faulthandler; faulthandler.enable()
print("Starting script…")

import time
import logging
# … rest of your imports …
print("Imported standard libs OK")

from embedding_faiss import embed_texts, save_faiss_index
print("Imported embedding_faiss OK")




import time
import logging
import requests
import xml.etree.ElementTree as ET
from typing import List, Tuple
from embedding_faiss import embed_texts, save_faiss_index
import re
import sys

# ─── Configuration ─────────────────────────────────────────────────────────────
# NOTE: This script downloads only the extracted plain text and metadata
#       (not full PDF files), so storage requirements are relatively modest
#       — on the order of tens of megabytes rather than gigabytes.
# THROTTLE controls how quickly we make API calls. Setting it too low or zero
# may overwhelm the API or your network. Consider verifying the total number of
# documents and manually confirming before running large-scale ingests.
SITEMAP_URL = "https://www.retsinformation.dk/eli/sitemap.xml"
NAMESPACE   = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
THROTTLE    = 0.05  # seconds between HTTP calls
MAX_DOCS    = 10  # limit for test run (set an int for testing)
CONFIRM_THRESHOLD = 5000  # prompt for confirmation if total docs exceed this

# ─── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eli_mvp")

# ─── 1. Fetch all ELI URLs ──────────────────────────────────────────────────────

def fetch_sitemap_urls(url: str) -> List[str]:
    resp = requests.get(url)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    # detect if sitemap index (has <sitemap> children)
    sitemap_tags = root.findall("sm:sitemap/sm:loc", NAMESPACE)
    if sitemap_tags:
        urls: List[str] = []
        for tag in sitemap_tags:
            sub = tag.text
            urls += fetch_sitemap_urls(sub)
            time.sleep(THROTTLE)
        return urls

    # leaf sitemap: collect <url><loc>
    leaf_urls = [loc.text for loc in root.findall("sm:url/sm:loc", NAMESPACE)]
    logger.info(f"  → Found {len(leaf_urls)} URLs in leaf sitemap")
    return leaf_urls

# ─── 2. Extract accn from ELI URL ─────────────────────────────────────────────

def extract_accn(eli_url: str) -> str:
    # example: .../eli/accn/L000001234/xml
    return eli_url.split("/accn/")[1].split("/")[0]

# ─── 3. Chunk text by section ─────────────────────────────────────────────────

SECTION_PATTERN = re.compile(r"(§\s*\d+|Artikel\s+\d+)", re.IGNORECASE)

def chunk_by_section(text: str) -> List[Tuple[str, str]]:
    parts = SECTION_PATTERN.split(text)
    chunks: List[Tuple[str, str]] = []
    header = "(Intro)"
    body = ""

    for token in parts:
        if SECTION_PATTERN.match(token):
            if body.strip():
                chunks.append((header, body.strip()))
            header = token.strip()
            body = ""
        else:
            body += token
    if body.strip():
        chunks.append((header, body.strip()))
    return chunks

# ─── 4. Download + process each document ──────────────────────────────────────

def process_and_store(accn: str, metadata: dict, full_text: str):
    """
    Store raw metadata & text, chunk into sections, then embed & index.
    Replace database calls in this function with your real persistence logic.
    """
    # 4a) Store raw metadata & text in your DB (pseudocode)
    # db.insert_document(accn, metadata)

    # 4b) Chunk by section
    sections = chunk_by_section(full_text)
    texts = [sec_text for (_, sec_text) in sections]

    # 4c) Embed and index
    embeddings = embed_texts(texts)
    save_faiss_index(accn, texts, embeddings)

    # 4d) Store section records with header metadata (pseudocode)
    # for idx, (hdr, txt) in enumerate(sections):
    #     db.insert_section(accn, hdr, txt)

    logger.info(f"Indexed {accn}: {len(texts)} sections")

# ─── 5. Harvester main routine ─────────────────────────────────────────────────

def eli_harvest_all():
    all_urls = fetch_sitemap_urls(SITEMAP_URL)
    total = len(all_urls)
    logger.info(f"Total ELI endpoints discovered: {total}")

    # confirm if large
    if CONFIRM_THRESHOLD and total > CONFIRM_THRESHOLD:
        ans = input(f"This will process {total} documents. Continue? (y/N) ")
        if ans.lower() != 'y':
            logger.info("Aborting as per user input.")
            sys.exit(0)

    for i, url in enumerate(all_urls, start=1):
        accn = extract_accn(url)
        logger.info(f"[{i}/{total}] Harvesting dokumentId={accn}")

        # fetch metadata + text
        meta_url = f"https://api.retsinformation.dk/api/v2/dokument/{accn}/metadata"
        text_url = f"https://api.retsinformation.dk/api/v2/dokument/{accn}/ekstraher/tekst"
        meta = requests.get(meta_url).json()
        full_text = requests.get(text_url).text

        process_and_store(accn, meta, full_text)
        time.sleep(THROTTLE)

        if MAX_DOCS and i >= MAX_DOCS:
            logger.info(f"Reached MAX_DOCS={MAX_DOCS}, stopping early.")
            break

if __name__ == "__main__":
    eli_harvest_all()

# ----------------------------------------------------------------------------
# To run:
#   python eli_mvp_pipeline.py
# Then wire up your FastAPI `/run_agent` to search FAISS indices created above.
