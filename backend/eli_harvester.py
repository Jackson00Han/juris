
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
MAX_DOCS    = None  # limit for test run (set an int for testing)
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

def process_and_store(accn, meta, text):
    sections = chunk_by_section(text)
    logger.info(f"{accn} → {len(sections)} sections (embedding skipped)")
