import os
import re
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from typing import List
from urllib.parse import urljoin

import requests
from tqdm import tqdm

BASE_URL = "https://storage.lczero.org/files/match_pgns/3/"
OUTPUT_PATH = "data/lc0-3.pgn"
MAX_WORKERS = 16
CHUNK_SIZE = 1024 * 1024  # 1 MiB
LIMIT = 0  # default maximum number of files to download; set <=0 for no limit


class PGNLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href", "")
        if href and href.lower().endswith(".pgn"):
            self.links.append(href)


def fetch_index(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_pgn_links(html: str) -> List[str]:
    parser = PGNLinkParser()
    parser.feed(html)
    # Normalize and ensure unique
    links = []
    for href in parser.links:
        # Handle relative and absolute
        full = urljoin(BASE_URL, href)
        if full not in links:
            links.append(full)
    return links


def stream_file(url: str) -> bytes:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    return resp.content


def download_and_collect(url: str):
    try:
        content = stream_file(url)
        return url, content, None
    except Exception as e:
        return url, None, e


def main():
    parser = argparse.ArgumentParser(description="Download LCZero PGN files and combine them into one file.")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum number of files to download (default: 200). Use 0 or negative for no limit.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Fetching index from {BASE_URL}...")
    html = fetch_index(BASE_URL)
    links = parse_pgn_links(html)

    if not links:
        print("No .pgn links found on the page.", file=sys.stderr)
        sys.exit(1)

    total_found = len(links)
    # Apply limit if provided (>0)
    if args.limit > 0 and total_found > args.limit:
        links = links[: args.limit]
    to_download = len(links)

    print(f"Found {total_found} .pgn files. Downloading {to_download} with {MAX_WORKERS} threads...")

    # Open target file once and append as downloads complete
    with open(OUTPUT_PATH, "wb") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(download_and_collect, url): url for url in links}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading PGNs"):
                url = futures[future]
                _url, content, error = future.result()
                if error is not None:
                    tqdm.write(f"Failed: {url} ({error})")
                    continue
                # Ensure file separation by a newline
                if not content.endswith(b"\n"):
                    content += b"\n"
                out_f.write(content)

    print(f"All done. Combined PGNs written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
