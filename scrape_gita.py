import argparse
import csv
import time
import sys
from typing import Optional

import requests
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; gita-scraper/1.0; +https://example.com)"
}


def get_soup(url: str, timeout: int = 15) -> Optional[BeautifulSoup]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException as e:
        print(f"Request error for {url}: {e}", file=sys.stderr)
        return None
    if r.status_code != 200:
        # non-200 -> treat as missing
        return None
    return BeautifulSoup(r.text, "html.parser")


def extract_by_class(soup: BeautifulSoup, class_name: str) -> Optional[str]:
    el = soup.find(class_=class_name)
    if not el:
        # sometimes multiple classes or slight variants: try contains
        el = soup.select_one(f'.{class_name}')
    if el:
        return el.get_text(" ", strip=True)
    return None


def extract_under_heading(soup: BeautifulSoup, heading_texts):
    """
    Find an element (h2/h3/h4) whose text matches one of heading_texts (case-insensitive,
    allowing whitespace). Return the combined text of the following sibling elements until
    the next heading of same level.
    """
    headings = soup.find_all(["h1", "h2", "h3", "h4"])
    lower_set = {t.lower() for t in heading_texts}
    for h in headings:
        ht = h.get_text(" ", strip=True).lower()
        # normalize common forms
        ht = " ".join(ht.split())
        if any(k in ht for k in lower_set):
            parts = []
            # gather next siblings until next heading tag
            for sib in h.next_siblings:
                if getattr(sib, "name", None) and sib.name in ("h1", "h2", "h3", "h4"):
                    break
                if getattr(sib, "get_text", None):
                    text = sib.get_text(" ", strip=True)
                    if text:
                        parts.append(text)
            return " ".join(parts).strip() or None
    return None


def scrape_page(chapter: int, verse: int):
    url = f"https://www.holy-bhagavad-gita.org/chapter/{chapter}/verse/{verse}/"
    soup = get_soup(url)
    if soup is None:
        return {"chapter": chapter, "verse": verse, "shloka": "", "translation": "", "commentary": "", "url": url, "status": "missing_or_error"}

    # Try direct class extractions first (as you requested)
    shloka = extract_by_class(soup, "bg-shlocks")
    translation = extract_by_class(soup, "bg-verse-translation")
    commentary = extract_by_class(soup, "bg-verse-commentary")

    # Fallbacks if any missing:
    if not shloka:
        # try generic nearby element: some pages show the verse lines as the top text under the title
        # attempt to pull the main verse block: look for an element with id or class containing 'verse' or first <p> under main article
        article = soup.find("article") or soup.find("div", {"id": "content"}) or soup.find("div", {"class": "content"})
        if article:
            # try to find an element that looks like the romanized/transliteration or devanagari
            cand = article.find(attrs={"class": lambda v: v and "shlock" in v}) or article.find("p")
            if cand:
                shloka = cand.get_text(" ", strip=True)

    if not translation:
        translation = extract_under_heading(soup, ["Translation", "Translation:", "BG", "Meaning", "Verse translation"])

    if not commentary:
        commentary = extract_under_heading(soup, ["Commentary", "Explanation", "Notes", "Commentary:"])

    return {
        "chapter": chapter,
        "verse": verse,
        "shloka": shloka or "",
        "translation": translation or "",
        "commentary": commentary or "",
        "url": url,
        "status": "ok"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-chapter", type=int, default=1)
    parser.add_argument("--end-chapter", type=int, default=18)
    parser.add_argument("--max-verse", type=int, default=100)
    parser.add_argument("--output", type=str, default="gita.csv")
    parser.add_argument("--delay", type=float, default=0.8, help="seconds between requests")
    args = parser.parse_args()

    rows = []
    total_iterations = (args.end_chapter - args.start_chapter + 1) * args.max_verse
    print(f"Will try chapters {args.start_chapter}..{args.end_chapter}, verses 1..{args.max_verse} (total tries: {total_iterations})")

    try:
        with open(args.output, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=["chapter", "verse", "shloka", "translation", "commentary", "url", "status"])
            writer.writeheader()

            for ch in range(args.start_chapter, args.end_chapter + 1):
                # iterate verses
                for v in tqdm(range(1, args.max_verse + 1)):
                    try:
                        item = scrape_page(ch, v)
                    except Exception as e:
                        print(f"Unhandled error scraping {ch}/{v}: {e}", file=sys.stderr)
                        item = {"chapter": ch, "verse": v, "shloka": "", "translation": "", "commentary": "", "url": f"https://www.holy-bhagavad-gita.org/chapter/{ch}/verse/{v}/", "status": "error"}
                    writer.writerow(item)
                    fout.flush()
                    time.sleep(args.delay)
    except KeyboardInterrupt:
        print("Interrupted by user. Partial CSV saved.")
    print("Done. CSV saved to", args.output)


if __name__ == "__main__":
    main()