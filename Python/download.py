import sqlite3
import requests
import time
from urllib.parse import quote
from io import BytesIO
from pdfminer.high_level import extract_text
import subprocess
import sys, json
import functools, builtins
import re
import shutil
print = functools.partial(builtins.print, flush=True)

DB_PATH = r"Z:\Programming\AILibrary\library.db"
connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()
cursor.executescript("""
CREATE TABLE IF NOT EXISTS books (
  id            INTEGER PRIMARY KEY,
  gutenberg_id  INTEGER UNIQUE,
  ia_title_id   TEXT UNIQUE,
  gb_title_id   TEXT UNIQUE,
  author        TEXT,
  title         TEXT,
  category      TEXT,
  source_url    TEXT,
  content       TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_gutenberg
ON books(gutenberg_id)
WHERE gutenberg_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_archive
ON books(ia_title_id)
WHERE ia_title_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_google
ON books(gb_title_id)
WHERE gb_title_id IS NOT NULL;
""")
connection.commit()

headers = {"User-Agent": "MVP-Library/0.1 (+no email)"}

def download_gutenberg_text(gid: int):
    gutenberg_url_templates = [
    f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
    f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
    f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8",
]
    
    for url in gutenberg_url_templates:
        try:
            attempt = requests.get(url, timeout=20, headers=headers)
            if attempt.status_code == 200 and attempt.text.strip():
                return attempt.text, url
        except requests.RequestException:
            pass
    return None, None

def download_ia_text(ia_id: str):
    ia_url_template = [
    f"https://archive.org/download/{ia_id}/{ia_id}_djvu.txt",
    f"https://archive.org/download/{ia_id}/{ia_id}.txt",
]  
    for url in ia_url_template:
        try:
            attempt = requests.get(url, timeout=20, headers=headers)
            if attempt.status_code == 200 and attempt.text.strip():
                return attempt.text, url
        except requests.RequestException:
            pass
    return None, None

def extract_text_from_pdf(gb_pdf):
    try:
        text = extract_text(BytesIO(gb_pdf.content))
        if text and text.strip():
            return text
    except Exception as e:
        print(f"pdfminer failed: {e}")

    return None

def extract_ocr_from_pdf(gb_pdf):
    if shutil.which("tesseract") is None:
        print("Tesseract not found - skipping OCR")
        return None

    if shutil.which("gswin64c") is None:
        print("Ghostscript not found - skipping OCR")
        return None
    
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ocrmypdf",
             "--skip-text", "--force-ocr", "-l", "eng", "-", "-"],
            input=gb_pdf.content,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # capture stderr for debugging
            check=True,
        )
        text = extract_text(BytesIO(proc.stdout))
        if text and text.strip():
            return text
        else:
            print("OCR produced no text.")
    except Exception as e:
        print(f"OCR failed: {e}")
    return None

def convert_pdf_to_text(gb_pdf):

    text = extract_text_from_pdf(gb_pdf)
    if text is not None:
        return text
    else:
        return extract_ocr_from_pdf(gb_pdf)

def extract_gb_pdf(meta):
    book_meta = (meta.get("accessInfo") or {}).get("pdf") or {}
    if not book_meta.get("isAvailable"):
        print("No downloadable PDF for this volume.")
        return None, None

    download_url = book_meta.get("downloadLink")
    if not download_url:
        print("PDF isAvailable but no downloadLink.")
        return None, None
    
    try:
        gb_pdf = requests.get(download_url, timeout=60, headers=headers)
        print(f"PDF status {gb_pdf.status_code}, length {len(gb_pdf.content)} bytes")

        # Heuristic: tiny “PDFs” are usually error/captcha pages
        if gb_pdf.status_code != 200 or len(gb_pdf.content) < 50000:
            print("Likely CAPTCHA or error page from Google – skipping.")
            return None, None
    except requests.RequestException as e:
        print(f"PDF download error: {e}")
        return None, None
    
    return convert_pdf_to_text(gb_pdf), download_url

def download_gb_text(gb_id: str, api_key: str | None = None):
    url = f"https://www.googleapis.com/books/v1/volumes/{gb_id}"
    params = {"projection": "full"}
    if api_key:
        params["key"] = api_key

    try:
        r = requests.get(url, params=params, timeout=20, headers=headers) 
        print(f"GB API status {r.status_code} for {gb_id}")
        if r.status_code != 200:
            return None, None
    except requests.RequestException as e:
        print(f"GB API request error for {gb_id}: {e}")
        return None, None

    return extract_gb_pdf(r.json())

data = json.loads(sys.stdin.read())

gutenberg_books = []
ia_books = []
gb_books = []

def get_gutenberg_id(url: str) -> int | None:
    if "/ebooks/" not in url:
        return None
    return int(url.split("/ebooks/")[1].split("?")[0])

def get_ia_id(url: str) -> str | None:
    if "/details/" not in url:
        return None
    return url.split("/details/")[1].split("/")[0]

def get_google_id(url: str) -> str | None:
    patterns = [
        r"/books/edition/[^/]+/([^?\/]+)",   # current format
        r"\bid=([^&]+)",                     # ?id= format
        r"/books\?id=([^&]+)",               # old format
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None

for item in data:
    lower = {k.lower(): v for k, v in item.items()}

    url = lower.get("url")
    book_title = lower.get("title")
    author   = lower.get("author") or "Unknown"
    category = lower.get("category") or "Uncategorized"

    if not url or not book_title:
        continue

    if "gutenberg.org" in url:
        gid = get_gutenberg_id(url)
        if gid is not None:
            gutenberg_books.append((gid, book_title, author, category))
    elif "archive.org" in url:
        ia_id = get_ia_id(url)
        if ia_id is not None:
            ia_books.append((ia_id, book_title, author, category))
    elif "google.com/books" in url or "www.google.com/books" in url:
        gb_id = get_google_id(url)
        if gb_id is not None:
            gb_books.append((gb_id, book_title, author, category))
    else:
        pass

while gutenberg_books:
    title_id, user_title, author, category = gutenberg_books.pop(0)
    print(f"Downloading {title_id}...", end="")
    text, url = download_gutenberg_text(title_id)
    if text:
        cursor.execute("""
            INSERT INTO books (gutenberg_id, author, title, category, source_url, content)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(gutenberg_id) DO UPDATE SET
              title=excluded.title,
              category=excluded.category,
              source_url=excluded.source_url,
              content=excluded.content,
              author=excluded.author
        """, (title_id, author, user_title, category, url, text))
        connection.commit()
        print("Pass")
    else:
        print("Fail")
    time.sleep(1)

while ia_books:
    title_id, user_title, author, category = ia_books.pop(0)
    print(f"Downloading {title_id}...", end="")
    safe_id = quote(title_id, safe="")
    text, url = download_ia_text(safe_id)
    if text:
        cursor.execute("""
            INSERT INTO books (ia_title_id, author, title, category, source_url, content)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(ia_title_id) DO UPDATE SET
              title=excluded.title,
              category=excluded.category,
              source_url=excluded.source_url,
              content=excluded.content,
              author=excluded.author
        """, (title_id, author, user_title, category, url, text))
        connection.commit()
        print("Pass")
    else:
        print("Fail")
    time.sleep(1)

while gb_books:
    title_id, user_title, author, category = gb_books.pop(0)
    print(f"Downloading {title_id}...", end="")
    safe_id = quote(title_id, safe="")
    text, url = download_gb_text(safe_id)
    if text:
        cursor.execute("""
            INSERT INTO books (gb_title_id, author, title, category, source_url, content)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(gb_title_id) DO UPDATE SET
              title=excluded.title,
              category=excluded.category,
              source_url=excluded.source_url,
              content=excluded.content,
              author=excluded.author
        """, (title_id, author, user_title, category, url, text))
        connection.commit()
        print("Pass")
    else:
        print("Fail")
    time.sleep(1)
    
connection.close()
print("Done.")