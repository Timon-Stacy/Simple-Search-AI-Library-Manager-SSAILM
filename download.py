import sqlite3
import requests
import time
from urllib.parse import quote
from io import BytesIO
from pdfminer.high_level import extract_text
import subprocess

connection = sqlite3.connect("library.db")
cursor = connection.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS books (
  id            INTEGER PRIMARY KEY,   -- internal row id
  gutenberg_id  INTEGER UNIQUE,        -- nullable; unique when present
  ia_title_id   TEXT UNIQUE,           -- nullable; unique when present
  gb_title_id   TEXT UNIQUE,
  title         TEXT,
  source_url    TEXT,
  content       TEXT
)
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
    except Exception:
        pass

    return None

def extract_ocr_from_pdf(gb_pdf):
    try:
        proc = subprocess.run(
            ["ocrmypdf", "--skip-text", "--force-ocr", "-l", "eng", "-", "-"],
            input=gb_pdf.content,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        text = extract_text(BytesIO(proc.stdout))
        if text and text.strip():
            return text
    except Exception:
        pass
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
        return None, None
    download_url = book_meta.get("downloadLink")
    if not download_url:
        return None, None
    
    try:
        gb_pdf = requests.get(download_url, timeout=60, headers=headers)
        if gb_pdf.status_code != 200 or not gb_pdf.content:
            return None, None
    except requests.RequestException:
        return None, None
    
    return convert_pdf_to_text(gb_pdf), download_url

def download_gb_text(gb_id: str, api_key: str | None = None):
    url = f"https://www.googleapis.com/books/v1/volumes/{gb_id}"
    params = {"projection": "full"}
    if api_key:
        params["key"] = api_key

    try:
        r = requests.get(url, params=params, timeout=20, headers=headers) 
        if r.status_code != 200:
            return None, None
    except requests.RequestException:
        return None, None

    # existing extract + convert
    return extract_gb_pdf(r.json())

gutenberg_books = []
ia_books = []
gb_books = []

while True:
    which_archive = input("Gutenberg (1), Internet Archive (2), Google Books (3) ")
    if not which_archive:
        break
    
    book_id = input("Book ID ")
    if not book_id:
        break

    book_title = input("Book Title ")
    if not book_title:
        break

    if which_archive == "1":
        gutenberg_books.append((int(book_id.strip()), book_title))
    elif which_archive == "2":
        ia_books.append((book_id, book_title))
    elif which_archive == "3":
        gb_books.append((book_id, book_title))
    else:
        pass

while gutenberg_books:
    title_id, user_title = gutenberg_books.pop(0)
    print(f"Downloading {title_id}…", end="")
    text, url = download_gutenberg_text(title_id)
    if text:

        cursor.execute("""
            INSERT INTO books (gutenberg_id, title, source_url, content)
            VALUES (?,?,?,?)
            ON CONFLICT(gutenberg_id) DO UPDATE SET
            title=excluded.title,
            source_url=excluded.source_url,
            content=excluded.content
        """, (title_id, user_title, url, text))

        connection.commit()
        print("Pass")
    else:
        print("Fail")
    time.sleep(1)

while ia_books:
    title_id, user_title = ia_books.pop(0)
    print(f"Downloading {title_id}…", end="")
    safe_id = quote(title_id, safe="")
    text, url = download_ia_text(safe_id)
    if text:
        
        cursor.execute("""
            INSERT INTO books (ia_title_id, title, source_url, content)
            VALUES (?,?,?,?)
            ON CONFLICT(ia_title_id) DO UPDATE SET
            title=excluded.title,
            source_url=excluded.source_url,
            content=excluded.content
        """, (title_id, user_title, url, text))

        connection.commit()
        print("Pass")
    else:
        print("Fail")
    time.sleep(1) 

while gb_books:
    title_id, user_title = gb_books.pop(0)
    print(f"Downloading {title_id}…", end="")
    safe_id = quote(title_id, safe="")
    text, url = download_gb_text(safe_id)
    if text:
        
        cursor.execute("""
            INSERT INTO books (gb_title_id, title, source_url, content)
            VALUES (?,?,?,?)
            ON CONFLICT(gb_title_id) DO UPDATE SET
            title=excluded.title,
            source_url=excluded.source_url,
            content=excluded.content
        """, (title_id, user_title, url, text))

        connection.commit()
        print("Pass")
    else:
        print("Fail")
    time.sleep(1) 
    


connection.close()
print("Done.")