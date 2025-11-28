#!/usr/bin/env python3
import os, sqlite3, time, argparse
import numpy as np
from typing import List, Tuple
import re
from dataclasses import dataclass
import sys

# Embeddings + FAISS
from sentence_transformers import SentenceTransformer
import faiss

import torch

DB_PATH_DEFAULT = "library.db"
INDEX_PATH_DEFAULT = "index.faiss"

SELECT_NEW_CHUNKS_SQL = """
-- Fetch only unembedded chunks; return exactly the 4 columns the loader expects
SELECT chunk.id, chunk.start_char, chunk.end_char, chunk.text
FROM chunks AS chunk
LEFT JOIN embeddings_status AS e ON e.chunk_id = chunk.id
WHERE e.chunk_id IS NULL
ORDER BY chunk.book_id, chunk.chunk_index
"""

@dataclass(frozen=True)
class Book:
    """Book class contains DB ID for it book_id, and the text string of it's entire contents"""
    id: int
    """DB index"""
    text: str
    """Contains the entire book's contents in a string"""

@dataclass(frozen=True)
class Chunk:
    """Chunk of original book text (≤ 2000 characters) with start/end offsets."""

    start: int
    """Inclusive start index (the first character of the chunk in the source text)."""

    end: int
    """Exclusive end index (one past the last character of the chunk)."""

    text: str
    """The chunk text (typically ≤ 2000 characters, may include backward overlap)."""


@dataclass(frozen=True)
class DbChunk:
    """A chunk row as stored in the database, with IDs and the text span."""

    id: int
    """Primary key of the chunk row (chunks.id)."""

    book_id: int
    """Foreign key of the source book (books.id)."""

    chunk_index: int
    """Order of the chunk within its book (0-based)."""

    span: Chunk
    """The chunk's content and offsets (see: span.start, span.end, span.text)."""

def log(msg: str):
    print(msg, flush=True)

def ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    # Chunks of each book's content – now reference books.id (internal PK)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id      INTEGER NOT NULL,
        chunk_index  INTEGER NOT NULL,   -- position within the book (0-based)
        start_char   INTEGER NOT NULL,   -- inclusive
        end_char     INTEGER NOT NULL,   -- exclusive
        text         TEXT NOT NULL,
        UNIQUE(book_id, chunk_index),
        FOREIGN KEY(book_id) REFERENCES books(id) ON DELETE CASCADE
    )
    """)

    # Tracks which chunks have been embedded & added to FAISS
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings_status (
        chunk_id     INTEGER PRIMARY KEY,  -- matches chunks.id
        embedded_at  TEXT NOT NULL,
        FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
    )
    """)
    conn.commit()

# Database IO
def load_books(conn: sqlite3.Connection) -> List[Book]:
    cur = conn.cursor()
    cur.execute("SELECT id, content FROM books")
    rows =  cur.fetchall()
    return [Book(id=row[0], text=row[1]) for row in rows]

def book_already_chunked(conn: sqlite3.Connection, book_id: int) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM chunks WHERE book_id=? LIMIT 1", (book_id,))
    return cur.fetchone() is not None

def insert_chunks(conn: sqlite3.Connection, book_id: int, chunks: list[Chunk]) -> None:
    rows = [(book_id, idx, chunk.start, chunk.end, chunk.text) for idx, chunk in enumerate(chunks)]
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO chunks (book_id, chunk_index, start_char, end_char, text) " 
            "VALUES (?,?,?,?,?)",
            rows
        )

def load_new_chunks(conn: sqlite3.Connection, limit: int | None = None) -> list[tuple[int, Chunk]]:
    """
    Return [(chunk_id, Chunk), ...] for all chunks that are not yet embedded.
    Matches the 4-column SELECT above.
    """
    cur = conn.cursor()
    sql = SELECT_NEW_CHUNKS_SQL + (" LIMIT ?" if limit is not None else "")
    params = (int(limit),) if limit is not None else ()
    cur.execute(sql, params)

    rows = cur.fetchall()  # [(chunk_id, start_char, end_char, text), ...]
    result: list[tuple[int, Chunk]] = []
    for (cid, start_char, end_char, text_val) in rows:
        result.append((
            cid,
            Chunk(start=start_char, end=end_char, text=text_val),
        ))
    return result


def mark_embedded(conn: sqlite3.Connection, chunk_ids: List[int]):
    cur = conn.cursor()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cur.executemany(
        "INSERT OR IGNORE INTO embeddings_status (chunk_id, embedded_at) VALUES (?,?)",
        [(cid, now) for cid in chunk_ids]
    )
    conn.commit()

# Chunking
def make_chunks(text: str, max_chars: int = 2000, window_size: int = 200) -> list[Chunk]:
    """
    Pack consecutive paragraphs together into chunks until reaching ~max_chars.
    
    FIXED VERSION: Properly tracks position through entire text without skipping content.

    Parameters
    ----------
    text : str
        Full book as one string
    max_chars : int, optional
        Max characters per chunk, default = 2000
    window_size : int, optional
        Size of overlap between chunks (buffers), default = 200
    
    Returns
    -------
    chunks : list
        List containing all the chunks
    """
    
    # Split into paragraphs
    paragraph_list = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    
    if not paragraph_list:
        return []
    
    chunks: list[Chunk] = []
    paragraph_buffer = []
    text_position = 0  # Tracks where we are in the ORIGINAL text
    paragraph_separator = "\n\n"
    
    # Find the actual position of each paragraph in the original text
    paragraph_positions = []
    search_start = 0
    for para in paragraph_list:
        # Find where this paragraph actually appears in the original text
        para_start = text.find(para, search_start)
        if para_start == -1:
            # This shouldn't happen, but handle it gracefully
            para_start = search_start
        paragraph_positions.append(para_start)
        search_start = para_start + len(para)
    
    i = 0
    while i < len(paragraph_list):
        # Start a new chunk
        paragraph_buffer = []
        buffer_start_pos = paragraph_positions[i]
        buffer_length = 0
        
        # Pack paragraphs into this chunk
        while i < len(paragraph_list):
            para = paragraph_list[i]
            para_len = len(para)
            
            # Check if adding this paragraph would exceed limit
            if paragraph_buffer and buffer_length + len(paragraph_separator) + para_len > max_chars:
                break
            
            # Add paragraph to buffer
            paragraph_buffer.append(para)
            buffer_length += para_len
            if len(paragraph_buffer) > 1:
                buffer_length += len(paragraph_separator)
            
            i += 1
        
        # Now create the chunk
        if paragraph_buffer:
            # Calculate the actual end position in original text
            last_para_idx = i - 1
            last_para = paragraph_list[last_para_idx]
            last_para_pos = paragraph_positions[last_para_idx]
            chunk_end = last_para_pos + len(last_para)
            
            # Add overlap window (look backward)
            window_start = max(buffer_start_pos - window_size, 0)
            
            # Extract the actual text from the original
            chunk_text = text[window_start:chunk_end]
            
            chunks.append(
                Chunk(
                    start=window_start,
                    end=chunk_end,
                    text=chunk_text,
                )
            )
    
    return chunks

# FAISS

def load_index(path: str) -> faiss.Index | None:
    
    if not os.path.exists(path):
        log(f"No index at {path}; will create a new one.")
        return None
    
    if os.path.getsize(path) == 0:
        log(f"Index file at {path} is empty; will create a new one.")
        return None
    
    try:
        log(f"Loading FAISS index from {path}...")
        index = faiss.read_index(path)
        log(f"Loaded FAISS index with {index.ntotal} vectors.")
        return index
    except Exception as e:
        log(f"Error loading FAISS index from {path}: {e}")
        return None

def save_index(index, path: str) -> None:
    faiss.write_index(index, path)

def build_or_load_idmap(index_path: str, embedding_dimension: int) -> faiss.IndexIDMap:
    index = load_index(index_path)
    if index is None:
        base = faiss.IndexFlatIP(embedding_dimension)
        idmap = faiss.IndexIDMap2(base)
        return idmap
    else:
        if index.d != embedding_dimension:
            raise RuntimeError(f"FAISS index dimension {index.d} != embedding dimension {embedding_dimension}. Delete {index_path} or rebuild with matching model")
        return index
    
# Embedding Pipeline

def encoder (texts: list[str], model, batch_size: int) -> np.ndarray:
    start_time = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True)
    embeddings = embeddings.astype("float32")
    duration = time.time() - start_time
    rate = len(texts) / max(duration, 1e-6)
    log(f"Embedded {len(texts)} chunks in {duration:.2f}s ({rate:.1f} chunks/sec).")
    return embeddings

def add_ids_to_faiss_index(embeddings: np.ndarray, index_path: str, chunk_ids: List[int]) -> None:
    embedding_dimension = embeddings.shape[1]
    index = build_or_load_idmap(index_path, embedding_dimension)
    ids_as_numpy_array = np.array(chunk_ids, dtype=np.int64)
    index.add_with_ids(embeddings, ids_as_numpy_array)
    save_index(index, index_path)
    log(f"Index now has {index.ntotal} vectors. Saved to {index_path}")

def embed_new_chunks(conn: sqlite3.Connection,
                     index_path: str,
                     model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                     batch_size: int = 64,
                     limit_new: int = None) -> None:
    new_chunks = load_new_chunks(conn, limit=limit_new)
    if not new_chunks:
        log("No new chunks to embed. Done.")
        return
    
    chunk_ids = [chunk_id for chunk_id, _ in new_chunks]
    texts = [chunk.text for _, chunk in new_chunks]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    log(f"Loaded model {model_name} on device={device}")

    embeddings = encoder(texts, model, batch_size)

    add_ids_to_faiss_index(embeddings, index_path, chunk_ids)

    mark_embedded(conn, chunk_ids)
    log("Recorded embedding status in DB")

# Orchestration
def prepare_chunks(conn: sqlite3.Connection, max_chars: int) -> None:
    books = load_books(conn)
    created_total = 0
    for book in books:
        if not book.text or book.text.strip() == "":
            continue
        if book_already_chunked(conn, book.id):
            continue
        chunks = make_chunks(book.text, max_chars=max_chars)
        insert_chunks(conn, book.id, chunks)
        log(f"Book {book.id}: created {len(chunks)} chunks.")
        created_total += len(chunks)
    if created_total == 0:
        log("No new chunks created (books already chunked).")
    else:
        log(f"Created {created_total} chunks total.")

def main():
    ap = argparse.ArgumentParser(description="Chunk + embed texts from library.db into a FAISS index.")
    ap.add_argument("--db", default=DB_PATH_DEFAULT, help="Path to SQLite DB (default: library.db)")
    ap.add_argument("--index", default=INDEX_PATH_DEFAULT, help="Path to FAISS index file")
    ap.add_argument("--max-chars", type=int, default=2000, help="Approx chars per chunk (default: 2000)")
    ap.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (default: 64)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of new chunks to embed this run")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"DB not found: {args.db}")

    conn = sqlite3.connect(args.db)
    # Enable FK support (off by default in SQLite)
    conn.execute("PRAGMA foreign_keys = ON;")
    ensure_schema(conn)

    log("== Chunk phase ==")
    prepare_chunks(conn, max_chars=args.max_chars)

    log("\n== Embed phase ==")
    embed_new_chunks(conn,
                     index_path=args.index,
                     model_name=args.model,
                     batch_size=args.batch_size,
                     limit_new=args.limit)

    conn.close()

if __name__ == "__main__":
    main()