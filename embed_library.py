#!/usr/bin/env python3
import os, sqlite3, time, argparse
import numpy as np
from typing import List, Tuple
import re

# Embeddings + FAISS
from sentence_transformers import SentenceTransformer
import faiss

import torch

DB_PATH_DEFAULT = "library.db"
INDEX_PATH_DEFAULT = "index.faiss"
SELECT_NEW_CHUNKS_SQL = """
SELECT chunks.id, chunks.text
FROM chunks
WHERE NOT EXISTS (
  SELECT 1 FROM embeddings_status
  WHERE embeddings_status.chunk_id = chunks.id
)
ORDER BY chunks.id
"""

def ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    # Chunks of each book's content — now reference books.id (internal PK)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id    INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        start_char INTEGER NOT NULL,
        end_char   INTEGER NOT NULL,
        text       TEXT NOT NULL,
        UNIQUE(book_id, chunk_index),
        FOREIGN KEY(book_id) REFERENCES books(id) ON DELETE CASCADE
    )
    """)
    # Tracks which chunks have been embedded & added to FAISS
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings_status (
        chunk_id    INTEGER PRIMARY KEY,
        embedded_at TEXT NOT NULL,
        FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
    )
    """)
    conn.commit()

def load_books(conn: sqlite3.Connection) -> List[Tuple[int, str]]:
    cur = conn.cursor()
    cur.execute("SELECT id, content FROM books")
    return cur.fetchall()

def book_already_chunked(conn: sqlite3.Connection, book_id: int) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM chunks WHERE book_id=? LIMIT 1", (book_id,))
    return cur.fetchone() is not None

def insert_chunks(conn: sqlite3.Connection, book_id: int, chunks: List[Tuple[int,int,str]]):
    cur = conn.cursor()
    rows = [(book_id, idex, start, end, text) for idex,(start, end, text) in enumerate(chunks)]
    cur.executemany(
        "INSERT OR IGNORE INTO chunks (book_id, chunk_index, start_char, end_char, text) VALUES (?,?,?,?,?)",
        rows
    )
    conn.commit()

def load_new_chunks(conn: sqlite3.Connection, limit: int = None) -> List[tuple[int, str]]:
    cur = conn.cursor()
    sql_query = SELECT_NEW_CHUNKS_SQL +(" LIMIT ?" if limit is not None else "")
    params = (int(limit),) if limit is not None else ()
    cur.execute(sql_query, params)
    return cur.fetchall()

def mark_embedded(conn: sqlite3.Connection, chunk_ids: List[int]):
    cur = conn.cursor()
    now = time.strftime("%Y-%m-%d $H:%M:%S")
    cur.executemany(
        "INSERT OR IGNORE INTO embeddings_status (chunk_id, embedded_at) VALUES (?,?)",
        [(cid, now) for cid in chunk_ids]
    )
    conn.commit()

def make_chunks(text: str, max_chars: int = 2000):
    paragraph_list = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks = []
    paragraph_buffer, paragraph_buffer_length = [], 0
    cursor = 0

    paragraph_separator = "\n\n"
    paragraph_separator_length = len(paragraph_separator)

    def is_buffer_within_limits(additional_length: int) -> bool:
        return paragraph_buffer_length + additional_length <= max_chars
    
    def append_content_from_buffer_to_chunk(window_size: int = 200) -> int:
        nonlocal cursor, paragraph_buffer, paragraph_buffer_length

        chunked_paragraph = paragraph_separator.join(paragraph_buffer)
        start = cursor
        end = start + len(chunked_paragraph)

        # Window Overlap
        window_start = max(start - window_size, 0)
        chunk_text = text[window_start:end]
        chunks.append((window_start, end, chunk_text))

        cursor = end + paragraph_separator_length

        paragraph_buffer, paragraph_buffer_length = [], 0
        return end
    
    def add_paragraph_to_buffer(paragraph: str, additional_length: int):
        nonlocal paragraph_buffer, paragraph_buffer_length
        paragraph_buffer.append(paragraph)
        paragraph_buffer_length += additional_length

    for paragraph in paragraph_list:
        additional_length = len(paragraph) + (paragraph_separator_length if paragraph_buffer else 0)
        
        if is_buffer_within_limits(additional_length):
            add_paragraph_to_buffer(paragraph, additional_length)
        else:
            append_content_from_buffer_to_chunk()
            paragraph_buffer, paragraph_buffer_length = [paragraph], len(paragraph)
    
    # If there is anything left in the buffer, append it to the chunk list
    if paragraph_buffer:
        append_content_from_buffer_to_chunk()

    return chunks

# FAISS Helpers

def load_index(path: str) -> faiss.Index | None:
    if os.path.exists(path):
        index = faiss.read_index(path)
        return index
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

def encoder (texts: List[str], model, batch_size: int) -> np.ndarray:
    start_time = time.time()
    embeddings = model.encode(texts,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=True)
    embeddings = embeddings.astype("float32")
    duration = time.time() - start_time
    rate = len(texts) / max(duration, 1e-6)
    print(f"Embedded {len(texts)} chunks in {duration:.2f}s ({rate:.1f} chunks/sec).")
    return embeddings

def add_ids_to_faiss_index(embeddings: np.ndarray, index_path: str, chunk_ids: List[int]) -> None:
    embedding_dimension = embeddings.shape[1]
    index = build_or_load_idmap(index_path, embedding_dimension)
    ids_as_numpy_array = np.array(chunk_ids, dtype=np.int64)
    index.add_with_ids(embeddings, ids_as_numpy_array)
    save_index(index, index_path)
    print(f"Index now has {index.ntotal} vectors. Saved to {index_path}")

def embed_new_chunks(conn: sqlite3.Connection,
                     index_path: str,
                     model_name: str = "jinaai/jina-embeddings-v3",
                     batch_size: int = 64,
                     limit_new: int = None) -> None:
    new_chunks = load_new_chunks(conn, limit=limit_new)
    if not new_chunks:
        print("No new chunks to embed. Done.")
        return
    
    chunk_ids = [chunk_id for chunk_id, _ in new_chunks]
    texts = [text for _, text in new_chunks]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    print(f"Loaded model {model_name} on device={device}")

    embeddings = encoder(texts, model, batch_size)

    add_ids_to_faiss_index(embeddings, index_path, chunk_ids)

    mark_embedded(conn, chunk_ids)
    print("Recorded embedding status in DB")

def prepare_chunks(conn: sqlite3.Connection, max_chars: int) -> None:
    books = load_books(conn)
    created_total = 0
    for book_id, content in books:
        if not content or content.strip() == "":
            continue
        if book_already_chunked(conn, book_id):
            continue
        chunks = make_chunks(content, max_chars=max_chars)
        insert_chunks(conn, book_id, chunks)
        print(f"Book {book_id}: created {len(chunks)} chunks.")
        created_total += len(chunks)
    if created_total == 0:
        print("No new chunks created (books already chunked).")
    else:
        print(f"Created {created_total} chunks total.")

def main():
    ap = argparse.ArgumentParser(description="Chunk + embed texts from library.db into a FAISS index.")
    ap.add_argument("--db", default=DB_PATH_DEFAULT, help="Path to SQLite DB (default: library.db)")
    ap.add_argument("--index", default=INDEX_PATH_DEFAULT, help="Path to FAISS index file")
    ap.add_argument("--max-chars", type=int, default=2000, help="Approx chars per chunk (default: 2000)")
    ap.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (default: 64)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of new chunks to embed this run")
    ap.add_argument("--model", default="jinaai/jina-embeddings-v3", help="Embedding model name")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"DB not found: {args.db}")

    conn = sqlite3.connect(args.db)
    # Enable FK support (off by default in SQLite)
    conn.execute("PRAGMA foreign_keys = ON;")
    ensure_schema(conn)

    print("== Chunk phase ==")
    prepare_chunks(conn, max_chars=args.max_chars)

    print("\n== Embed phase ==")
    embed_new_chunks(conn,
                     index_path=args.index,
                     model_name=args.model,
                     batch_size=args.batch_size,
                     limit_new=args.limit)

    conn.close()

if __name__ == "__main__":
    main()
