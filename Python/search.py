from sentence_transformers import SentenceTransformer
import sqlite3
from typing import Dict, List
import torch
import faiss
import numpy as np
import json
import sys
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

# Default globals (will be overwritten by argparse)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
db_path = "library.db"
idx_path = "index.faiss"

# Top K defaults (will also be overwritten by argparse)
k = 5
min_score = -1.0
fetch_k = 200
book_ids = None
q = ""  # will be set from args
use_semantic = True  # True for semantic, False for literal

def row_factory(conn: sqlite3.Connection):
    conn.row_factory = sqlite3.Row
    return conn

def load_model():
    try:
        conn = row_factory(sqlite3.connect(db_path, check_same_thread=False))
        conn.execute("PRAGMA foreign_keys = ON;")

        index = faiss.read_index(idx_path)
        dim = index.d

        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

        test = model.encode(["ping"], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        if test.shape[1] != dim:
            raise RuntimeError(f"Model dim {test.shape[1]} != index dim {dim}. Use the same model as index.")
        return conn, model, index, dim
    except Exception as e:
        print(f"Search error:\n{e}", file=sys.stderr)
        return None, None, None, None

def load_db_only():
    """Load just the database connection for literal search (no model/index needed)"""
    try:
        conn = row_factory(sqlite3.connect(db_path, check_same_thread=False))
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    except Exception as e:
        print(f"Database error:\n{e}", file=sys.stderr)
        return None

def fetch_chunks_by_ids(conn: sqlite3.Connection, ids: List[int]) -> Dict[int, sqlite3.Row]:
    if not ids:
        return {}

    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    q = """
        SELECT
            c.id,
            c.book_id,
            c.text,
            b.title,
            b.source_url,
            b.author,
            b.category
        FROM chunks AS c
        JOIN books AS b ON c.book_id = b.id
        WHERE c.id IN ({})
    """.format(",".join(["?"] * len(ids)))

    cur.execute(q, ids)
    rows = cur.fetchall()
    return {row["id"]: row for row in rows}

def semantic_search(conn, model, index, dim, q, k=5, min_score=-1.0, fetch_k=200, book_ids=None):
    """Semantic search using FAISS vector similarity"""
    try:
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        if qv.shape[1] != dim:
            raise RuntimeError("Model/index dimension mismatch.")
        
        _fetch_k = max(fetch_k, k)
        D, I = index.search(qv, _fetch_k)
        ids = [int(i) for i in I[0] if i != -1]
        scores = D[0][:len(ids)].tolist()
        
        rows = fetch_chunks_by_ids(conn, ids)
        results = []
        
        for cid, score in zip(ids, scores):
            raw_row = rows.get(cid)
            if not raw_row:
                continue
            row = dict(raw_row)
            
            if book_ids and row["book_id"] not in book_ids:
                continue
            
            if score < min_score:
                continue
            
            results.append({
                "score": float(score),
                "book_id": int(row["book_id"]),
                "title": row["title"],
                "author": row["author"],
                "category": row["category"],
                "url": row["source_url"],
                "text": row["text"],
            })
            
            if len(results) >= k:
                break
        
        return results
    except Exception as e:
        msg = f"Semantic search error:\n{e}"
        print(msg, file=sys.stderr)
        return []

def literal_search(conn, q, k=5, book_ids=None):
    """Literal full-text search using FTS5"""
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Use FTS5 for fast full-text search
        sql = """
            SELECT 
                rank as score,
                b.author,
                b.title,
                b.category,
                b.source_url as url,
                c.text,
                c.book_id
            FROM chunks_fts fts
            JOIN chunks c ON c.id = fts.rowid
            JOIN books b ON b.id = c.book_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        
        cur.execute(sql, (q, k))
        rows = cur.fetchall()
        
        results = []
        for row in rows:
            row_dict = dict(row)
            
            # Filter by book_ids if specified
            if book_ids and row_dict["book_id"] not in book_ids:
                continue
            
            results.append({
                "score": float(row_dict["score"]),
                "book_id": int(row_dict["book_id"]),
                "title": row_dict["title"],
                "author": row_dict["author"],
                "category": row_dict["category"],
                "url": row_dict["url"],
                "text": row_dict["text"],
            })
        
        return results
        
    except Exception as e:
        msg = f"Literal search error:\n{e}"
        print(msg, file=sys.stderr)
        return []

# Main

ap = argparse.ArgumentParser(description="Search library.db using semantic or literal search.")
ap.add_argument("--q", help="Query string", required=True)
ap.add_argument("--k", type=int, default=5, help="Show top K results (default: 5)")
ap.add_argument("--min_score", type=float, default=-1.0, help="Min similarity score for semantic search (default: -1.0)")
ap.add_argument("--fetch_k", type=int, default=200, help="Number of candidates to fetch from FAISS (default: 200)")
ap.add_argument("--model_name", help="Model name to use for embedding and searching", required=True)
ap.add_argument("--db", default="library.db", help="Path to SQLite database")
ap.add_argument("--index", default="index.faiss", help="Path to FAISS index file")
ap.add_argument("--semantic", action="store_true", help="Use semantic search instead of literal search")
args = ap.parse_args()

# overwrite globals from args
q = args.q
k = args.k
min_score = args.min_score
fetch_k = args.fetch_k
model_name = args.model_name
db_path = args.db
idx_path = args.index
use_semantic = args.semantic

# Perform search
if use_semantic:
    # Semantic search - load model and index
    conn, model, index, dim = load_model()
    if conn is None or model is None or index is None or dim is None:
        sys.exit(1)
    results = semantic_search(conn, model, index, dim, q=q, k=k, min_score=min_score, fetch_k=fetch_k)
else:
    # Literal search - just load database
    conn = load_db_only()
    if conn is None:
        sys.exit(1)
    results = literal_search(conn, q=q, k=k)

# Output results as JSON
sys.stdout.reconfigure(encoding='utf-8')
print(json.dumps(results, ensure_ascii=False))
conn.close()