#!/usr/bin/env python3
import os
import sqlite3
import threading
import webbrowser
import textwrap
from typing import Dict, List

import numpy as np
# NOTE: heavy libs moved to lazy import inside _load_all()
# import faiss
# import torch
# from sentence_transformers import SentenceTransformer

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Library Semantic Search (Tk)"

def summarize(text: str, width: int = 100, max_lines: int = 6) -> str:
    wrapped = textwrap.wrap((text or "").replace("\n", " "), width=width)
    s = "\n".join(wrapped[:max_lines])
    return s + (" …" if len(wrapped) > max_lines else "")

def row_factory(conn: sqlite3.Connection):
    conn.row_factory = sqlite3.Row
    return conn

def list_books(conn: sqlite3.Connection):
    q = """
    SELECT
        b.id AS id,
        b.gutenberg_id AS gutenberg_id,
        b.ia_title_id AS ia_title_id,
        b.title AS title,
        b.source_url AS source_url,
        COUNT(c.id) AS chunks,
        SUM(CASE WHEN e.chunk_id IS NOT NULL THEN 1 ELSE 0 END) AS embedded
    FROM books b
    LEFT JOIN chunks c ON c.book_id = b.id
    LEFT JOIN embeddings_status e ON e.chunk_id = c.id
    GROUP BY b.id, b.gutenberg_id, b.ia_title_id, b.title, b.source_url
    ORDER BY b.title
    """
    cur = conn.cursor()
    cur.execute(q)
    return [dict(r) for r in cur.fetchall()]

def fetch_chunks_by_ids(conn: sqlite3.Connection, ids: List[int]) -> Dict[int, sqlite3.Row]:
    if not ids:
        return {}
    cur = conn.cursor()
    q = f"""
    SELECT
        c.id,
        c.book_id,
        c.chunk_index,
        c.text,
        b.title,
        b.source_url
    FROM chunks c
    LEFT JOIN books b ON b.id = c.book_id
    WHERE c.id IN ({",".join(["?"]*len(ids))})
    """
    cur.execute(q, ids)
    rows = cur.fetchall()
    return {row["id"]: row for row in rows}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x750")

        # State
        self.conn: sqlite3.Connection | None = None
        self.index = None
        self.index_dim: int | None = None
        self.model = None
        self.device = "cpu"  # set after lazy torch import
        self.last_rows: Dict[int, sqlite3.Row] = {}
        self.db_lock = threading.Lock()

        # UI
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(5, weight=1)

        # DB + Index + Model
        ttk.Label(top, text="DB:").grid(row=0, column=0, sticky="w")
        self.db_var = tk.StringVar(value="library.db")
        db_entry = ttk.Entry(top, textvariable=self.db_var, width=40)
        db_entry.grid(row=0, column=1, sticky="w")
        ttk.Button(top, text="Browse", command=self._pick_db).grid(row=0, column=2, padx=(4,12))

        ttk.Label(top, text="Index:").grid(row=0, column=3, sticky="w")
        self.idx_var = tk.StringVar(value="index.faiss")
        idx_entry = ttk.Entry(top, textvariable=self.idx_var, width=40)
        idx_entry.grid(row=0, column=4, sticky="w")
        ttk.Button(top, text="Browse", command=self._pick_index).grid(row=0, column=5, sticky="w", padx=(4,12))

        ttk.Label(top, text="Model:").grid(row=1, column=0, sticky="w")
        self.model_var = tk.StringVar(value="jinaai/jina-embeddings-v3")
        model_entry = ttk.Entry(top, textvariable=self.model_var, width=40)
        model_entry.grid(row=1, column=1, sticky="w")
        ttk.Button(top, text="Load", command=self._load_all).grid(row=1, column=2, padx=(4,12))
        self.status = ttk.Label(top, text="Ready.", foreground="#555")
        self.status.grid(row=1, column=3, columnspan=3, sticky="w")

        # Split main area: left = books, right = search/results
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        # Left: Books panel
        left = ttk.Frame(main, padding=(0,0,8,0))
        left.columnconfigure(0, weight=1); left.rowconfigure(1, weight=1)
        ttk.Label(left, text="Books (select to filter)").grid(row=0, column=0, sticky="w")
        self.books_tv = ttk.Treeview(
            left,
            columns=("book_id","gutenberg_id","ia_id","title","chunks","embedded"),
            show="headings",
            selectmode="extended",
            height=12
        )
        self.books_tv.heading("book_id", text="BookID")
        self.books_tv.heading("gutenberg_id", text="Gutenberg ID")
        self.books_tv.heading("ia_id", text="IA ID")
        self.books_tv.heading("title", text="Title")
        self.books_tv.heading("chunks", text="Chunks")
        self.books_tv.heading("embedded", text="Embedded")

        self.books_tv.column("book_id", width=80, anchor="e")
        self.books_tv.column("gutenberg_id", width=100, anchor="e")
        self.books_tv.column("ia_id", width=160, anchor="w")
        self.books_tv.column("title", width=320)
        self.books_tv.column("chunks", width=80, anchor="e")
        self.books_tv.column("embedded", width=80, anchor="e")

        self.books_tv.grid(row=1, column=0, sticky="nsew")
        books_scroll = ttk.Scrollbar(left, command=self.books_tv.yview)
        self.books_tv.configure(yscroll=books_scroll.set)
        books_scroll.grid(row=1, column=1, sticky="ns")
        ttk.Button(left, text="Refresh", command=self._refresh_books).grid(row=2, column=0, sticky="w", pady=(6,0))

        # Right: Search panel
        right = ttk.Frame(main)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)

        ttk.Label(right, text="Query").grid(row=0, column=0, sticky="w")
        self.q_text = tk.Text(right, height=4, wrap="word")
        self.q_text.grid(row=1, column=0, sticky="ew")

        params = ttk.Frame(right)
        params.grid(row=2, column=0, sticky="ew", pady=6)
        for i in range(6): params.columnconfigure(i, weight=1)

        ttk.Label(params, text="Top K").grid(row=0, column=0, sticky="w")
        self.k_var = tk.IntVar(value=5)
        ttk.Spinbox(params, from_=1, to=100, textvariable=self.k_var, width=6).grid(row=1, column=0, sticky="w")

        ttk.Label(params, text="Min cosine").grid(row=0, column=1, sticky="w")
        self.min_var = tk.StringVar(value="-1.0")
        ttk.Entry(params, textvariable=self.min_var, width=8).grid(row=1, column=1, sticky="w")

        ttk.Label(params, text="Fetch K").grid(row=0, column=2, sticky="w")
        self.fetchk_var = tk.IntVar(value=200)
        ttk.Spinbox(params, from_=10, to=5000, increment=10, textvariable=self.fetchk_var, width=8).grid(row=1, column=2, sticky="w")

        ttk.Button(params, text="Search", command=self._do_search).grid(row=1, column=5, sticky="e")

        # Results table
        ttk.Label(right, text="Results (double-click a row to open source)").grid(row=3, column=0, sticky="w")
        self.res_tv = ttk.Treeview(
            right,
            columns=("rank","cos","book_id","chunk","title","url"),
            show="headings"
        )
        for col, text, w, anchor in [
            ("rank","#",40,"e"),
            ("cos","cos",80,"e"),
            ("book_id","BookID",80,"e"),
            ("chunk","Chunk",80,"e"),
            ("title","Title",420,"w"),
            ("url","URL",260,"w"),
        ]:
            self.res_tv.heading(col, text=text)
            self.res_tv.column(col, width=w, anchor=anchor)
        self.res_tv.grid(row=4, column=0, sticky="nsew")
        res_scroll = ttk.Scrollbar(right, command=self.res_tv.yview)
        self.res_tv.configure(yscroll=res_scroll.set)
        res_scroll.grid(row=4, column=1, sticky="ns")
        self.res_tv.bind("<Double-1>", self._open_selected_url)
        self.res_tv.bind("<<TreeviewSelect>>", self._show_selected_snippet)

        # Snippet area
        ttk.Label(right, text="Snippet").grid(row=5, column=0, sticky="w")
        self.snip = tk.Text(right, height=8, wrap="word")
        self.snip.grid(row=6, column=0, sticky="nsew")
        snip_scroll = ttk.Scrollbar(right, command=self.snip.yview)
        self.snip.configure(yscroll=snip_scroll.set)
        snip_scroll.grid(row=6, column=1, sticky="ns")

        main.add(left, weight=1)
        main.add(right, weight=2)

    # ----------- actions -----------
    def _pick_db(self):
        p = filedialog.askopenfilename(title="Select SQLite DB", filetypes=[("SQLite DB","*.db"),("All","*.*")])
        if p: self.db_var.set(p)

    def _pick_index(self):
        p = filedialog.askopenfilename(title="Select FAISS index", filetypes=[("FAISS index","*.faiss"),("All","*.*")])
        if p: self.idx_var.set(p)

    def _set_status(self, msg: str):
        self.status.configure(text=msg)

    def _load_all(self):
        db_path = self.db_var.get().strip()
        idx_path = self.idx_var.get().strip()
        model_name = self.model_var.get().strip() or "jinaai/jina-embeddings-v3"

        if not os.path.exists(db_path):
            messagebox.showerror(APP_TITLE, f"DB not found: {db_path}")
            return
        if not os.path.exists(idx_path):
            messagebox.showerror(APP_TITLE, f"Index not found: {idx_path}")
            return

        def worker():
            try:
                self._set_status("Opening DB…")
                conn = row_factory(sqlite3.connect(db_path, check_same_thread=False))
                conn.execute("PRAGMA foreign_keys = ON;")

                # Lazy imports here (so the window can open even if these fail)
                self._set_status("Loading FAISS index…")
                import faiss  # type: ignore
                index = faiss.read_index(idx_path)
                dim = index.d

                self._set_status("Loading model (this can take a moment)…")
                import torch  # type: ignore
                from sentence_transformers import SentenceTransformer  # type: ignore

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)

                test = model.encode(["ping"], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
                if test.shape[1] != dim:
                    raise RuntimeError(f"Model dim {test.shape[1]} != index dim {dim}. Use the same model as index.")

                self.conn, self.index, self.index_dim, self.model = conn, index, dim, model
                self._set_status(f"Loaded. device={self.device} dim={dim} vectors={self.index.ntotal}")
                self.after(0, self._refresh_books)
            except Exception as e:
                msg = f"Search error:\n{e}"
                def show():
                    self._set_status("Search error.")
                    messagebox.showerror(APP_TITLE, msg)
                self.after(0, show)

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_books(self):
        if not self.conn:
            return
        for i in self.books_tv.get_children():
            self.books_tv.delete(i)
        try:
            # >>> add this line
            with self.db_lock:
                rows = list_books(self.conn)
            # <<< end change
            for r in rows:
                self.books_tv.insert(
                    "",
                    "end",
                    iid=str(r["id"]),
                    values=(
                        r["id"],
                        r["gutenberg_id"] if r["gutenberg_id"] is not None else "",
                        r["ia_title_id"] if r["ia_title_id"] is not None else "",
                        r["title"] or "",
                        r["chunks"] or 0,
                        r["embedded"] or 0
                    )
                )
        except Exception as e:
            msg = f"Search error:\n{e}"
            def show():
                self._set_status("Search error.")
                messagebox.showerror(APP_TITLE, msg)
            self.after(0, show)


    def _selected_book_ids(self) -> List[int]:
        sel = self.books_tv.selection()
        return [int(iid) for iid in sel] if sel else []

    def _do_search(self):
        if not (self.conn and self.index and self.model):
            messagebox.showinfo(APP_TITLE, "Load DB/Index/Model first.")
            return
        q = self.q_text.get("1.0", "end").strip()
        if not q:
            return
        try:
            k = int(self.k_var.get())
            min_score = float(self.min_var.get())
            fetch_k = int(self.fetchk_var.get())
        except Exception:
            messagebox.showerror(APP_TITLE, "Invalid parameters.")
            return
        book_ids = set(self._selected_book_ids()) or None

        # clear previous
        for i in self.res_tv.get_children():
            self.res_tv.delete(i)
        self.snip.delete("1.0", "end")
        self._set_status("Searching…")

        def worker(q=q, k=k, min_score=min_score, fetch_k=fetch_k, book_ids=book_ids):
            try:
                qv = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
                if qv.shape[1] != self.index_dim:
                    raise RuntimeError("Model/index dimension mismatch.")
                _fetch_k = max(fetch_k, k)
                D, I = self.index.search(qv, _fetch_k)
                ids = [int(i) for i in I[0] if i != -1]
                scores = D[0][:len(ids)].tolist()
                with self.db_lock:
                    rows = fetch_chunks_by_ids(self.conn, ids)
                self.last_rows = rows

                results = []
                for cid, score in zip(ids, scores):
                    row = rows.get(cid)
                    if not row: continue
                    if book_ids and row["book_id"] not in book_ids: continue
                    if score < min_score: continue
                    results.append((cid, score, row))
                    if len(results) >= k:
                        break

                def update_ui():
                    if not results:
                        self._set_status("No results. Try raising Fetch-K or lowering Min cosine.")
                        return
                    for rank, (cid, score, row) in enumerate(results, start=1):
                        self.res_tv.insert(
                            "", "end", iid=str(cid),
                            values=(rank, f"{score:.3f}", row["book_id"],
                                    row["chunk_index"], row["title"] or "", row["source_url"] or "")
                        )
                    self._set_status(f"Found {len(results)} result(s).")
                self.after(0, update_ui)
            except Exception as e:
                msg = f"Search error:\n{e}"
                def show():
                    self._set_status("Search error.")
                    messagebox.showerror(APP_TITLE, msg)
                self.after(0, show)

        threading.Thread(target=worker, daemon=True).start()

    def _open_selected_url(self, event=None):
        sel = self.res_tv.selection()
        if not sel: return
        cid = int(sel[0])
        row = self.last_rows.get(cid)
        if row and row["source_url"]:
            webbrowser.open(row["source_url"])

    def _show_selected_snippet(self, event=None):
        sel = self.res_tv.selection()
        if not sel: return
        cid = int(sel[0])
        row = self.last_rows.get(cid)
        if not row: return
        self.snip.delete("1.0", "end")
        self.snip.insert("1.0", summarize(row["text"], width=100, max_lines=14))

if __name__ == "__main__":
    app = App()
    app.mainloop()
