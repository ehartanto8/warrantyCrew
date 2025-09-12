import os, io, json, hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import tiktoken
from pypdf import PdfReader
import docx2txt
from openai import OpenAI

OPENAI_EMBED_MODEL = "text-embedding-3-small" # -large more expensive
INDEX_DIR = os.getenv("KB_INDEX_DIR", "/data/vm_kb")
DOCS_DIR = os.getenv("KB_DOCS_DIR", os.path.join(os.path.dirname(__file__), "docs"))
EMB_DTYPE = np.float32
CHUNK_TOKENS = int(os.getenv("KB_CHUNK_TOKENS", "500")) # more precise than embeddings
CHUNK_OVERLAP_TOKENS = int(os.getenv("KB_CHUNK_OVERLAP", "60")) # avoids cutting sentences

os.makedirs(INDEX_DIR, exist_ok = True)

# Chunk record
@dataclass
class KBChunk:
    text: str
    source: str
    page: Optional[int]
    sha: str # detect duplicates

# Main class
class SimpleKB:
    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client or OpenAI()
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vectors: Optional[np.ndarray] = None # (N, d), vector = coordinates
        self.meta: List[Dict[str, Any]] = []
        self.index_fp = os.path.join(INDEX_DIR, "kb_vectors.npy")
        self.meta_fp = os.path.join(INDEX_DIR, "kb_meta.json")

        self.load()

        # If nothing loaded, build index once at startup
        # If index exists, reindex only if docs digest changed
        try:
            current = self._docs_digest(DOCS_DIR)
        except Exception as e:
            current = ""
            print(f"[KB] digest failed: {e}")

        saved = self._load_saved_digest()

        if self.vectors is None or not len(self.meta):
            try:
                n, d = self.reindex()
                if current:
                    self._save_digest(current)
                print(f"[KB] Auto-built index at startup with {n} chunks, dim {d}")
            except Exception as e:
                print(f"[KB] Startup reindex failed: {e}")
        elif current and saved and current != saved:
            try:
                n, d = self.reindex()
                self._save_digest(current)
                print(f"[KB] Detected docs change; reindexed {n} chunks, dim {d}")
            except Exception as e:
                print(f"[KB] Change-detec reindex failed: {e}")
        else:
            if current and not saved:
                self._save_digest(current)
            print("[KB] Loaded existing index, no doc changes detected")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Return top-k chunks as {page_content, metadata} dicts. Common RAG tool outputs.
        if self.vectors is None or not len(self.meta):
            return []
            qvec = self._embed_texts([query])[0]
            sims = self._cosine_sim(self.vectors, qvec)
            idx = np.argsort(-sims)[:k] # descending, top-k indices
            out = []
            for i in idx:
                m = dict(self.meta[i])
                m["score"] = float(sims[i])
                out.append({
                    "page_content": m.pop("text", ""),
                    "metadata": m
                })
            return out

    # Rebuild the vectors (rebuild the coordinates), since changing, moving documents in data, the text and embeddings need to be refreshed
    def reindex(self) -> Tuple[int, int]:
        # Re-scan DOCS_DIR and rebuild the index.
        chunks = self._load_all_chunks(DOCS_DIR)
        if not chunks: # reset/clear
            self.vectors = None
            self.meta = []
            self._save()
            return (0, 0)

        texts = [c.text for c in chunks] # embedding purpose
        vecs = self._embed_texts(texts)
        self.vectors = vecs.astype(EMB_DTYPE)
        self.meta = [{
            "source": c.source,
            "page": c.page,
            "sha": c.sha,
            "text": c.text
        } for c in chunks] # stores richer info

        self._save()
        return (len(chunks), self.vectors.shape[1])

    def _save(self):
        if self.vectors is not None:
            np.save(self.index_fp, self.vectors)
        else:
            try: os.remove(self.index_fp)
            except FileNotFoundError: pass

        with open(self.meta_fp, "w", encodings = "utf-8") as f:
            json.dump(self.meta, f)

    def _load(self):
        try:
            self.vectors = np.load(self.index_fp).astype(EMB_DTYPE)
            with open(self.meta_fp, "r", encoding = "utf-8") as f:
                self.meta = json.load(f)
        except Exception:
            self.vectors, self.meta = None, []

    # Embeddings (batched)
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        out = []
        B = 64 # batch size
        for i in range(0, len(texts), B):
            batch = texts[i : i + B]
            resp = self.client.embeddings.create(model = OPENAI_EMBED_MODEL, input = batch)
            vecs = [np.array(e.embedding, dtype = EMB_DTYPE) for e in resp.data]
            out.append(np.vstack(vecs))
        return np.vstack(out)

    # Check for similarity using cosine. -1 (opposite), 1 (identical). Angle between 2 vectors.
    def _cosine_sim(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        denom = (np.linalg.norm(A, axis = 1) * np.linalg.norm(b) + 1e-8)
        return (A @ b) / denom

    # Chunk, by tokens (embeddings & LLMS cont tokens.
    def _chunk_text(self, text:str) -> List[str]:
        toks = self.enc.encode(text)
        chunks = []
        step = max(1, CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS) # ensures adjacent windows overlap, so not cutting the word
        for i in range(0, len(toks), step):
            window = toks[i:i + CHUNK_TOKENS]
            if not window: break
            chunks.append(self.enc.decode(window))
        return chunks

    def _load_all_chunks(self, root: str) -> List[KBChunk]:
        chunks: List[KBChunk] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                fpath = os.path.join(dirpath, fn)
                ext = os.path.splitext(fn.lower())[1]
                try:
                    if ext == ".pdf":
                        chunks.extend(self._pdf_chunks(fpath))
                    elif ext == ".docx":
                        chunks.extend(self._pdf_chunks(fpath))
                    elif ext in (".txt"):
                        chunks.extend(self._text_chunks(fpath))
                except Exception as e:
                    print(f"Failed to load {fpath}: {e}")
        return chunks

    # PDF loader, page-by-page
    def _pdf_chunks(self, fpath: str) -> List[KBChunk]:
        reader = PdfReader(fpath)
        out: List[KBChunk] = []
        for i, page in enumerate(reader.pages, start = 1):
            txt = page.extract_text() or ""
            if not txt.strip(): continue
            for ch in self._chunk_text(txt):
                sha = hashlib.sha1((fpath + str(i) + ch).encode("utf-8")).hexdigest()
                out.append(KBChunk(text = ch, source = fpath, page = i, sha = sha))
        return out

    # Docx loader, read entire body at once
    def _docx_chunks(self, fpath: str) -> List[KBChunk]:
        txt = docx2txt.process(fpath) or ""
        out: List[KBChunk] = []
        for ch in self._chunk_text(txt):
            sha = hashlib.sha1((fpath + ch).encode("utf-8")).hexdigest()
            out.append(KBChunk(text = ch, source = fpath, page = None, sha = sha))
        return out

    # Txt loader
    def _text_chunks(self, fpath: str) -> List[KBChunk]:
        with io.open(fpath, "r", encoding = "utf-8", errors = "ignore") as f:
            txt = f.read()
        out: List[KBChunk] = []
        for ch in self._chunk_text(txt):
            sha = hashlib.sha1((fpath + ch).encode("utf-8")).hexdigest()
            out.append(KBChunk(text = ch, source = fpath, page = None, sha = sha))
        return out

    # Auto-reindex check
    def _docs_digest(self, root:str) -> str:
        # Compute a stable digest of the docs tree (paths + sizes + mtimes)
        items = []
        for dirpath, _, filenames in os.walk(root): # _ don't need the dirnames
            for fn in sorted(filenames):
                fpath = os.path.join(dirpath, fn)
                try:
                    st = os.stats(fpath)
                except FileNotFoundError:
                    continue
                rel = os.path.relpath(fpath, root)
                items.append(f"{rel}|{st.st_size}|{int(st.st_mtime)}")
        data = "\n".join(items).encode("utf-8")
        return hashlib.sha1(data).hexdigest()

    def _load_saved_digest(self) -> str:
        fp = os.path.join(INDEX_DIR, "docs_digest.txt")
        try:
            with open(fp, "r", encoding = "utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""

    def _save_digest(self, digest:str):
        fp = os.path.join(INDEX_DIR, "docs_digest.txt")
        with open(fp, "w", encoding = "utf-8") as f:
            f.write(digest)