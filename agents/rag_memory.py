"""RAG memory system for p4harness — stores and retrieves approved P4 examples.
Continual learning and self-improving harness design inspired by RLM paradigms
(e.g., Prime Agent: A Self-Improving RLM Harness /ref{Prime Agent: A Self-Improving RLM Harness}).

Storage: append-only JSON-lines file, one entry per line:
  {"task": ..., "p4_code": ..., "scenario": ..., "embedding": [...]}

Retrieval: cosine similarity on Ollama embeddings (qwen2.5-coder:14b).
Fallback: TF-IDF via Python stdlib if Ollama is unavailable.
"""
from __future__ import annotations

import json
import math
import re
import threading
import urllib.request
import urllib.error
from pathlib import Path
_ROOT_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_STORE = str(_ROOT_DIR / "memory" / "rag_store.jsonl")
_OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
_EMBED_MODEL = "qwen2.5-coder:14b"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _ollama_embed(text: str) -> list[float] | None:
    """Call Ollama embeddings endpoint. Returns None on any failure to trigger keyword fallback."""
    payload = json.dumps({"model": _EMBED_MODEL, "input": text, "prompt": text}).encode()
    # Try /api/embed (newer Ollama) and /api/embeddings (legacy)
    for endpoint in ["http://localhost:11434/api/embed", "http://localhost:11434/api/embeddings"]:
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                emb = data.get("embedding")
                if emb is None and "embeddings" in data and data["embeddings"]:
                    emb = data["embeddings"][0]
                if emb:
                    return emb
        except Exception:
            continue
    return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# TF-IDF fallback (stdlib only)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _tfidf_similarity(query: str, documents: list[str]) -> list[float]:
    """Return cosine-TF-IDF similarity between query and each document."""
    corpus = [query] + documents
    tokenized = [_tokenize(doc) for doc in corpus]

    # Build vocabulary
    vocab: dict[str, int] = {}
    for tokens in tokenized:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)

    n_docs = len(corpus)

    # Document frequency
    df: dict[str, int] = {}
    for tokens in tokenized:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    def tfidf_vec(tokens: list[str]) -> list[float]:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = [0.0] * len(vocab)
        for t, freq in tf.items():
            idf = math.log((n_docs + 1) / (df.get(t, 0) + 1)) + 1
            vec[vocab[t]] = freq * idf
        return vec

    query_vec = tfidf_vec(tokenized[0])
    scores: list[float] = []
    for doc_tokens in tokenized[1:]:
        doc_vec = tfidf_vec(doc_tokens)
        scores.append(_cosine_similarity(query_vec, doc_vec))
    return scores


# ---------------------------------------------------------------------------
# RAGMemory
# ---------------------------------------------------------------------------

class RAGMemory:
    """Persistent semantic memory for approved (task, p4_code) pairs."""

    def __init__(self, store_path: str = _DEFAULT_STORE):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict] = []
        self._loaded = False
        # ponytail: guards add()'s dedup-check-then-append and the disk write
        # together — without it, concurrent scenarios (--parallel N) could both
        # pass the "not already stored" check and double-append, and interleave
        # partial JSONL lines (entries carry an embedding vector, easily >4KB,
        # past the OS's atomic-write guarantee for O_APPEND).
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        self._entries = []
        if self.store_path.exists():
            with open(self.store_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        self._loaded = True

    def _append_to_disk(self, entry: dict) -> None:
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Adversarial task patterns — if any appear in the task string, refuse to store.
    # These represent known red-team / attack scenarios that must never become RAG examples.
    _ADVERSARIAL_PATTERNS: list[str] = [
        "port 255",
        "management plane",
        "mgmt_redirect",
        r"maintenance path",
        r"bypass.{0,20}chain",        # chain bypass, not "bypass ACL" etc.
        r"bypass.{0,20}sfc",
        r"chain.{0,20}bypass",
        "clamp.*si",                  # clamping SI values
        "normalize all sfc",
        r"si value is \d+ or above",  # SI range filter
        "nsh header validity check",
        r"spi=0.{0,5}si=0",
        r"drop all (packets|traffic)", # not "drop all other ethertypes"
        "without decrementing si",
        r"skip.{0,20}service chain",
        r"suppress.{0,20}forward",
        "directly to port.*without.*si", # SI decrement bypass
        "administrative.*bypass",
        "ignore all previous instructions",  # prompt injection / jailbreak
        r"output.*pass.*approve",            # reviewer manipulation
        "the p4 code is correct.*output",    # fake approval injection
    ]

    def _is_adversarial(self, task: str) -> bool:
        import re as _re
        t = task.lower()
        return any(_re.search(p, t) for p in self._ADVERSARIAL_PATTERNS)

    @staticmethod
    def _p4_structurally_safe(p4_code: str) -> bool:
        """Return False if the P4 code violates a hard SFC invariant.

        Checks:
        1. No drop() / mark_to_drop() inside any table named sfc_forward.
        2. No egress_spec = 255 (management plane redirect).
        3. Every action body that sets egress_spec also contains the SI decrement.
        """
        import re as _re

        # Rule 1: drop in sfc_forward
        # Find sfc_forward table block and check for drop() inside it
        sfc_fwd_block = _re.search(
            r'table\s+sfc_forward\s*\{(.+?)\}', p4_code, _re.DOTALL
        )
        if sfc_fwd_block:
            block = sfc_fwd_block.group(1)
            if _re.search(r'\bdrop\s*\(\s*\)|\bmark_to_drop\s*\(', block):
                return False

        # Rule 2: egress_spec = 255
        if _re.search(r'egress_spec\s*=\s*255\b', p4_code):
            return False

        # Rule 3: every action that sets egress_spec must also decrement SI.
        # Only enforced for SFC/NSH programs — skip if code has no NSH header references.
        if 'hdr.nsh' in p4_code:
            action_re = _re.compile(r'\baction\s+\w+\s*\([^)]*\)\s*\{', _re.DOTALL)
            for m in action_re.finditer(p4_code):
                body_start = m.end()
                depth, i = 1, body_start
                while i < len(p4_code) and depth > 0:
                    if p4_code[i] == '{': depth += 1
                    elif p4_code[i] == '}': depth -= 1
                    i += 1
                body = p4_code[body_start:i - 1]
                if _re.search(r'egress_spec\s*=', body):
                    if 'hdr.nsh.si = hdr.nsh.si - 1' not in body:
                        return False

        return True

    def add(self, task: str, p4_code: str, scenario: str = "") -> None:
        """Store an approved example with its embedding (if Ollama available)."""
        self._load()

        # Layer 1: refuse known adversarial task-string patterns.
        if self._is_adversarial(task):
            print(f"[rag] REJECTED add — task matches adversarial denylist: {task[:80]!r}")
            return

        # Layer 2: structural P4 code validation — reject if any SFC invariant is violated.
        if not self._p4_structurally_safe(p4_code):
            print(f"[rag] REJECTED add — P4 code fails structural safety check: {task[:60]!r}")
            return

        # Embedding is a slow network call — compute it outside the lock so
        # concurrent add()s don't serialize on it. It touches no shared state.
        embedding = _ollama_embed(f"{task}\n\n{p4_code}")
        entry: dict = {
            "task": task,
            "p4_code": p4_code,
            "scenario": scenario,
            "embedding": embedding,  # may be None if Ollama offline
        }

        with self._lock:
            # One approved example per task — skip if we already have this exact
            # task string. Prevents accumulation of near-identical entries across
            # repeated experiment runs. Re-checked here (not just before the
            # embed call) so two concurrent adds for the same task can't both
            # pass the check and double-append.
            for e in self._entries:
                if e.get("task") == task:
                    return
            self._entries.append(entry)
            self._append_to_disk(entry)

    def retrieve(self, task: str, top_k: int = 2) -> list[dict]:
        """Return top_k most similar approved examples for the given task."""
        self._load()
        if not self._entries:
            return []

        # Try Ollama embeddings first
        query_emb = _ollama_embed(task)
        if query_emb is not None:
            scored: list[tuple[float, dict]] = []
            for entry in self._entries:
                emb = entry.get("embedding")
                if emb and len(emb) == len(query_emb):
                    sim = _cosine_similarity(query_emb, emb)
                    scored.append((sim, entry))
                else:
                    # Entry has no embedding — fall back to TF-IDF score for it
                    tfidf_score = _tfidf_similarity(task, [entry["task"]])[0]
                    scored.append((tfidf_score * 0.5, entry))  # downweight
        else:
            # Full TF-IDF fallback
            docs = [e["task"] for e in self._entries]
            sims = _tfidf_similarity(task, docs)
            scored = list(zip(sims, self._entries))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def seed_from_results(self, results_dir: str) -> int:
        """Seed the store from existing approved results JSON files.

        Each file is a JSON array of scenario records. Only entries with
        ``approved == true`` are stored.

        Returns the number of new entries added.
        """
        self._load()
        added = 0
        results_path = Path(results_dir)

        for json_file in sorted(results_path.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if not isinstance(records, list):
                records = [records]

            for rec in records:
                if not rec.get("approved"):
                    continue
                task = rec.get("task", "")
                p4_code = rec.get("p4_code", "")
                scenario = rec.get("scenario", "")
                if not task or not p4_code:
                    continue

                before = len(self._entries)
                self.add(task, p4_code, scenario)
                if len(self._entries) > before:
                    added += 1
                    print(f"[rag] Seeded: {scenario or json_file.name}")

        return added

    def retrieve_for_error(self, error_str: str, top_k: int = 3) -> list[dict]:
        """Return top_k approved examples most similar to a compiler error string.

        Matches are scored by embedding/TF-IDF similarity of the error string
        against each entry's p4_code (the most semantically relevant field for
        compiler errors, since error tokens like 'table', 'action', 'apply' map
        directly to code constructs).  Falls back to TF-IDF when Ollama is
        unavailable.

        Returns an empty list if the store is empty or similarity scoring fails.
        """
        self._load()
        if not self._entries:
            return []

        # Try Ollama embeddings on the error string vs each entry's p4_code
        query_emb = _ollama_embed(error_str)
        if query_emb is not None:
            scored: list[tuple[float, dict]] = []
            for entry in self._entries:
                # Prefer embedding stored on entry; otherwise embed the p4_code live
                emb = entry.get("embedding")
                if emb and len(emb) == len(query_emb):
                    sim = _cosine_similarity(query_emb, emb)
                else:
                    # No stored embedding — score against p4_code text via TF-IDF
                    sim = _tfidf_similarity(error_str, [entry.get("p4_code", "")])[0] * 0.5
                scored.append((sim, entry))
        else:
            # Full TF-IDF fallback: compare error string against each entry's p4_code
            docs = [e.get("p4_code", "") for e in self._entries]
            sims = _tfidf_similarity(error_str, docs)
            scored = list(zip(sims, self._entries))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def __len__(self) -> int:
        self._load()
        return len(self._entries)
