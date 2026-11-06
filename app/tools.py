import csv
import hashlib
import html
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import httpx
import torch
import torch.nn as nn

from app import config

LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
FIELDS = ("speaker", "party", "job", "state", "subject", "context")
COUNT_KEYS = ("barely_true", "false", "half_true", "mostly_true", "pants_on_fire")
HASH_SALT = b"disinfo-aitu-v1"
# Per-field hash buckets of the thesis config/hybrid.yaml (2026-04-27 and later).
# Older checkpoints use 256 buckets per field; the layout is chosen from the embedding size.
PER_FIELD_BUCKETS = {"speaker": 4096, "party": 64, "job": 1024, "state": 256, "subject": 1024, "context": 1024}
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job", "state", "party",
    "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts",
    "pants_on_fire_counts", "context",
]  # fmt: skip
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML = re.compile(r"<[^>]+>")

# Baseline reliability by publisher; the assessor's LLM score is blended with this.
RELIABLE_DOMAINS = {
    "politifact.com": 0.9, "factcheck.org": 0.9, "snopes.com": 0.85, "apnews.com": 0.9, "reuters.com": 0.9,
    "bbc.com": 0.85, "bbc.co.uk": 0.85, "npr.org": 0.85, "nytimes.com": 0.8, "washingtonpost.com": 0.8,
    "wsj.com": 0.8, "theguardian.com": 0.8, "wikipedia.org": 0.75, "britannica.com": 0.8,
    "cbo.gov": 0.9, "census.gov": 0.9, "cdc.gov": 0.9, "bls.gov": 0.9, "congress.gov": 0.9,
}  # fmt: skip
LOW_DOMAINS = {
    "facebook.com": 0.2, "x.com": 0.2, "twitter.com": 0.2, "reddit.com": 0.25, "medium.com": 0.35,
    "quora.com": 0.2, "youtube.com": 0.3, "tiktok.com": 0.2, "instagram.com": 0.2,
}  # fmt: skip


def clean_text(text: str) -> str:
    """Same light cleaning RoBERTa saw in training: unescape, drop URLs/HTML, squeeze spaces."""
    text = html.unescape(str(text or "")).strip()
    text = _HTML.sub(" ", _URL.sub(" ", text))
    return re.sub(r"\s+", " ", text).strip()


def hash_bucket(value: str | None, num_buckets: int) -> int:
    """Deterministic blake2b feature hashing, identical to the thesis implementation."""
    raw = "" if value is None else str(value)
    digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=8, key=HASH_SALT).digest()
    return int.from_bytes(digest, "big") % num_buckets


def slugify(name: str) -> str:
    """'Donald Trump' -> 'donald-trump' (LIAR speaker ids)."""
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def dense_features(counts: dict[str, float]) -> list[float]:
    """5-dim credibility vector + 4 normalized scalars, as in the thesis preprocessing."""
    c = [float(counts.get(k) or 0.0) for k in COUNT_KEYS]
    total = sum(c)
    if total <= 0:
        return [0.2] * 5 + [0.0] * 4
    false, pants = c[1], c[4]
    return [x / total for x in c] + [total / 50.0, math.log1p(total) / 5.0, pants / total, (false + pants) / total]


class MetadataBranch(nn.Module):
    def __init__(self, bucket_sizes: list[int], embedding_dim: int, output_dim: int) -> None:
        super().__init__()
        self.categorical_embedding = nn.Embedding(sum(bucket_sizes), embedding_dim)
        offsets = torch.tensor([sum(bucket_sizes[:i]) for i in range(len(bucket_sizes))])
        self.register_buffer("field_offsets", offsets, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(9 + embedding_dim * len(bucket_sizes), output_dim), nn.GELU(), nn.Dropout(0.0)
        )

    def forward(self, dense: torch.Tensor, categorical_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.categorical_embedding(categorical_ids + self.field_offsets).flatten(1)
        return self.projection(torch.cat([dense, embedded], dim=1))


class HybridClassifier(nn.Module):
    """RoBERTa [CLS] + metadata branch -> fusion MLP -> 6 logits (thesis architecture, inference copy)."""

    def __init__(
        self,
        encoder: nn.Module,
        bucket_sizes: list[int],
        embedding_dim: int,
        metadata_dim: int,
        fusion_hidden: int,
        num_labels: int = 6,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooler_dropout = nn.Dropout(0.0)
        self.metadata_branch = MetadataBranch(bucket_sizes, embedding_dim, metadata_dim)
        self.classifier = nn.Sequential(
            nn.Linear(encoder.config.hidden_size + metadata_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(fusion_hidden, num_labels),
        )

    def forward(self, input_ids, attention_mask, dense, categorical_ids) -> torch.Tensor:
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        fused = torch.cat([pooled, self.metadata_branch(dense, categorical_ids)], dim=1)
        return self.classifier(fused)


class HybridModel:
    """The thesis checkpoint, loaded once; classifies one statement with its speaker metadata."""

    def __init__(self, checkpoint: Path, base_model: Path | str = config.BASE_MODEL_DIR, max_length: int = 64) -> None:
        from transformers import AutoConfig, AutoModel, AutoTokenizer  # heavy import, keep it lazy

        if not checkpoint.exists():
            raise FileNotFoundError(f"Hybrid checkpoint not found: {checkpoint} (set HYBRID_CHECKPOINT)")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        embedding = state["metadata_branch.categorical_embedding.weight"]
        total_buckets = embedding.shape[0]
        if total_buckets == sum(PER_FIELD_BUCKETS.values()):
            per_field = PER_FIELD_BUCKETS
        else:
            per_field = {field: total_buckets // len(FIELDS) for field in FIELDS}
        self.bucket_sizes = [per_field[field] for field in FIELDS]
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(str(base_model)))
        self.model = HybridClassifier(
            encoder,
            self.bucket_sizes,
            embedding.shape[1],
            metadata_dim=state["metadata_branch.projection.0.weight"].shape[0],
            fusion_hidden=state["classifier.0.weight"].shape[0],
        )
        self.model.load_state_dict(state)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(base_model))
        self.max_length = max_length
        self.version = f"{checkpoint.name}@{hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:12]}"

    @torch.no_grad()
    def predict(self, text: str, metadata: dict) -> dict[str, float]:
        encoded = self.tokenizer(clean_text(text), truncation=True, max_length=self.max_length, return_tensors="pt")
        dense = torch.tensor([dense_features(metadata.get("counts") or {})], dtype=torch.float32)
        categorical = torch.tensor(
            [[hash_bucket(metadata.get(f), n) for f, n in zip(FIELDS, self.bucket_sizes, strict=True)]]
        )
        logits = self.model(encoded["input_ids"], encoded["attention_mask"], dense, categorical)
        return dict(zip(LABELS, torch.softmax(logits, dim=1)[0].tolist(), strict=True))


_lock = threading.Lock()
_model: HybridModel | None = None


def get_model(checkpoint: Path) -> HybridModel:
    """Loads the checkpoint on first use (a few seconds) and keeps it for the whole process."""
    global _model
    with _lock:
        if _model is None:
            _model = HybridModel(checkpoint)
        return _model


class SpeakerHistory:
    """LIAR speaker metadata: party, job, state and PolitiFact credibility counts, keyed by speaker slug."""

    def __init__(self, tsv: Path) -> None:
        self.by_slug: dict[str, dict] = {}
        self.source = str(tsv) if tsv.exists() else None
        if not self.source:
            return
        with tsv.open(encoding="utf-8") as handle:
            for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
                row += [""] * (len(LIAR_COLUMNS) - len(row))  # ~1 % of LIAR rows lack the trailing context column
                record = dict(zip(LIAR_COLUMNS, row, strict=True))
                self.by_slug.setdefault(
                    record["speaker"],
                    {
                        "speaker": record["speaker"],
                        "party": record["party"],
                        "job": record["job"],
                        "state": record["state"],
                        "counts": {key: float(record[f"{key}_counts"] or 0) for key in COUNT_KEYS},
                    },
                )

    def lookup(self, name: str | None) -> dict | None:
        return self.by_slug.get(slugify(name)) if name else None


def domain_of(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    return host[4:] if host.startswith("www.") else host


def domain_reliability(domain: str) -> float:
    """Baseline trust in a publisher from a short rule table; unknown publishers get 0.5."""
    for table in (RELIABLE_DOMAINS, LOW_DOMAINS):
        for known, score in table.items():
            if domain == known or domain.endswith("." + known):
                return score
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.85
    return 0.5


class WebSearch:
    """Web search with a result cache in SQLite.

    Backends: `openrouter` (the LLM gateway's web plugin, same key), `tavily` (TAVILY_API_KEY),
    `cache` (offline replay of earlier searches, for demos without network).
    """

    def __init__(self, backend: str, store, llm=None, max_results: int = config.SEARCH_RESULTS) -> None:
        self.backend, self.store, self.llm, self.max_results = backend, store, llm, max_results

    def search(self, query: str) -> tuple[list[dict], bool]:
        """Returns (results, from_cache); a result is {url, title, snippet, published}."""
        cached = self.store.cached_search(
            query, None if self.backend == "cache" else self.backend, config.SEARCH_CACHE_HOURS
        )
        if cached is not None:
            return cached, True
        if self.backend == "cache":
            return [], False
        results = self._tavily(query) if self.backend == "tavily" else self._openrouter(query)
        self.store.save_search(self.backend, query, results)
        return results, False

    def _tavily(self, query: str) -> list[dict]:
        import os

        for attempt in range(3):  # retries only for the transient failures: timeouts, 429, 5xx
            try:
                response = httpx.post(
                    "https://api.tavily.com/search",
                    json={"query": query, "max_results": self.max_results, "search_depth": "basic"},
                    headers={"Authorization": f"Bearer {os.environ.get('TAVILY_API_KEY', '')}"},
                    timeout=config.SEARCH_TIMEOUT_S,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError("transient", request=response.request, response=response)
                response.raise_for_status()
                return [
                    {
                        "url": r["url"],
                        "title": r.get("title", ""),
                        "snippet": r.get("content", ""),
                        "published": r.get("published_date"),
                    }
                    for r in response.json().get("results", [])
                ]
            except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError):
                if attempt == 2:
                    raise
                time.sleep(1.0 + attempt)
        return []

    def _openrouter(self, query: str) -> list[dict]:
        if self.llm is None:
            raise RuntimeError("openrouter search backend needs the LLM client")
        response = self.llm.client.chat.completions.create(
            model=config.SEARCH_MODEL,
            messages=[
                {"role": "system", "content": "You are a search assistant. Use web search and answer in one sentence."},
                {"role": "user", "content": f"Find reliable web sources that discuss this claim: {query}"},
            ],
            extra_body={  # Exa returns the page excerpts in the citations; the native engines return links only
                "plugins": [{"id": "web", "engine": "exa", "max_results": self.max_results}],
                "usage": {"include": True},
            },
            name="search_web",
            metadata={"agent": "retriever"},
        )
        message = response.choices[0].message
        annotations = getattr(message, "annotations", None) or []
        results, seen = [], set()
        for annotation in annotations:
            data = annotation.model_dump() if hasattr(annotation, "model_dump") else dict(annotation)
            citation = data.get("url_citation") or data
            url = citation.get("url")
            if url and url not in seen:
                seen.add(url)
                results.append(
                    {
                        "url": url,
                        "title": citation.get("title") or "",
                        "snippet": citation.get("content") or "",
                        "published": None,
                    }
                )
        return results


def check_links(urls: list[str]) -> dict[str, bool]:
    """HEAD request per URL in parallel (GET on 405); unreachable links are reported, never retried."""
    headers = {"User-Agent": "claim-check/0.1"}

    def probe(url: str) -> bool:
        try:
            response = httpx.head(url, timeout=4, follow_redirects=True, headers=headers)
            if response.status_code == 405:
                response = httpx.get(url, timeout=4, follow_redirects=True, headers=headers)
            return response.status_code < 400 or response.status_code in (401, 403)  # exists, but blocks bots
        except httpx.HTTPError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(zip(urls, pool.map(probe, urls), strict=True))
