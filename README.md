# Product Similarity Service

Finds similar products in the Amazon Fashion dataset by brand, colour,
sales_price, weight, and rating. Built for the SAP CXII technical
exercise.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# data/archive.zip is bundled; the app auto-extracts it on first run.
uvicorn app.main:app --reload
# -> http://localhost:8000/find_similar_products?product_id=<id>&num_similar=5
```

Docker:

```bash
docker build -t product-similarity .
docker run -p 8000:8000 product-similarity
```

Part 3 benchmark (brute-force vs FAISS, latency + recall):

```bash
python scripts/build_index.py --num-queries 300 --k 10
```

---

## Part 1 — **find_similar_products**

app/similarity.py (logic) + app/features.py (shared feature engineering,
also used by Part 3) + app/data_loader.py (loading/cleaning).

### The dataset is messier than the exercise spec suggests

Before picking a similarity measure, the fields had to be cleaned:

| Field | Issue | Handling |
|---|---|---|
| weight | Free text ("200 g", "1.2 kg"); **79%** of rows use a 999999999 sentinel for "unknown" | Parsed to grams with unit conversion; sentinel and unparsable text → NaN |
| sales_price | Missing on ~10% of rows | Coerced to float; missing → NaN |
| brand | Missing on ~27%; thousands of distinct values | Missing → "UNKNOWN"; long tail capped to top-N (MAX_CATEGORY_VALUES, default 200) → "OTHER" |
| colour | Missing on ~80% (only ~6k/30k rows have it) | Same treatment as brand |
| rating | Clean — always present, always numeric | Used as-is |

Ignoring this and feeding raw strings/sentinels into a similarity measure
would silently break it — e.g. the 999999999 weight sentinel, treated as
a real value, would dominate every Euclidean distance calculation and make
weight the *only* thing that matters.

### Feature vector

- **Numeric** (sales_price, weight_grams, rating): median-imputed,
  then z-score standardized. Standardizing matters because the raw scales
  are wildly different (price in the hundreds/thousands vs. rating 0-5) —
  without it, price differences would dominate the distance regardless of
  brand or rating match. Median (not mean) imputation because price/weight
  are right-skewed; the median is a more representative "unknown" default.
  Missing values are imputed rather than dropped so a product missing one
  attribute isn't excluded from the whole catalog.
- **Categorical** (brand, colour): one-hot encoded, with missing values
  as their own "UNKNOWN" category and long-tail values bucketed into
  "OTHER". This keeps the vector's dimensionality bounded (~400 dims for
  this dataset) independent of catalog size — important once Part 3 needs
  to hold the whole matrix in memory for a vector index.

### Similarity measure: cosine, not Euclidean

The combined vector mixes a handful of dense numeric dimensions with many
sparse one-hot dimensions. Euclidean distance in that mix is dominated by
the numeric dimensions (they contribute large per-dimension differences)
and effectively ignores whether brand/colour match. Cosine similarity
normalizes for vector magnitude, so a strong categorical match (same
brand *and* colour) contributes proportionally rather than being drowned
out. This is also why standardizing the numeric features first still
matters even under cosine — without it, price alone could still dominate
the vector's direction.

### Tie-breaking

Per the exercise note ("tie-breaker can be based on any attribute"), ties
in cosine score are broken by rating (higher first), then price (lower
first) — deterministic, and a reasonable default recommendation ordering.

### Complexity

Brute-force: O(n) to build the matrix, O(n*d) per query (one
cosine_similarity call against the full matrix — ~54ms measured on this
30k-row, ~400-dim dataset). Fine at this scale; addressed for larger scale
in Part 3.

---

## Part 2 — FastAPI microservice

app/main.py.

```
GET /find_similar_products?product_id=<id>&num_similar=<n>   -> ["id1", "id2", ...]
GET /health                                                    -> {"status": "ok", ...}
```

- num_similar is validated (1 <= num_similar <= MAX_NUM_SIMILAR, default
  cap 100) → 422 if out of range.
- Unknown product_id → 404, not a generic exception.
- Unexpected errors → 500, logged server-side, without leaking internals
  in the response.
- The index is built once at process startup (lifespan), not lazily on
  first request — so the first real request isn't slow.
- docker run's HEALTHCHECK targets /health.

---

## Part 3 (bonus) — vector search for scale

app/vector_index.py. Algorithm: **HNSW** (Hierarchical Navigable Small
World graphs), via faiss.IndexHNSWFlat.

> Malkov, Y. A., & Yashunin, D. A. (2016/2018). *Efficient and robust
> approximate nearest neighbor search using Hierarchical Navigable Small
> World graphs.* IEEE TPAMI. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)

**Why HNSW over IVF or Annoy:**

- Brute-force is already O(n*d) — at 30k rows that's ~54ms, fine for an
  interactive API. The exercise asks to optimize for datasets 100-1000x
  larger, where a full scan per request stops being viable at either
  latency or CPU-cost-per-request.
- HNSW gives sub-linear query time with high recall and, unlike IVF,
  needs no training/clustering pass over the data distribution — it's
  built incrementally, which matters for a catalog that keeps getting new
  products. IVF (+ product quantization) is more memory-efficient at very
  large scale (100M+ vectors), which isn't the constraint here.
- Annoy (tree-based) is simpler to ship, but its recall/speed trade-off
  at a given memory budget is generally worse than HNSW's, and it needs a
  full rebuild for inserts rather than supporting them incrementally.
- Cosine similarity is preserved by L2-normalizing vectors and searching
  with inner product (METRIC_INNER_PRODUCT) — same notion of
  "similar" as Part 1, computed approximately instead of exhaustively, and
  built from the *same* feature vectors (app/features.py) so results
  are directly comparable.

**Measured** (scripts/build_index.py, 300 random queries, k=10, this
30k-row dataset, default efSearch=64):

| Backend | mean latency | p95 latency |
|---|---|---|
| Brute-force (Part 1) | 54.9ms | 61.7ms |
| FAISS HNSW (Part 3) | 0.9ms | 1.5ms |

~60x faster per query. Index build time is the trade-off: 1.5s
(brute-force, no real "build" needed) vs. ~10s (HNSW graph construction) —
amortized once at startup, worth it once query volume is high.

**Recall caveat, measured honestly:** recall@10 against the brute-force
ground truth is ~0.77-0.80, not the >95% HNSW is usually capable of.
Investigating why: 42% of products in this dataset (12,500/30,000) share
an *exact-duplicate* feature vector with at least one other product — a
direct consequence of how sparse the source data is (missing brand/colour/
price/weight all collapse to the same imputed defaults). For those
products there are many genuinely-tied nearest neighbors, and brute-force
only wins the "ground truth" comparison because it applies a rating/price
tie-break that isn't part of the vector FAISS searches over — both answers
are valid, they just don't agree on *which* tied member to return. This is
a data-sparsity artifact, not evidence that HNSW's approximation is poor;
efSearch can be tuned up (128, 256 — tested, recall improves to ~0.80-0.82)
at a small latency cost via FAISS_HNSW_EF_SEARCH if exact tie agreement
with the brute-force backend matters for a given use case.

**Usage:** opt-in via SIMILARITY_BACKEND=faiss (default is brute, exact,
and already fast enough for this dataset — no reason to trade correctness
for speed until the catalog actually grows).

---

## Other design decisions & trade-offs

- **Data shipped as data/archive.zip, not the extracted 74MB ldjson** —
  keeps the Docker image and git repo smaller; extracted once, lazily, on
  first run (see data_loader._ensure_data_file_present).
- **In-memory index only** — no persistence of the fitted feature matrix
  or FAISS index to disk. Fine for a single-node demo.
- **No image-based or text-embedding similarity** — the exercise's
  suggested attribute set (brand, colour, price, weight, rating) is
  entirely structured/tabular, so it didn't justify pulling in a CNN or
  transformer just to say the box was checked. Documented as a next step,
  not implemented, in the interest of the ~4-6h scope: product_name /
  meta_keywords are free text and would need TF-IDF or sentence
  embeddings; image_urls would need downloading + a pretrained CNN
  (ResNet/EfficientNet) for a visual embedding, combined with the
  structured score behind a configurable weight.
- **num_similar capped at 100** (MAX_NUM_SIMILAR) — an unbounded value
  is either meaningless (a "top-100000-similar" list isn't a
  recommendation) or a DoS vector (forcing a full O(n) sort per request).



## For more details - Kindly refer
DESIGN.md          
PART3_VECTOR_SEARCH.md  

