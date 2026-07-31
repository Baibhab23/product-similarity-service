# Product Similarity Service

Finds similar products in the Amazon Fashion dataset by brand, colour,
sales_price, weight, rating, and visual appearance. Built for the SAP CXII technical
exercise.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The dataset archive is bundled; the app auto-extracts it on first run.
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

The similarity logic, shared feature engineering, and dataset loading/cleaning modules.

### The dataset is messier than the exercise spec suggests

Before picking a similarity measure, the fields had to be cleaned:

| Field | Issue | Handling |
|---|---|---|
| weight | Free text ("200 g", "1.2 kg"); **79%** of rows use a 999999999 sentinel for "unknown" | Parsed to grams with unit conversion; sentinel and unparsable text → NaN |
| sales_price | Missing on ~10% of rows | Coerced to float; missing → NaN |
| brand | Missing on ~27%; thousands of distinct values | Missing → "UNKNOWN"; long tail capped to top-N (default 200) → "OTHER" |
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

```
GET /find_similar_products?product_id=<id>&num_similar=<n>   -> ["id1", "id2", ...]
GET /health                                                    -> {"status": "ok", ...}
```

- num_similar is validated (1 <= num_similar <= 100) → 422 if out of range.
- Unknown product_id → 404, not a generic exception.
- Unexpected errors → 500, logged server-side, without leaking internals
  in the response.
- The index is built once at process startup, not lazily on
  first request — so the first real request isn't slow.
- The Docker health check targets /health.

---

## Part 3 (bonus) — vector search for scale

Two ANN backends implemented and benchmarked: **HNSW** and **IVF**, both via FAISS.

| Backend | Build time | Mean latency | p95 latency | Recall@10 |
|---|---|---|---|---|
| Brute-force | 0.06s | 20.6ms | 26.7ms | 1.00 (exact) |
| FAISS HNSW | 1.40s | 0.36ms | 0.60ms | 0.78 |
| FAISS IVF | 0.21s | 0.56ms | 0.77ms | 0.93 |

HNSW is fastest; IVF has better recall at ~1.5x the latency. Both are ~55x faster than brute-force per query.

Select backend via env var:
- **SIMILARITY_BACKEND=brute** (default) — exact, no build cost
- **SIMILARITY_BACKEND=faiss** — HNSW, best raw speed
- **SIMILARITY_BACKEND=ivf** — IVF, best recall among ANN options

**Note on Annoy:** Annoy was evaluated as a third candidate but has an unresolved segfault on Apple Silicon (ARM64) across all Python versions — a known open issue in the library ([github.com/spotify/annoy](https://github.com/spotify/annoy/issues)). It was not implemented for that reason. Theoretically, Annoy would be slower than HNSW at the same recall target — as shown in ANN-Benchmarks (Aumüller et al., 2020, [arXiv:1807.05614](https://arxiv.org/abs/1807.05614)) — making HNSW the better choice regardless.

See ANN_ALGORITHM_ANALYSIS for algorithm details, parameter tuning, and trade-off analysis.

---

## Part 4 (bonus) — image-based similarity

Visual embeddings extracted from product images are stored in a persistent
**ChromaDB** collection. At query time, Chroma returns visually similar products
which are merged with the structured-feature results using **Reciprocal Rank
Fusion (RRF)**. Disabled by default — opt in with IMAGE_ENABLED=true.

### How it works

1. A one-time offline build script downloads product images, runs them through
   **MobileNetV3-Small** (576-dim embeddings), and upserts into the Chroma
   collection in batches. Resumable — already-stored products are skipped on re-run.
2. At serve time, each query fires two searches:
   - Structured backend (brute/FAISS/IVF) → ranked list by cosine similarity
   - Chroma → ranked list by visual similarity
3. RRF merges both lists into a final ranking. Products appearing high in both
   lists rank highest; products missing an image fall back gracefully to the
   structured score.

### Setup

```bash
# Step 1 — build the Chroma collection (one-time, ~45 min for 30k products)
python scripts/build_image_vectors.py

# Step 2 — start the server with image similarity enabled
IMAGE_ENABLED=true uvicorn app.main:app --reload
```

Docker (Chroma collection is baked into the image):

```bash
docker build -t product-similarity .
docker run -e IMAGE_ENABLED=true -p 8000:8000 product-similarity
```

### Configuration

| Env var | Default | Description |
|---|---|---|
| IMAGE_ENABLED | false | Enable image similarity + RRF fusion |
| CHROMA_PATH | data/chroma | Path to persistent ChromaDB directory |
| CHROMA_COLLECTION | product_images | Chroma collection name |
| RRF_K | 60 | RRF rank-discounting constant (higher = less aggressive) |

### Design decisions

- **Offline build, not startup inference** — downloading and embedding 30k images
  at server startup would block traffic for 45+ minutes. The build runs
  once and persists vectors to Chroma; the server just opens the collection.
- **ChromaDB over a flat matrix file** — supports upsert by product ID (no full
  rebuild when products change), persistent across restarts, and handles the ANN
  search itself. No full matrix held in memory at serve time.
- **MobileNetV3-Small over EfficientNet-B0** — ~3× faster CPU inference (the
  bottleneck for the offline build), 576-dim output vs. 1280-dim, comparable
  visual feature quality for product images.
- **RRF over weighted score blending** — no score normalization needed across two
  different systems (cosine vs. Chroma's internal distance). RRF is robust to
  scale differences and doesn't require tuning a weight parameter.
- **Zero-fill for missing images** — products with dead URLs or failed downloads
  get a zero vector. They don't contribute to visual similarity but remain
  reachable via the structured backend.

---

## Other design decisions & trade-offs

- **Dataset archive bundled, not the extracted file** —
  keeps the Docker image and git repo smaller; extracted once, lazily, on
  first run.
- **In-memory index only** — no persistence of the fitted feature matrix
  or FAISS index to disk. Fine for a single-node demo.
- **num_similar capped at 100** — an unbounded value
  is either meaningless (a "top-100000-similar" list isn't a
  recommendation) or a DoS vector (forcing a full O(n) sort per request).

---

## Expected Output & Results

The API returns a ranked list of similar product IDs. Below is a sample response from Bruno for **product_id=26d41bdc1495de290bc8e6062d927729&num_similar=5**:

![API Output](docs/Output.png)

**200 OK** — 5 similar product IDs returned in ~233ms (brute-force backend, cold start included).

---

## For more details

See DESIGN.md and ANN_ALGORITHM_ANALYSIS.md.
