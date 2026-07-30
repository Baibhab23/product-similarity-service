# Part 3 — ANN Vector Search: HNSW vs IVF

## The Problem

The brute-force backend checked every product for every query — O(n*d) per request. At 30k rows that was about 20ms, which was fine. But at 1 million+ products, that would become seconds per request. The solution was approximate nearest neighbor (ANN) search: accept a small loss in accuracy in exchange for much faster queries.

Two ANN algorithms were implemented and benchmarked: **FAISS HNSW** and **FAISS IVF**.

---

## FAISS HNSW

**Reference**: Malkov & Yashunin (2016/2018). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE TPAMI. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)

HNSW builds a multi-layer graph over the vector space. The top layers are sparse and used for fast navigation; the bottom layer is a dense proximity graph for fine-grained search. A query starts at the top, descends toward the target, then does a local beam search at the bottom — O(log n) per query.

**Key advantage:** Supports incremental inserts without any retraining. New products can be added to a live catalog without rebuilding the index.

### HNSW Parameters

| Parameter | Default | Effect |
|---|---|---|
| M | 32 | Links per node per layer — higher = better recall, more memory |
| efConstruction | 200 | Beam width during build — higher = better graph quality, slower build |
| efSearch | 64 | Beam width during query — higher = better recall, slower queries |

All configurable via environment variables (FAISS_HNSW_M, FAISS_HNSW_EF_CONSTRUCTION, FAISS_HNSW_EF_SEARCH).

---

## FAISS IVF

**Reference**: Johnson, Douze & Jégou (2019). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)

IVF partitions the vector space into **nlist** Voronoi cells using k-means clustering. At query time only the **nprobe** nearest cells are searched — sub-linear cost. Build is fast once training completes (~0.21s vs HNSW's ~1.40s), but the training pass itself requires the full dataset upfront.

**Key limitation:** Adding new products requires rebuilding the index from scratch (k-means clusters must be recomputed). Not ideal for a catalog that grows continuously.

### IVF Parameters

| Parameter | Default | Effect |
|---|---|---|
| nlist | 100 | Number of Voronoi cells — more cells = faster queries, lower recall |
| nprobe | 10 | Cells searched per query — higher = better recall, slower queries |

Configurable via FAISS_IVF_NLIST and FAISS_IVF_NPROBE environment variables.

---

## Cosine Similarity via Inner Product

Both backends use the same approach as Part 1: vectors are L2-normalized to unit length, then FAISS searches with METRIC_INNER_PRODUCT. Inner product on unit vectors equals cosine similarity. This keeps the definition of "similar" identical across all three backends — results are directly comparable.

---

## Three-way Benchmark

300 random queries, k=10, 30k-row dataset. Full results in the README.

### Why Recall Is Lower Than Expected

Both algorithms typically achieve >95% recall. Here HNSW is ~0.78 and IVF is ~0.93. The gap is a data quality issue, not an algorithm problem. About 42% of products share an identical feature vector — missing brand, colour, price, and weight all collapse to the same imputed defaults. For those products, brute-force picks a specific tie-winner using rating and price, but neither ANN backend applies that tie-break. Both ANN answers are valid; they just disagree on *which* tied neighbor to return. This is a data-sparsity artifact. IVF's higher recall here is partly because its cell-based search structure happens to agree with brute-force's tie-breaking more often than HNSW's graph traversal does.

---

## vs. Annoy

**Reference**: Bernhardsson (2013). *Annoy: Approximate Nearest Neighbors in C++/Python.* [github.com/spotify/annoy](https://github.com/spotify/annoy). Benchmark: Aumüller et al. (2020). *ANN-Benchmarks.* [arXiv:1807.05614](https://arxiv.org/abs/1807.05614)

Annoy (tree-based) was not implemented because: (1) its recall/speed trade-off at a given memory budget is generally worse than HNSW's per ANN-Benchmarks, and (2) like IVF, it requires a full rebuild for every insert — making it impractical for a live catalog. HNSW and IVF cover the speed/recall trade-off space more usefully for this use case.

---

## Default Backend: Brute-force

Both ANN backends are approximate. For 30k products, brute-force is exact and already fast — no reason to trade correctness for speed at this scale. The ANN backends are opt-in via the SIMILARITY_BACKEND environment variable.
