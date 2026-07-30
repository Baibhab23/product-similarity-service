# Part 3 — FAISS HNSW: Algorithm Choice & Implementation

## The Problem

The brute-force backend checked every product for every query. This was O(n*d) per request. At 30k rows it took about 54ms, which was fine. But at 1 million+ products, that would have become seconds per request. I needed a faster approach for large catalogs.

The solution was approximate nearest neighbor search. I accepted a small loss in accuracy in exchange for much faster queries.

---

## Why HNSW

**Reference**: Malkov & Yashunin (2016/2018). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE TPAMI. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)

HNSW built a multi-layer graph over the vector space. The top layers were sparse and used for fast navigation. The bottom layer was a dense proximity graph for fine-grained search. A query started at the top, descended toward the target, then did a local beam search at the bottom.

### vs. IVF (Inverted File Index)

**Reference**: Johnson, Douze & Jégou (2017). *Billion-scale similarity search with GPUs.* IEEE Big Data. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)

IVF clustered the data using k-means before building the index. This required a training pass over the full dataset. That meant new products could not be added without rebuilding from scratch. A product catalog grows continuously, so this was a hard limitation. HNSW supported incremental inserts without any retraining.

IVF made sense at 100M+ vectors where memory was the main constraint. That was not the case here.

**If IVF had been used instead:** The query latency would have been similar — both are sub-millisecond at 30k rows. The recall would also have been comparable with a well-tuned nprobe value. However, the startup time would have been longer because of the k-means clustering step. The bigger practical problem would have appeared when new products needed to be added. With IVF, the entire index would need to be rebuilt from scratch. With HNSW, new products are inserted incrementally. For a product catalog that receives new items regularly, this difference matters significantly in production.

### vs. Annoy

**Reference**: Bernhardsson (2013). *Annoy: Approximate Nearest Neighbors in C++/Python.* [github.com/spotify/annoy](https://github.com/spotify/annoy). Benchmark comparison: Aumüller et al. (2020). *ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms.* Information Systems. [arXiv:1807.05614](https://arxiv.org/abs/1807.05614)

Annoy was simpler to install and ship. But its recall at a given memory budget was generally worse than HNSW, as shown in the ANN-Benchmarks study above. Like IVF, Annoy also required a full rebuild when new products were added. HNSW handled incremental inserts natively, which mattered for a live catalog.

**If Annoy had been used instead:** The query latency at 30k rows would have been similar to HNSW. However, to achieve the same recall as HNSW, Annoy would have required either more trees (meaning more memory) or more nodes checked per query (meaning slower queries). The ANN-Benchmarks paper shows this trade-off clearly — at the same recall target, HNSW consistently used less memory and was faster. The insert limitation would also have applied here. Adding a new product to Annoy requires rebuilding all trees from scratch, making it impractical for a catalog that updates frequently.

---

## Cosine Similarity via Inner Product

I used cosine similarity here, same as Part 1. FAISS with METRIC_INNER_PRODUCT computed inner products. Inner product equaled cosine similarity when vectors were L2-normalized to unit length. I normalized all vectors before adding them to the index.

This kept the definition of similar identical across both backends. I used the same feature vectors from features.py for both. Results were directly comparable between brute-force and FAISS.

---

## HNSW Parameters

| Parameter | Default | Effect |
|---|---|---|
| M | 32 | Number of links per node per layer. Higher meant better recall and more memory. |
| efConstruction | 200 | Beam width during index build. Higher meant better graph quality and slower build. |
| efSearch | 64 | Beam width during query. Higher meant better recall and slower queries. |

All three were configurable via environment variables. No code change was needed to tune them per deployment.

---

## Measured Results

Benchmark: 300 random queries, k=10, on the 30k-row dataset.

| Backend | Mean latency | p95 latency | Recall@10 |
|---|---|---|---|
| Brute-force | 54.9ms | 61.7ms | 1.00 (exact) |
| FAISS HNSW | 0.9ms | 1.5ms | ~0.78 |

HNSW was about 60x faster per query. The trade-off was index build time: ~10s for HNSW vs ~1.5s for brute-force. That cost was paid once at startup.

### Why Recall Was Lower Than Expected

HNSW typically achieved over 95% recall. Here it was around 0.78. The reason was a data quality issue, not an algorithm problem. About 42% of products shared an identical feature vector. This happened because so many attributes were missing. Missing brand, colour, price, and weight all collapsed to the same imputed defaults. For those products, there were many valid tied nearest neighbors. Brute-force picked a specific tie-winner using rating and price. FAISS did not apply that tie-break, so it picked a different valid answer. Both answers were correct. Raising efSearch to 128 improved recall to around 0.80–0.82 at a small latency cost.

---

## Why the Default Was Still Brute-force

HNSW was approximate. For 30k products, brute-force was exact and already fast. There was no reason to trade correctness for speed at this scale. The FAISS backend was opt-in via the SIMILARITY_BACKEND environment variable. It was ready to use when the catalog actually grew large enough to need it.
