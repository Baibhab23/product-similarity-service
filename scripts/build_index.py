import argparse
import random
import time

from app.data_loader import load_products
from app.similarity import ProductSimilarityIndex
from app.vector_index import FaissSimilarityIndex


def recall_at_k(ground_truth: list[str], candidate: list[str], k: int) -> float:
    gt_set = set(ground_truth[:k])
    cand_set = set(candidate[:k])
    if not gt_set:
        return 1.0
    return len(gt_set & cand_set) / len(gt_set)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_products()
    print(f"{len(df)} products loaded.\n")

    print("Building brute-force index...")
    t0 = time.time()
    brute = ProductSimilarityIndex(df)
    print(f"  build time: {time.time() - t0:.2f}s")

    print("Building FAISS HNSW index...")
    t0 = time.time()
    faiss_idx = FaissSimilarityIndex(df)
    print(f"  build time: {time.time() - t0:.2f}s\n")

    random.seed(42)
    sample_ids = random.sample(list(df["uniq_id"]), min(args.num_queries, len(df)))

    brute_times, faiss_times, recalls = [], [], []
    for pid in sample_ids:
        t0 = time.time()
        gt = brute.find_similar(pid, args.k)
        brute_times.append(time.time() - t0)

        t0 = time.time()
        cand = faiss_idx.find_similar(pid, args.k)
        faiss_times.append(time.time() - t0)

        recalls.append(recall_at_k(gt, cand, args.k))

    def summarize(name: str, times: list[float]) -> None:
        times_ms = sorted(t * 1000 for t in times)
        p50 = times_ms[len(times_ms) // 2]
        p95 = times_ms[int(len(times_ms) * 0.95)]
        print(f"{name:>12}: mean={sum(times_ms)/len(times_ms):6.3f}ms  p50={p50:6.3f}ms  p95={p95:6.3f}ms")

    print(f"--- {len(sample_ids)} queries, k={args.k} ---")
    summarize("brute-force", brute_times)
    summarize("faiss-hnsw", faiss_times)
    print(f"\nmean recall@{args.k} (faiss vs brute-force ground truth): {sum(recalls)/len(recalls):.4f}")

if __name__ == "__main__":
    main()
