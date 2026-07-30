import argparse
import random
import time

from app.data_loader import load_products
from app.similarity import ProductSimilarityIndex
from app.vector_index import FaissSimilarityIndex
from app.ivf_index import IVFSimilarityIndex


def recall_at_k(ground_truth: list[str], candidate: list[str], k: int) -> float:
    gt_set = set(ground_truth[:k])
    cand_set = set(candidate[:k])
    if not gt_set:
        return 1.0
    return len(gt_set & cand_set) / len(gt_set)

def benchmark(name: str, index, sample_ids: list, brute, k: int):
    times, recalls = [], []
    for pid in sample_ids:
        gt = brute.find_similar(pid, k)
        t0 = time.time()
        cand = index.find_similar(pid, k)
        times.append(time.time() - t0)
        recalls.append(recall_at_k(gt, cand, k))

    times_ms = sorted(t * 1000 for t in times)
    mean = sum(times_ms) / len(times_ms)
    p50 = times_ms[len(times_ms) // 2]
    p95 = times_ms[int(len(times_ms) * 0.95)]
    mean_recall = sum(recalls) / len(recalls)
    print(f"{name:>12}: mean={mean:6.3f}ms  p50={p50:6.3f}ms  p95={p95:6.3f}ms  recall@{k}={mean_recall:.4f}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-queries", type=int, default=300)
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
    print(f"  build time: {time.time() - t0:.2f}s")

    print("Building FAISS IVF index...")
    t0 = time.time()
    ivf_idx = IVFSimilarityIndex(df)
    print(f"  build time: {time.time() - t0:.2f}s\n")

    random.seed(42)
    sample_ids = random.sample(list(df["uniq_id"]), min(args.num_queries, len(df)))

    # Pre-compute brute-force times separately for fair comparison
    brute_times = []
    for pid in sample_ids:
        t0 = time.time()
        brute.find_similar(pid, args.k)
        brute_times.append(time.time() - t0)

    times_ms = sorted(t * 1000 for t in brute_times)
    mean = sum(times_ms) / len(times_ms)
    p50 = times_ms[len(times_ms) // 2]
    p95 = times_ms[int(len(times_ms) * 0.95)]
    print(f"--- {len(sample_ids)} queries, k={args.k} ---")
    print(f"{'brute-force':>12}: mean={mean:6.3f}ms  p50={p50:6.3f}ms  p95={p95:6.3f}ms  recall@{args.k}=1.0000 (ground truth)")

    benchmark("faiss-hnsw", faiss_idx, sample_ids, brute, args.k)
    benchmark("faiss-ivf", ivf_idx, sample_ids, brute, args.k)

if __name__ == "__main__":
    main()
