import argparse
import io
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import requests
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_IMAGE_DIM = 576   
_BATCH_SIZE = 256
_DOWNLOAD_WORKERS = 64


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=_BATCH_SIZE)
    p.add_argument("--workers", type=int, default=_DOWNLOAD_WORKERS)
    return p.parse_args()


def _get_model():
    import torch
    import torchvision.models as models
    from torchvision import transforms

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def _first_url(raw) -> Optional[str]:
    if not raw or not isinstance(raw, str):
        return None
    first = raw.split("|")[0].strip()
    return first if first.startswith("http") else None


def _download_one(args):
    idx, url, timeout = args
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return idx, Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return idx, None


def _download_batch(urls: List[Optional[str]], timeout: float, workers: int) -> dict:
    tasks = [(i, url, timeout) for i, url in enumerate(urls) if url]
    images: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, img in pool.map(_download_one, tasks):
            images[idx] = img
    return images


def _infer_batch(images: dict, urls: List[Optional[str]], model, preprocess) -> List[Optional[np.ndarray]]:
    import torch

    tensors, positions = [], []
    for i, url in enumerate(urls):
        if not url:
            continue
        img = images.get(i)
        if img is not None:
            tensors.append(preprocess(img))
            positions.append(i)

    results: List[Optional[np.ndarray]] = [None] * len(urls)
    if tensors:
        with torch.no_grad():
            vecs = model(torch.stack(tensors)).numpy().astype(np.float32)
        for pos, vec in zip(positions, vecs):
            results[pos] = vec
    return results


def _get_or_reset_collection(client, name: str):
    import chromadb
    try:
        col = client.get_collection(name)
        sample = col.peek(limit=1)
        if sample["embeddings"] and len(sample["embeddings"][0]) != _IMAGE_DIM:
            logger.warning(
                "Collection dim=%d != expected %d — deleting and rebuilding.",
                len(sample["embeddings"][0]), _IMAGE_DIM,
            )
            client.delete_collection(name)
            col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    except Exception:
        col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    return col


def main():
    args = _parse_args()

    from app.config import settings
    from app.data_loader import load_products
    import chromadb

    os.makedirs(settings.CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    col = _get_or_reset_collection(client, settings.CHROMA_COLLECTION)

    df = load_products()
    ids = df["uniq_id"].tolist()
    urls = df["image_urls__small"].tolist() if "image_urls__small" in df.columns else [None] * len(df)

    existing = set(col.get(ids=ids, include=[])["ids"])
    pending_ids = [uid for uid in ids if uid not in existing]
    pending_urls = [urls[ids.index(uid)] for uid in pending_ids]

    logger.info(
        "%d total products, %d already in Chroma, %d to embed",
        len(ids), len(existing), len(pending_ids),
    )
    if not pending_ids:
        logger.info("Nothing to do.")
        return

    logger.info("Loading model (MobileNetV3-Small)...")
    model, preprocess = _get_model()

    bs = args.batch_size
    batches = [
        (pending_ids[s:s + bs], [_first_url(u) for u in pending_urls[s:s + bs]])
        for s in range(0, len(pending_ids), bs)
    ]

    t0 = time.time()
    ok = 0
    done = 0

    # Pipeline: submit next batch download while inference runs on current batch
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    next_future = prefetch_executor.submit(_download_batch, batches[0][1], args.timeout, args.workers)

    for i, (batch_ids, batch_urls) in enumerate(batches):
        images = next_future.result()

        if i + 1 < len(batches):
            next_future = prefetch_executor.submit(
                _download_batch, batches[i + 1][1], args.timeout, args.workers
            )

        vecs = _infer_batch(images, batch_urls, model, preprocess)

        upsert_ids, upsert_vecs = [], []
        for uid, vec in zip(batch_ids, vecs):
            embedding = vec if vec is not None else np.zeros(_IMAGE_DIM, dtype=np.float32)
            upsert_ids.append(uid)
            upsert_vecs.append(embedding.tolist())
            if vec is not None:
                ok += 1

        col.upsert(ids=upsert_ids, embeddings=upsert_vecs)
        done += len(batch_ids)

        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(pending_ids) - done) / rate if rate > 0 else float("inf")
        logger.info(
            "%d/%d upserted  ok=%d  rate=%.1f/s  eta=%.0fs",
            done, len(pending_ids), ok, rate, eta,
        )

    prefetch_executor.shutdown(wait=False)
    logger.info(
        "Done. %d/%d embeddings extracted, %d zero-filled. Collection size: %d",
        ok, len(pending_ids), len(pending_ids) - ok, col.count(),
    )


if __name__ == "__main__":
    main()
