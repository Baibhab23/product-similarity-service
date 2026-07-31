import json
import logging

import numpy as np
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

_IMAGE_DIM = 1280


def build_image_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Load pre-built EfficientNet-B0 embeddings from disk and return a matrix of shape (len(df), _IMAGE_DIM).
    """
    vecs_path = settings.IMAGE_VECTORS_PATH
    ids_path = settings.IMAGE_VECTORS_IDS_PATH

    try:
        stored_matrix = np.load(vecs_path)
        with open(ids_path) as f:
            stored_ids = json.load(f)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Image vectors not found ({e}). "
        ) from e

    id_to_row = {uid: i for i, uid in enumerate(stored_ids)}

    matrix = np.zeros((len(df), _IMAGE_DIM), dtype=np.float32)
    hit = 0
    for i, uid in enumerate(df["uniq_id"]):
        row = id_to_row.get(uid)
        if row is not None:
            matrix[i] = stored_matrix[row]
            hit += 1

    logger.info(
        "Loaded image matrix: %d/%d rows matched from %s",
        hit, len(df), vecs_path,
    )
    return matrix
