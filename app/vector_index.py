import logging
import threading
from typing import List, Optional
import numpy as np
import pandas as pd
from app.config import settings
from app.data_loader import load_products
from app.features import build_feature_matrix

logger = logging.getLogger(__name__)
class ProductNotFoundError(KeyError):
    """Raised when a product_id does not exist in the dataset."""
    
def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid div-by-zero for an all-zero vector
    return (matrix / norms).astype(np.float32)
class FaissSimilarityIndex:
    """
    HNSW-backed approximate similarity search, using the same feature
    vectors as the brute-force backend so results are
    directly comparable.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        try:
            import faiss  # local import: optional dependency
        except ImportError as e:
            raise RuntimeError(
                "faiss-cpu is not installed. `pip install faiss-cpu` or use "
                "SIMILARITY_BACKEND=brute."
            ) from e
        self._faiss = faiss
        self.df: pd.DataFrame = df if df is not None else load_products()
        self._id_to_row: dict[str, int] = {
            pid: i for i, pid in enumerate(self.df["uniq_id"])
        }
        bundle = build_feature_matrix(self.df)
        self._normalized = _l2_normalize(bundle)
        dims = self._normalized.shape[1]

        index = faiss.IndexHNSWFlat(dims, settings.FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = settings.FAISS_HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = settings.FAISS_HNSW_EF_SEARCH
        index.add(self._normalized)
        self.index = index
        logger.info(
            "Built FAISS HNSW index: %d vectors x %d dims (M=%d, efConstruction=%d)",
            self._normalized.shape[0], dims,
            settings.FAISS_HNSW_M, settings.FAISS_HNSW_EF_CONSTRUCTION,
        )
    def row_of(self, product_id: str) -> int:
        try:
            return self._id_to_row[product_id]
        except KeyError:
            raise ProductNotFoundError(product_id) from None

    def find_similar(self, product_id: str, num_similar: int) -> List[str]:
        row = self.row_of(product_id)
        query = self._normalized[row : row + 1]
        # Search for one extra since the product itself will be its own
        # nearest neighbour (score 1.0) and needs to be filtered out.
        k = num_similar + 1
        _scores, indices = self.index.search(query, k)

        result_ids = []
        for idx in indices[0]:
            if idx == -1 or idx == row:
                continue
            result_ids.append(self.df["uniq_id"].iloc[idx])
            if len(result_ids) == num_similar:
                break
        return result_ids

_index: Optional[FaissSimilarityIndex] = None
_index_lock = threading.Lock()

def get_index() -> FaissSimilarityIndex:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = FaissSimilarityIndex()
    return _index

def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    if num_similar <= 0:
        return []
    return get_index().find_similar(product_id, num_similar)
