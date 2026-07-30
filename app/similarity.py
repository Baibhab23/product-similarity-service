import logging
import threading
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.data_loader import load_products
from app.features import build_feature_matrix

logger = logging.getLogger(__name__)

class ProductNotFoundError(KeyError):
    """Raised when a product_id does not exist in the dataset."""

class ProductSimilarityIndex:
    """
    Holds the cleaned product DataFrame plus the fitted feature matrix
    used for similarity lookups. Built once at process startup.
    """
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df: pd.DataFrame = df if df is not None else load_products()
        self._id_to_row: dict[str, int] = {
            pid: i for i, pid in enumerate(self.df["uniq_id"])
        }
        bundle = build_feature_matrix(self.df)
        self.feature_matrix: np.ndarray = bundle

    def row_of(self, product_id: str) -> int:
        try:
            return self._id_to_row[product_id]
        except KeyError:
            raise ProductNotFoundError(product_id) from None

    def find_similar(self, product_id: str, num_similar: int) -> List[str]:
        """
        Return up to `num_similar` product ids most similar to `product_id`,
        sorted by similarity descending, excluding the product itself.
        Ties are broken by rating (higher first), then sales_price (lower
        first) — an arbitrary but deterministic choice per the exercise note.
        """
        row = self.row_of(product_id)
        query_vec = self.feature_matrix[row : row + 1]
        scores = cosine_similarity(query_vec, self.feature_matrix).ravel()

        order = np.lexsort(
            (
                self.df["sales_price"].fillna(np.inf).values,  # 3rd key, ascending
                -self.df["rating"].fillna(-1).values,           # 2nd key, ascending (= rating desc)
                -scores,                                         # 1st key, ascending (= score desc)
            )
        )
        result_ids = []
        for idx in order:
            if idx == row:
                continue
            result_ids.append(self.df["uniq_id"].iloc[idx])
            if len(result_ids) == num_similar:
                break
        return result_ids

# --- Module-level singleton + functional API (matches the exercise's spec) ---

_index: Optional[ProductSimilarityIndex] = None
_index_lock = threading.Lock()

def get_index() -> ProductSimilarityIndex:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:  # re-check inside the lock
                _index = ProductSimilarityIndex()
    return _index

def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    # Public API required by the exercise spec.
    if num_similar <= 0:
        return []
    return get_index().find_similar(product_id, num_similar)
