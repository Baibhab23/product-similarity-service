import logging
import threading
from typing import List

from app.config import settings

logger = logging.getLogger(__name__)


class ChromaImageIndex:
    def __init__(self):
        try:
            import chromadb
        except ImportError as e:
            raise RuntimeError(
                "chromadb is not installed. `pip install chromadb` or set IMAGE_ENABLED=false."
            ) from e

        client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self._col = client.get_collection(settings.CHROMA_COLLECTION)
        logger.info(
            "Opened Chroma collection '%s' (%d vectors) at %s",
            settings.CHROMA_COLLECTION,
            self._col.count(),
            settings.CHROMA_PATH,
        )

    def find_similar(self, product_id: str, num_similar: int) -> List[str]:
        # Fetch the query product's embedding, then query the collection.
        result = self._col.get(ids=[product_id], include=["embeddings"])
        embeddings = result["embeddings"]
        if embeddings is None or len(embeddings) == 0:
            return []
        query_vec = result["embeddings"][0]
        # Request one extra to filter self out in case it comes back
        hits = self._col.query(
            query_embeddings=[query_vec],
            n_results=num_similar + 1,
            include=[],  
        )
        return [
            pid for pid in hits["ids"][0]
            if pid != product_id
        ][:num_similar]

_index: "ChromaImageIndex | None" = None
_index_lock = threading.Lock()

def get_image_index() -> ChromaImageIndex:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = ChromaImageIndex()
    return _index
