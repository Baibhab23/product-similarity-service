import logging
from contextlib import asynccontextmanager
from typing import Annotated, List
from fastapi import FastAPI, HTTPException, Query
from app.config import settings
from app.schemas import HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if settings.SIMILARITY_BACKEND == "faiss":
    from app import vector_index as backend
elif settings.SIMILARITY_BACKEND == "ivf":
    from app import ivf_index as backend
else:
    from app import similarity as backend

def _rrf(ranked_lists: List[List[str]], k: int = 60) -> List[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])]

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Building similarity index (backend=%s)...", settings.SIMILARITY_BACKEND)
    backend.get_index()
    if settings.IMAGE_ENABLED:
        from app.image_index import get_image_index
        get_image_index()
    logger.info("Index ready.")
    yield

app = FastAPI(
    title="Product Similarity Service",
    description="Find similar products by brand, price, weight, colour, rating, and visual appearance.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get(
    "/health",
    responses={503: {"description": "Index not ready"}},
)
def health() -> HealthResponse:
    try:
        idx = backend.get_index()
        return HealthResponse(
            status="ok",
            products_loaded=len(idx.df),
            backend=settings.SIMILARITY_BACKEND,
        )
    except Exception:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail="Index not ready")

@app.get(
    "/find_similar_products",
    responses={
        404: {"description": "Product not found"},
        500: {"description": "Internal error"},
    },
)
def get_similar_products(
    product_id: Annotated[str, Query(min_length=1, description="uniq_id of the source product")],
    num_similar: Annotated[int, Query(ge=1, le=settings.MAX_NUM_SIMILAR, description="How many similar products to return")],
) -> List[str]:
    try:
        fetch_n = num_similar * 3 if settings.IMAGE_ENABLED else num_similar

        structured_ids = backend.find_similar_products(product_id, fetch_n)

        if not settings.IMAGE_ENABLED:
            return structured_ids[:num_similar]

        from app.image_index import get_image_index
        image_ids = get_image_index().find_similar(product_id, fetch_n)

        return _rrf([structured_ids, image_ids], k=settings.RRF_K)[:num_similar]

    except backend.ProductNotFoundError:
        raise HTTPException(status_code=404, detail=f"product_id '{product_id}' not found")
    except Exception:
        logger.exception("find_similar_products failed for product_id=%s", product_id)
        raise HTTPException(status_code=500, detail="Internal error computing similar products")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
