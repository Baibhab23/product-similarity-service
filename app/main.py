import logging
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, Query
from app.config import settings
from app.schemas import HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Select the similarity backend once, at import time. 
# So the rest of the app doesn't need to know which one is active.
if settings.SIMILARITY_BACKEND == "faiss":
    from app import vector_index as backend
elif settings.SIMILARITY_BACKEND == "ivf":
    from app import ivf_index as backend
else:
    from app import similarity as backend

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Build the index eagerly so /health only returns ok once the service
    # can actually serve traffic, and the first real request isn't slow.
    logger.info("Building similarity index (backend=%s)...", settings.SIMILARITY_BACKEND)
    backend.get_index()
    logger.info("Index ready.")
    yield

app = FastAPI(
    title="Product Similarity Service",
    description="Find similar products by brand, price, weight, colour and rating.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
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

@app.get("/find_similar_products", response_model=List[str])
def get_similar_products(
    product_id: str = Query(..., min_length=1, description="uniq_id of the source product"),
    num_similar: int = Query(
        ..., ge=1, le=settings.MAX_NUM_SIMILAR, description="How many similar products to return"
    ),
) -> List[str]:
    try:
        return backend.find_similar_products(product_id, num_similar)
    except backend.ProductNotFoundError:
        raise HTTPException(status_code=404, detail=f"product_id '{product_id}' not found")
    except Exception:
        logger.exception("find_similar_products failed for product_id=%s", product_id)
        raise HTTPException(status_code=500, detail="Internal error computing similar products")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
