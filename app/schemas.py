from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    products_loaded: int
    backend: str
