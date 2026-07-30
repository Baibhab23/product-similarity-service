import os
class Settings:
    # File paths
    DATA_PATH = os.getenv(
        "DATA_PATH",
        "data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson",
    )
    DATA_ZIP_PATH = os.getenv("DATA_ZIP_PATH", "data/archive.zip")

    # Feature attributes for similarity calculation
    NUMERIC_ATTRS = ["sales_price", "weight_grams", "rating"]
    CATEGORICAL_ATTRS = ["brand", "colour"]

    # Cap high-cardinality categories to keep one-hot encoding compact
    MAX_CATEGORY_VALUES = int(os.getenv("MAX_CATEGORY_VALUES", "200"))

    # Backend choice: "brute" (scikit-learn cosine), "faiss" (HNSW), or "ivf"
    SIMILARITY_BACKEND = os.getenv("SIMILARITY_BACKEND", "brute")

    # FAISS HNSW index tuning
    FAISS_HNSW_M = int(os.getenv("FAISS_HNSW_M", "32"))
    FAISS_HNSW_EF_CONSTRUCTION = int(os.getenv("FAISS_HNSW_EF_CONSTRUCTION", "200"))
    FAISS_HNSW_EF_SEARCH = int(os.getenv("FAISS_HNSW_EF_SEARCH", "64"))

    # FAISS IVF index tuning
    FAISS_IVF_NLIST = int(os.getenv("FAISS_IVF_NLIST", "100"))   
    FAISS_IVF_NPROBE = int(os.getenv("FAISS_IVF_NPROBE", "10"))  

    MAX_NUM_SIMILAR = int(os.getenv("MAX_NUM_SIMILAR", "100"))

settings = Settings()