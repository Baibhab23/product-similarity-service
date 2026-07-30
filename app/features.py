import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import settings

logger = logging.getLogger(__name__)

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric = df[settings.NUMERIC_ATTRS].copy()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric_scaled = StandardScaler().fit_transform(numeric.values)

    categorical_encoded = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(df[settings.CATEGORICAL_ATTRS])
    matrix = np.hstack([numeric_scaled, categorical_encoded]).astype(np.float32)
    logger.info(
        "Built feature matrix: %d products x %d dims (%d numeric + %d categorical)",
        matrix.shape[0], matrix.shape[1],
        numeric_scaled.shape[1], categorical_encoded.shape[1],
    )
    return matrix
