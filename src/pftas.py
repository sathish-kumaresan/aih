from pathlib import Path

import mahotas
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PFTAS_DIM = 162
_FEATURE_COLS = [f"pftas_{i:03d}" for i in range(PFTAS_DIM)]


def pftas_rgb(image_path) -> np.ndarray:
    img = np.asarray(Image.open(image_path).convert("RGB"))
    return np.concatenate([mahotas.features.pftas(img[:, :, c]) for c in range(3)])


def _cache_is_valid(cache_path: Path, df: pd.DataFrame) -> bool:
    if not cache_path.exists():
        return False
    cached = pd.read_parquet(cache_path, columns=["path"])
    return set(cached["path"]) == set(df["path"])


def load_or_extract_features(df: pd.DataFrame, cache_path) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if _cache_is_valid(cache_path, df):
        return pd.read_parquet(cache_path)

    feats = np.zeros((len(df), PFTAS_DIM), dtype=np.float32)
    for i, p in enumerate(tqdm(df["path"], desc="PFTAS")):
        feats[i] = pftas_rgb(p)

    out = pd.concat(
        [df[["path", "label", "patient_id", "subtype", "split"]].reset_index(drop=True),
         pd.DataFrame(feats, columns=_FEATURE_COLS)],
        axis=1,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    return out


def feature_columns() -> list[str]:
    return list(_FEATURE_COLS)
