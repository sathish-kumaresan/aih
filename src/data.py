import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

_NAME_RE = re.compile(
    r"^SOB_([BM])_([A-Z0-9]+)-(\d{2})-([0-9A-Z]+)-(\d+)-(\d+)\.png$"
)


def published_counts_400x() -> dict:
    return {"benign": 588, "malignant": 1232, "total": 1820}


def list_images(root) -> tuple[pd.DataFrame, list[str]]:
    root = Path(root)
    rows, unparsed = [], []
    for p in sorted(root.rglob("*.png")):
        if ":Zone.Identifier" in p.name:
            continue
        m = _NAME_RE.match(p.name)
        if not m:
            unparsed.append(str(p))
            continue
        bm, subtype, year, patient_id, mag, seq = m.groups()
        rows.append({
            "path": str(p),
            "label": 1 if bm == "M" else 0,
            "subtype": subtype,
            "patient_id": patient_id,
            "year": year,
            "seq": seq,
            "split": p.parent.parent.name,
        })
    return pd.DataFrame(rows), unparsed


def find_patient_leak(df: pd.DataFrame) -> dict:
    train_p = set(df.loc[df.split == "train", "patient_id"])
    test_p = set(df.loc[df.split == "test", "patient_id"])
    shared = sorted(train_p & test_p)
    return {
        "shared_patient_count": len(shared),
        "shared_patient_ids": shared,
        "train_images_in_leak": int(df[(df.split == "train") & df.patient_id.isin(shared)].shape[0]),
        "test_images_in_leak": int(df[(df.split == "test") & df.patient_id.isin(shared)].shape[0]),
    }


def provided_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)


def honest_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    train_idx, test_idx = next(gss.split(df, df.label, groups=df.patient_id))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def grouped_cv(seed: int, n_splits: int = 5) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
