import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import (
    find_patient_leak,
    honest_split,
    list_images,
    provided_split,
    published_counts_400x,
)
from src.metrics import image_metrics, patient_metrics, summarize
from src.models import build
from src.pftas import load_or_extract_features
from src.utils import dump_yaml, load_yaml, run_dir, set_global_seed


def _write_metrics(path: Path, image_m: dict, patient_m: dict) -> None:
    payload = {"image": image_m, "patient": patient_m}
    path.write_text(json.dumps(payload, indent=2))


def _attach_n_per_patient(df_pred: pd.DataFrame) -> pd.DataFrame:
    counts = df_pred.groupby("patient_id")["path"].transform("count")
    df_pred = df_pred.copy()
    df_pred["n_images_for_patient"] = counts.astype(int)
    return df_pred


def _run_one_seed(tier_id: int, cfg, seed: int, splits: dict, val_df) -> None:
    set_global_seed(seed)
    selector = build(tier_id, cfg, seed)
    hyperparams = selector.select_hyperparams(
        train_df=splits["honest"][0], val_df=val_df, cfg=cfg, seed=seed,
    )
    print(f"[seed {seed}] hyperparams = {hyperparams}")

    for protocol in ["honest", "provided"]:
        train_df, test_df = splits[protocol]
        tier = build(tier_id, cfg, seed)
        tier.fit(train_df=train_df, val_df=val_df, cfg=cfg, seed=seed, hyperparams=hyperparams)
        margin, proba = tier.score(test_df)

        df_pred = pd.DataFrame({
            "path": test_df["path"].to_numpy(),
            "patient_id": test_df["patient_id"].to_numpy(),
            "label": test_df["label"].to_numpy(),
            "margin": margin,
            "proba": proba,
            "pred": (proba >= 0.5).astype(int),
            "seed": seed,
            "protocol": protocol,
        })
        df_pred = _attach_n_per_patient(df_pred)

        image_m = image_metrics(df_pred["label"], df_pred["margin"], df_pred["proba"])
        patient_m = patient_metrics(df_pred)

        rdir = run_dir(tier_id, protocol, seed)
        _write_metrics(rdir / "metrics.json", image_m, patient_m)
        df_pred.to_parquet(rdir / "predictions.parquet", index=False)
        (rdir / "best_params.json").write_text(json.dumps(hyperparams, indent=2))
        tier.save(rdir / tier.model_filename)
        dump_yaml(cfg, rdir / "config_resolved.yaml")

        print(f"[seed {seed}/{protocol}] "
              f"img AUROC {image_m['auroc']:.3f}, "
              f"pat AUROC {patient_m['auroc']:.3f}")


def _aggregate(tier_id: int, protocol: str, seeds: list[int], leak_count: int) -> None:
    image_dicts, patient_dicts = [], []
    for s in seeds:
        m = json.loads((run_dir(tier_id, protocol, s) / "metrics.json").read_text())
        image_dicts.append(m["image"])
        patient_dicts.append(m["patient"])

    lc = leak_count if protocol == "provided" else None
    image_df = summarize(image_dicts, leak_count=lc).assign(level="image")
    patient_df = summarize(patient_dicts, leak_count=lc).assign(level="patient")
    summary = pd.concat([image_df, patient_df], ignore_index=True)

    out_root = Path("results/reports")
    out_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_root / f"tier{tier_id}_{protocol}_summary.csv", index=False)

    cm_lines = []
    for s, d in zip(seeds, image_dicts):
        cm_lines.append(f"seed {s} image: {d['confusion_matrix']}")
    for s, d in zip(seeds, patient_dicts):
        cm_lines.append(f"seed {s} patient: {d['confusion_matrix']}")
    (out_root / f"tier{tier_id}_{protocol}_confusion.txt").write_text("\n".join(cm_lines) + "\n")


def _gpu_preamble(cfg) -> None:
    import torch
    import timm

    assert torch.cuda.is_available(), "Tier 2/3 requires CUDA; no silent CPU fallback."
    print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA: {torch.version.cuda}  torch: {torch.__version__}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    probe = timm.create_model(
        cfg.model.backbone, pretrained=True, num_classes=0, global_pool="",
    ).cuda().eval()
    img_size = int(cfg.input.img_size)
    with torch.no_grad():
        out = probe(torch.randn(1, 3, img_size, img_size).cuda())
    print(f"backbone output shape: {tuple(out.shape)}")
    assert out.dim() == 4, f"expected (B,C,H,W) spatial map, got {tuple(out.shape)} — timm may be returning pooled features"
    assert out.shape[1] == 1280, f"expected 1280 channels, got {out.shape[1]}"
    del probe, out
    torch.cuda.empty_cache()


def _carve_val(
    honest_train_df: pd.DataFrame,
    provided_train_df: pd.DataFrame,
    provided_test_df: pd.DataFrame,
    cfg,
):
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=float(cfg.val.fraction_of_honest_train),
        random_state=int(cfg.val.random_state),
    )
    hf_idx, val_idx = next(gss.split(
        honest_train_df, honest_train_df.label, groups=honest_train_df.patient_id,
    ))
    val_df = honest_train_df.iloc[val_idx].reset_index(drop=True)
    honest_train_df = honest_train_df.iloc[hf_idx].reset_index(drop=True)

    val_patient_ids = set(val_df.patient_id)
    overlap_train = len(val_patient_ids & set(provided_train_df.patient_id))
    overlap_test = len(val_patient_ids & set(provided_test_df.patient_id))
    print(f"val: {len(val_df)} images / {val_df.patient_id.nunique()} patients  "
          f"present in provided_train: {overlap_train}  in provided_test: {overlap_test}  "
          f"(early-stopping leak into provided protocol)")
    provided_train_df = provided_train_df[
        ~provided_train_df.patient_id.isin(val_patient_ids)
    ].reset_index(drop=True)
    val_leak = {
        "val_patients_in_provided_train_pre_removal": int(overlap_train),
        "val_patients_in_provided_test": int(overlap_test),
    }
    return val_df, honest_train_df, provided_train_df, val_leak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.config is None:
        args.config = f"configs/tier{args.tier}.yaml"

    cfg = load_yaml(args.config)

    if cfg.tier in (2, 3):
        _gpu_preamble(cfg)

    df, unparsed = list_images(cfg.data.root)
    published = published_counts_400x()
    print(f"on-disk images: {len(df)} (published {published['total']}, gap {published['total'] - len(df)})")
    print(f"on-disk patients: {df.patient_id.nunique()}")
    if unparsed:
        print(f"unparsed filenames: {len(unparsed)}")
        for u in unparsed:
            print(f"  {u}")

    leak = find_patient_leak(df)
    leak_root = Path(f"results/runs/tier{args.tier}")
    leak_root.mkdir(parents=True, exist_ok=True)
    print(f"leaked patients across provided train/test: {leak['shared_patient_count']}")

    if cfg.tier == 1:
        feat_df = load_or_extract_features(df, cfg.features.cache_path)
    else:
        feat_df = df

    splits = {
        "honest": honest_split(feat_df),
        "provided": provided_split(feat_df),
    }

    if cfg.tier in (2, 3):
        honest_train_df, honest_test_df = splits["honest"]
        provided_train_df, provided_test_df = splits["provided"]
        val_df, honest_train_df, provided_train_df, val_leak = _carve_val(
            honest_train_df, provided_train_df, provided_test_df, cfg,
        )
        leak.update(val_leak)
        splits = {
            "honest": (honest_train_df, honest_test_df),
            "provided": (provided_train_df, provided_test_df),
        }
    else:
        val_df = None

    (leak_root / "leak_status.json").write_text(json.dumps(leak, indent=2))

    for k, (tr, te) in splits.items():
        print(f"split {k}: train {len(tr)} images / {tr.patient_id.nunique()} patients, "
              f"test {len(te)} images / {te.patient_id.nunique()} patients")

    seeds = [args.seed] if args.seed is not None else list(cfg.seeds)
    for s in seeds:
        _run_one_seed(args.tier, cfg, s, splits, val_df)

    if len(seeds) > 1:
        for protocol in ["honest", "provided"]:
            _aggregate(args.tier, protocol, seeds, leak["shared_patient_count"])
            print(f"wrote results/reports/tier{args.tier}_{protocol}_summary.csv")


if __name__ == "__main__":
    main()
