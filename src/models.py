from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from src.data import grouped_cv
from src.pftas import feature_columns


class Tier(ABC):
    model_filename: str = "model.bin"

    @abstractmethod
    def select_hyperparams(self, train_df, val_df, cfg, seed) -> dict: ...
    @abstractmethod
    def fit(self, train_df, val_df, cfg, seed, hyperparams: dict) -> None: ...
    @abstractmethod
    def score(self, df) -> tuple[np.ndarray, np.ndarray]: ...
    @abstractmethod
    def save(self, path) -> None: ...
    @abstractmethod
    def load(self, path) -> None: ...


def _Xy(df: pd.DataFrame):
    return df[feature_columns()].to_numpy(dtype=np.float32), df["label"].to_numpy()


def _pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=False, random_state=seed)),
    ])


class Tier1(Tier):
    model_filename = "model.joblib"

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.hyperparams: dict | None = None

    def select_hyperparams(self, train_df, val_df, cfg, seed) -> dict:
        X, y = _Xy(train_df)
        gs = GridSearchCV(
            estimator=_pipeline(seed),
            param_grid={
                "svm__C": list(cfg.model.c_grid),
                "svm__gamma": list(cfg.model.gamma_grid),
            },
            cv=grouped_cv(seed, n_splits=cfg.cv.n_splits),
            scoring=cfg.cv.scoring,
            n_jobs=-1,
            refit=False,
        )
        gs.fit(X, y, groups=train_df["patient_id"].to_numpy())
        return {
            "C": float(gs.best_params_["svm__C"]),
            "gamma": gs.best_params_["svm__gamma"],
        }

    def fit(self, train_df, val_df, cfg, seed, hyperparams: dict) -> None:
        X, y = _Xy(train_df)
        self.pipeline = _pipeline(seed).set_params(
            svm__C=hyperparams["C"], svm__gamma=hyperparams["gamma"],
        )
        self.pipeline.fit(X, y)
        self.hyperparams = hyperparams

    def score(self, df) -> tuple[np.ndarray, np.ndarray]:
        X, _ = _Xy(df)
        margin = self.pipeline.decision_function(X)
        proba = 1.0 / (1.0 + np.exp(-margin))
        return margin, proba

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "hyperparams": self.hyperparams}, path)

    def load(self, path) -> None:
        obj = joblib.load(path)
        self.pipeline = obj["pipeline"]
        self.hyperparams = obj["hyperparams"]


class EffV2SBinary(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(backbone.num_features, 1)

    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.pool(feats).flatten(1)
        return self.head(pooled).squeeze(-1)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        avg = self.mlp(x.mean(dim=(2, 3)))
        mx = self.mlp(x.amax(dim=(2, 3)))
        weight = torch.sigmoid(avg + mx)
        return x * weight.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        weight = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * weight


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial(self.channel(x))


class EffV2SCBAM(nn.Module):
    def __init__(self, backbone, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.backbone = backbone
        self.cbam = CBAM(backbone.num_features, reduction, kernel_size)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(backbone.num_features, 1)

    def forward(self, x):
        feats = self.cbam(self.backbone(x))
        pooled = self.pool(feats).flatten(1)
        return self.head(pooled).squeeze(-1)


def _cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    warmup_steps = max(1, warmup_steps)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
    )


def _eval_auroc(model, loader, device, amp_dtype) -> float:
    model.eval()
    margins, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x).float()
            margins.append(out.cpu().numpy())
            labels.append(y.numpy())
    margin = np.concatenate(margins)
    label = np.concatenate(labels).astype(int)
    if len(set(label.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(label, margin))


class Tier2(Tier):
    model_filename = "model.pt"

    def __init__(self):
        self.model: nn.Module | None = None
        self.best_epoch: int | None = None
        self.best_val_auroc: float | None = None
        self.hyperparams: dict = {}
        self._img_size: int = 384
        self._model_kwargs_used: dict = {}

    def _eval_batch_size(self) -> int:
        return 32 if self._img_size >= 384 else 64

    def _build_model(self, backbone, **kwargs) -> nn.Module:
        return EffV2SBinary(backbone)

    def _model_kwargs(self, cfg) -> dict:
        return {}

    def _param_groups(self, model, cfg) -> list[dict]:
        return [
            {"params": model.backbone.parameters(), "lr": float(cfg.optim.lr_backbone)},
            {"params": list(model.pool.parameters()) + list(model.head.parameters()),
             "lr": float(cfg.optim.lr_head)},
        ]

    def select_hyperparams(self, train_df, val_df, cfg, seed) -> dict:
        return {}

    def fit(self, train_df, val_df, cfg, seed, hyperparams: dict) -> None:
        import timm
        from src.dataset import BreaKHisDataset, _seed_worker, eval_transform, train_transform

        device = torch.device("cuda")
        self._img_size = int(cfg.input.img_size)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_scaler = (amp_dtype == torch.float16)
        scaler = torch.amp.GradScaler("cuda") if use_scaler else None

        backbone = timm.create_model(
            cfg.model.backbone, pretrained=True, num_classes=0, global_pool="",
        )
        self._model_kwargs_used = self._model_kwargs(cfg)
        model = self._build_model(backbone, **self._model_kwargs_used).to(device)

        cj = cfg.augment.color_jitter
        train_tx = train_transform(
            img_size=self._img_size,
            hflip=cfg.augment.hflip,
            vflip=cfg.augment.vflip,
            rotation_deg=cfg.augment.rotation_deg,
            brightness=cj.brightness, contrast=cj.contrast,
            saturation=cj.saturation, hue=cj.hue,
        )
        eval_tx = eval_transform(self._img_size)

        g_train = torch.Generator(); g_train.manual_seed(seed)
        train_loader = DataLoader(
            BreaKHisDataset(train_df, train_tx),
            batch_size=int(cfg.train.batch_size), shuffle=True,
            num_workers=int(cfg.train.num_workers), pin_memory=True, drop_last=False,
            worker_init_fn=_seed_worker, generator=g_train,
        )
        g_val = torch.Generator(); g_val.manual_seed(seed)
        val_loader = DataLoader(
            BreaKHisDataset(val_df, eval_tx),
            batch_size=int(cfg.train.batch_size), shuffle=False,
            num_workers=int(cfg.train.num_workers), pin_memory=True,
            worker_init_fn=_seed_worker, generator=g_val,
        )

        optimizer = torch.optim.AdamW(
            self._param_groups(model, cfg),
            weight_decay=float(cfg.optim.weight_decay),
        )
        steps_per_epoch = max(1, len(train_loader))
        scheduler = _cosine_with_warmup(
            optimizer,
            total_steps=steps_per_epoch * int(cfg.train.epochs),
            warmup_steps=steps_per_epoch * int(cfg.schedule.warmup_epochs),
        )
        n_pos = int((train_df["label"] == 1).sum())
        n_neg = int((train_df["label"] == 0).sum())
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_auroc = -1.0
        best_epoch = -1
        patience = 0
        early_stop = int(cfg.train.early_stop_patience)
        for epoch in range(int(cfg.train.epochs)):
            model.train()
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            val_auroc = _eval_auroc(model, val_loader, device, amp_dtype)
            improved = val_auroc > best_auroc
            tag = "best" if improved else f"patience {patience + 1}/{early_stop}"
            print(f"  epoch {epoch}: val_auroc={val_auroc:.4f}  ({tag})")
            if improved:
                best_auroc = val_auroc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= early_stop:
                    print(f"  early-stop at epoch {epoch}")
                    break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = int(cfg.train.epochs) - 1
            best_auroc = float("nan")
        model.load_state_dict(best_state)
        self.model = model
        self.best_epoch = best_epoch
        self.best_val_auroc = float(best_auroc)
        self.hyperparams = {"best_epoch": best_epoch, "best_val_auroc": float(best_auroc)}

    def score(self, df) -> tuple[np.ndarray, np.ndarray]:
        from src.dataset import BreaKHisDataset, eval_transform

        device = torch.device("cuda")
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        loader = DataLoader(
            BreaKHisDataset(df, eval_transform(self._img_size)),
            batch_size=self._eval_batch_size(),
            shuffle=False, num_workers=4, pin_memory=True,
        )
        self.model.eval()
        margins = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    out = self.model(x).float()
                margins.append(out.cpu().numpy())
        margin = np.concatenate(margins)
        proba = 1.0 / (1.0 + np.exp(-margin))
        return margin, proba

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "hyperparams": self.hyperparams,
            "img_size": self._img_size,
            "model_kwargs": self._model_kwargs_used,
        }, path)

    def load(self, path) -> None:
        import timm
        obj = torch.load(path, map_location="cuda")
        backbone = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False,
            num_classes=0, global_pool="",
        )
        self._model_kwargs_used = obj.get("model_kwargs", {})
        self.model = self._build_model(backbone, **self._model_kwargs_used).cuda()
        self.model.load_state_dict(obj["state_dict"])
        self.hyperparams = obj.get("hyperparams", {})
        self.best_epoch = self.hyperparams.get("best_epoch")
        self.best_val_auroc = self.hyperparams.get("best_val_auroc")
        self._img_size = int(obj.get("img_size", 384))


class Tier3(Tier2):
    def _build_model(self, backbone, **kwargs) -> nn.Module:
        return EffV2SCBAM(backbone, **kwargs)

    def _model_kwargs(self, cfg) -> dict:
        return {
            "reduction": int(cfg.cbam.reduction),
            "kernel_size": int(cfg.cbam.kernel_size),
        }

    def _param_groups(self, model, cfg) -> list[dict]:
        return [
            {"params": model.backbone.parameters(), "lr": float(cfg.optim.lr_backbone)},
            {"params": (list(model.cbam.parameters()) + list(model.pool.parameters())
                        + list(model.head.parameters())),
             "lr": float(cfg.optim.lr_head)},
        ]


def build(tier_id: int, cfg, seed: int) -> Tier:
    return {1: Tier1, 2: Tier2, 3: Tier3}[tier_id]()
