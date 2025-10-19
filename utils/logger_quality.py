"""
Lightweight logger for model quality stats.

- Always appends JSON records (tag "QUALITY" or "QUALITY_BEST") to LOG_PATH.
- When called with best=True, also mirrors the record to a per-SHA sidecar:
    logs/<git_short_sha>/best.json   (created/updated only if strictly better)

Usage:
    from utils.logger_quality import log_quality

    # per-epoch (unchanged)
    log_quality(epoch, lr, train_loss, val_loss, val_iou)

    # when you save a new best model:
    log_quality(epoch, lr, train_loss, val_loss, val_iou, best=True)
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

# You can change LOG_PATH to a project-local file if you prefer
LOG_PATH = "/var/log/yolo_train.log"

# Base directory for sidecars; override with env QUALITY_SIDE_DIR if desired
SIDE_DIR = Path(os.environ.get("QUALITY_SIDE_DIR", "logs"))

def _git_short():
    """Return short git commit hash or 'NA' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "NA"

def _sidecar_path(sha: str) -> Path:
    return SIDE_DIR / sha / "best.json"

def _load_sidecar(sha: str):
    p = _sidecar_path(sha)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _is_better(new: dict, old: dict | None) -> bool:
    """Return True if 'new' beats 'old' (or no old exists).

    Primary metric: higher val_iou.
    Tie-breakers: lower val loss, then higher epoch (i.e., latest).
    """
    if old is None:
        return True
    new_key = (
        float(new.get("val_iou", -1.0)),
        -float(new.get("val", float("inf"))),
        int(new.get("epoch", -1)),
    )
    old_key = (
        float(old.get("val_iou", -1.0)),
        -float(old.get("val", float("inf"))),
        int(old.get("epoch", -1)),
    )
    return new_key > old_key

def _write_sidecar(sha: str, rec: dict):
    p = _sidecar_path(sha)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")

def log_quality(epoch, lr, train_loss, val_loss, val_iou, *, best: bool = False):
    """
    Append one JSON record with training metrics.

    Parameters
    ----------
    epoch : int
    lr : float
    train_loss : float
    val_loss : float
    val_iou : float
    best : bool (kw-only, default False)
        If True, tag the record as QUALITY_BEST and update logs/<sha>/best.json
        iff this is strictly better than the current sidecar.
    """
    sha = _git_short()
    tag = "QUALITY_BEST" if best else "QUALITY"
    rec = {
        "tag": tag,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "commit": sha,
        "epoch": int(epoch),
        "lr": float(lr),
        "train": float(train_loss),
        "val": float(val_loss),
        "val_iou": float(val_iou),
    }

    # 1) Always append to the rolling log
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    # 2) If flagged as best, upsert the per-SHA sidecar only if improved
    if best and sha != "NA":
        current = _load_sidecar(sha)
        if _is_better(rec, current):
            _write_sidecar(sha, rec)

