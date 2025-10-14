"""
Lightweight logger for model quality stats.
Writes JSON records tagged "QUALITY" to /var/log/yolo_train.log.

You can import both:
    from utils.logger_quality import log_quality, LOG_PATH
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

LOG_PATH = "/var/log/yolo_train.log"   # You can change this to a project file

def _git_short():
    """Return short git commit hash or 'NA' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "NA"

def log_quality(epoch, lr, train_loss, val_loss, val_iou):
    """
    Append one JSON record with training metrics.
    Example usage:
        log_quality(epoch, lr, train_loss, val_loss, val_iou)
    """
    rec = {
        "tag": "QUALITY",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "commit": _git_short(),
        "epoch": int(epoch),
        "lr": float(lr),
        "train": float(train_loss),
        "val": float(val_loss),
        "val_iou": float(val_iou),
    }
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")

