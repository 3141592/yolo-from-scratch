"""
Append latest or best model quality stats to README.md

Priority order:
1. Prefer logs/<sha>/best.json sidecar (if present and valid)
2. Else fall back to scanning the rolling log for the best val_iou
3. If nothing found, exit gracefully

Keeps your README updated with lines like:
  10/18/2025 Commit f06afb5 Epoch 011 | lr 0.010000 | train 0.0487 | val 0.0253 | val_iou 0.6122
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# -------- CONFIG --------
README = "README.md"
LOG_PATH = "/var/log/yolo_train.log"
SIDE_DIR = Path("logs")  # matches logger_quality default
# -------------------------

def git_short():
    """Return short git SHA or 'NA'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "NA"

def ensure_progress_header(txt: str) -> str:
    if "## Progress" not in txt:
        txt += "\n## Progress\n\n"
    return txt

def load_sidecar(sha: str):
    """Return dict from logs/<sha>/best.json or None."""
    path = SIDE_DIR / sha / "best.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def all_quality_rows(log_path):
    """Return list of QUALITY/QUALITY_BEST dicts from log."""
    rows = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "{" not in line:
                    continue
                try:
                    obj = json.loads(line[line.find("{"):])
                except Exception:
                    continue
                if obj.get("tag", "").startswith("QUALITY"):
                    rows.append(obj)
    except FileNotFoundError:
        pass
    return rows

def pick_best(rows):
    """Pick best record by val_iou, break ties by lower val loss and higher epoch."""
    if not rows:
        return None
    return max(rows, key=lambda r: (
        float(r.get("val_iou", -1)),
        -float(r.get("val", float("inf"))),
        int(r.get("epoch", -1)),
    ))

def upsert_readme_line(readme_text: str, commit: str, new_line: str) -> str:
    """Replace existing commit line or append if not present."""
    pattern = re.compile(rf"Commit {re.escape(commit)} .*")
    if pattern.search(readme_text):
        return pattern.sub(new_line, readme_text)
    else:
        return readme_text.rstrip() + "\n" + new_line + "\n"

def main():
    sha = git_short()
    best = load_sidecar(sha)

    # Fallback: scan rolling log
    if best is None:
        rows = all_quality_rows(LOG_PATH)
        best = pick_best([r for r in rows if r.get("commit") == sha] or rows)

    if not best:
        print("No QUALITY or sidecar record found.")
        sys.exit(0)

    dt = datetime.strptime(best["date"], "%Y-%m-%d").strftime("%m/%d/%Y")
    line = (
        f"{dt} Commit {best.get('commit','NA')} "
        f"Epoch {int(best['epoch']):03d} | "
        f"lr {float(best['lr']):.6f} | "
        f"train {float(best['train']):.4f} | "
        f"val {float(best['val']):.4f} | "
        f"val_iou {float(best['val_iou']):.4f}"
    )

    readme = Path(README)
    if not readme.exists():
        readme.write_text("# README\n\n## Progress\n\n", encoding="utf-8")
    txt = readme.read_text(encoding="utf-8")

    new_txt = upsert_readme_line(ensure_progress_header(txt), sha, line)
    readme.write_text(new_txt, encoding="utf-8")
    print(f"Appended or updated: {line}")

if __name__ == "__main__":
    main()

