#!/usr/bin/env python3
"""
After training (when log_quality() has run):

python -m scripts.append_latest_quality
git add README.md
git commit -m "Append latest model quality to README"

"""

import json, sys, re
from pathlib import Path
from datetime import datetime
from utils.logger_quality import LOG_PATH   # 🔹 imported from your logger module

README = "README.md"

def last_quality(path):
    """Return the most recent QUALITY record from the log."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in reversed(f.readlines()):
                i = line.find("{")
                if i >= 0:
                    try:
                        obj = json.loads(line[i:].strip())
                    except Exception:
                        continue
                    if obj.get("tag") == "QUALITY":
                        return obj
    except FileNotFoundError:
        pass
    return None

def ensure_progress_header(text):
    """Add a Progress section if missing."""
    if re.search(r"(?im)^\s*##\s*Progress\s*$", text):
        return text
    return text.rstrip() + "\n\n## Progress\n\n"

def main():
    rec = last_quality(LOG_PATH)
    if not rec:
        print(f"No QUALITY record found in {LOG_PATH}")
        sys.exit(0)

    readme = Path(README)
    if not readme.exists():
        readme.write_text("# README\n\n## Progress\n\n", encoding="utf-8")
    txt = readme.read_text(encoding="utf-8")

    # Skip if already logged for this commit
    if rec.get("commit", "NA") in txt:
        print("README already has this commit; nothing to do.")
        return

    txt = ensure_progress_header(txt)
    dt = datetime.strptime(rec["date"], "%Y-%m-%d").strftime("%m/%d/%Y")
    line = (f"{dt} Commit {rec.get('commit','NA')} "
            f"Epoch {int(rec['epoch']):03d} | lr {rec['lr']:.6f} | "
            f"train {rec['train']:.4f} | val {rec['val']:.4f} | "
            f"val_iou {rec['val_iou']:.4f}")

    readme.write_text(txt.rstrip() + "\n" + line + "\n", encoding="utf-8")
    print(f"Appended: {line}")

if __name__ == "__main__":
    main()

