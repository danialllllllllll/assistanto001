# organise_project_safe.py
# -------------------------------------------------
#  SAFE ORGANISER – run as many times as you want
# -------------------------------------------------
import os
import shutil
import hashlib
from pathlib import Path
import difflib
import json

# ------------------------------------------------------------------
# 1. CONFIG – what goes where (add/remove folders as you like)
# ------------------------------------------------------------------
FOLDERS = {
    "learning": [".py"],                     # Python source
    "data":     [".npy", ".json", ".csv"],   # Data files
    "utils":    [".html", ".css", ".js", ".txt"]  # UI / misc
}

# ------------------------------------------------------------------
# 2. PROTECTED PATHS – never touch these (Replit internals, git, etc.)
# ------------------------------------------------------------------
PROTECTED = {
    ".git", ".cache", ".pythonlibs", "__pycache__",
    "backup_before_merge", "node_modules", ".replit"
}

# ------------------------------------------------------------------
# 3. BACKUP (only created the *first* time)
# ------------------------------------------------------------------
BACKUP = Path("backup_before_merge")
if not BACKUP.exists():
    print("[BACKUP] Creating initial backup …")
    shutil.copytree(".", BACKUP, dirs_exist_ok=True)
    print(f"[BACKUP] Done → {BACKUP}")
else:
    print("[BACKUP] Already exists – skipping creation")

# ------------------------------------------------------------------
# 4. HELPERS
# ------------------------------------------------------------------
def file_hash(p: Path) -> str:
    """Fast SHA-256 of a file (used for duplicate detection)."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def similar(a: Path, b: Path) -> bool:
    """True if two files have the same name *and* content."""
    if a.suffix != b.suffix:
        return False
    if a.name != b.name:
        return False
    return file_hash(a) == file_hash(b)

def is_protected(p: Path) -> bool:
    """Never move/delete anything in PROTECTED."""
    return any(part in PROTECTED for part in p.parts)

# ------------------------------------------------------------------
# 5. MAIN LOGIC
# ------------------------------------------------------------------
def main():
    root = Path(".")
    all_files = [p for p in root.rglob("*") if p.is_file() and not is_protected(p)]

    # ---- 5.1 Merge obvious duplicates (same name + same hash) ----
    merged = set()
    for i, f1 in enumerate(all_files):
        if f1 in merged:
            continue
        for f2 in all_files[i + 1 :]:
            if similar(f1, f2):
                print(f"[MERGE] {f2} → {f1}")
                with open(f1, "a", encoding="utf-8") as out:
                    out.write("\n# === MERGED FROM " + f2.name + " ===\n")
                    out.write(open(f2, "r", encoding="utf-8").read())
                f2.unlink()
                merged.add(f2)

    # ---- 5.2 Move files into the correct folder (skip if already there) ----
    moved = 0
    for target_folder, extensions in FOLDERS.items():
        dest = root / target_folder
        dest.mkdir(exist_ok=True)

        for f in all_files:
            if f.suffix in extensions and f.parent != dest and not is_protected(f):
                dest_path = dest / f.name
                if dest_path.exists():
                    # same name already exists – keep the one with the newer timestamp
                    if f.stat().st_mtime > dest_path.stat().st_mtime:
                        print(f"[REPLACE] {dest_path} ← {f}")
                        dest_path.unlink()
                        shutil.move(str(f), str(dest_path))
                        moved += 1
                    else:
                        print(f"[SKIP] {f} (newer version already in {dest})")
                        f.unlink()
                else:
                    print(f"[MOVE] {f} → {dest}")
                    shutil.move(str(f), str(dest_path))
                    moved += 1

    print(f"[DONE] Moved {moved} files")

    # ---- 5.3 Remove empty directories (except root & protected) ----
    for d in sorted(root.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()) and not is_protected(d) and d != root:
            print(f"[CLEAN] Removing empty folder {d}")
            d.rmdir()

if __name__ == "__main__":
    main()