from pathlib import Path
import os

def clean_pkls(base_dir: Path, dry_run: bool = True):
    """
    Delete all numbered/extra .pkl files in subfolders of base_dir,
    but keep the canonical 'tree_id.pkl' file.

    Parameters
    ----------
    base_dir : Path
        The root data directory containing tree subfolders.
    dry_run : bool
        If True, only print what would be deleted. If False, actually delete.
    """
    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue

        # tree_id is always the folder name (e.g. "353_8.1_255W")
        tree_id = folder.name.split("_")[0]  # first part only
        # keep_name = f"{tree_id}.pkl"

        for pkl_file in folder.glob("*.pkl"):
            # if pkl_file.name == keep_name:
            #     print(f"Keeping {pkl_file}")
            #     continue

            if dry_run:
                print(f"[DRY RUN] Would delete {pkl_file}")
            else:
                print(f"Deleting {pkl_file}")
                pkl_file.unlink()

if __name__ == "__main__":
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Actually delete files (default: dry-run)")
    args = parser.parse_args()

    base = Path.cwd() / "data"
    clean_pkls(base, dry_run=not args.delete)
