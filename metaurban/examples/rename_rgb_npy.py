#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def collect_pairs(root: Path):
    rgb_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("rgb")])
    pairs = []

    for rgb_dir in rgb_dirs:
        suffix = rgb_dir.name[3:]
        data_dir = root / f"data{suffix}"

        if not data_dir.exists() or not data_dir.is_dir():
            print(f"[WARN] Missing matching data folder for {rgb_dir.name}: expected {data_dir.name}")
            continue

        pngs = sorted(rgb_dir.glob("*.png"))
        for png_path in pngs:
            stem = png_path.stem
            npy_path = data_dir / f"{stem}.npy"
            merged_npy_path = data_dir / f"{stem}_merged.npy"

            if not npy_path.exists():
                print(f"[WARN] Missing npy for {png_path}")
                continue

            pairs.append({
                "png": png_path,
                "npy": npy_path,
                "merged_npy": merged_npy_path if merged_npy_path.exists() else None,
                "source_rgb_dir": rgb_dir.name,
                "source_data_dir": data_dir.name,
                "old_stem": stem,
            })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("./recorded_dataset"))
    parser.add_argument("--out-rgb", type=str, default="rgb_merged")
    parser.add_argument("--out-data", type=str, default="data_merged")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--digits", type=int, default=6)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them")
    args = parser.parse_args()

    root = args.root.resolve()
    out_rgb = root / args.out_rgb
    out_data = root / args.out_data
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(root)
    if not pairs:
        print("[ERROR] No valid png/npy pairs found.")
        return

    op = shutil.copy2 if args.copy else shutil.move

    index = args.start_index
    manifest_lines = ["new_stem,source_rgb_dir,source_data_dir,old_stem,has_merged_npy"]

    for item in pairs:
        new_stem = f"step_{index:0{args.digits}d}"

        new_png = out_rgb / f"{new_stem}.png"
        new_npy = out_data / f"{new_stem}.npy"

        op(str(item["png"]), str(new_png))
        op(str(item["npy"]), str(new_npy))

        has_merged = item["merged_npy"] is not None
        if has_merged:
            new_merged = out_data / f"{new_stem}_merged.npy"
            op(str(item["merged_npy"]), str(new_merged))

        manifest_lines.append(
            f"{new_stem},{item['source_rgb_dir']},{item['source_data_dir']},{item['old_stem']},{int(has_merged)}"
        )
        index += 1

    manifest_path = root / "merge_manifest.csv"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

    print(f"[OK] Total merged pairs: {len(pairs)}")
    print(f"[OK] Images -> {out_rgb}")
    print(f"[OK] Data   -> {out_data}")
    print(f"[OK] Manifest -> {manifest_path}")
    print(f"[INFO] Mode: {'copy' if args.copy else 'move'}")


if __name__ == "__main__":
    main()
