import os
from pathlib import Path

img_dir = Path("./recorded_dataset/rgb")
npy_dir = Path("./recorded_dataset/data")

img_ids = set()

for img_file in img_dir.glob("step_*.png"):
    img_ids.add(img_file.stem)

print(f"[INFO] Found {len(img_ids)} images")

deleted_count = 0
checked_count = 0

for npy_file in npy_dir.glob("step_*.npy"):
    checked_count += 1

    stem = npy_file.stem

    if stem.endswith("_merged"):
        base_stem = stem[:-7]
    else:
        base_stem = stem

    if base_stem not in img_ids:
        print(f"[DELETE] {npy_file.name}")
        os.remove(npy_file)
        deleted_count += 1

print("\n====== DONE ======")
print(f"Checked npy files: {checked_count}")
print(f"Deleted npy files: {deleted_count}")