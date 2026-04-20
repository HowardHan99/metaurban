import json
from collections import Counter

path = "./recorded_dataset/label/vlm_labels.jsonl"

counter = Counter()
total = 0

with open(path, "r") as f:
    for line in f:
        data = json.loads(line)
        
        label = data["vlm_label_id"]
        if label == 2:
            print(data["image_path"], data["reason"])
        counter[label] += 1
        total += 1

print("Total:", total)
print()

for k, v in counter.items():
    print(f"{k}: {v} ({v/total:.2%})")



# # delete npy files that have neutral labels, and also delete the corresponding png files
# import json
# from pathlib import Path

# LABEL_JSONL = Path("./recorded_dataset/label/vlm_labels.jsonl")
# DATA_DIR = Path("./recorded_dataset/data_merged")
# RGB_DIR = Path("./recorded_dataset/rgb_merged")

# NEUTRAL_LABEL_ID = 1


# def main():
#     if not LABEL_JSONL.exists():
#         raise FileNotFoundError(f"Label file not found: {LABEL_JSONL}")

#     neutral_ids = []

#     with open(LABEL_JSONL, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             data = json.loads(line)
#             if data.get("vlm_label_id") == NEUTRAL_LABEL_ID:
#                 neutral_ids.append(int(data["idx"]))

#     print(f"Found {len(neutral_ids)} neutral samples.")

#     deleted_npy = 0
#     deleted_png = 0
#     missing_npy = 0
#     missing_png = 0

#     for idx in neutral_ids:
#         npy_path = DATA_DIR / f"step_{idx:06d}.npy"
#         png_path = RGB_DIR / f"step_{idx:06d}.png"

#         if npy_path.exists():
#             npy_path.unlink()
#             deleted_npy += 1
#         else:
#             missing_npy += 1

#         if png_path.exists():
#             png_path.unlink()
#             deleted_png += 1
#         else:
#             missing_png += 1

#     print("Done.")
#     print(f"Deleted npy: {deleted_npy}")
#     print(f"Deleted png: {deleted_png}")
#     print(f"Missing npy: {missing_npy}")
#     print(f"Missing png: {missing_png}")


# if __name__ == "__main__":
#     main()