import json
from collections import Counter

path = "./recorded_dataset/label/vlm_labels.jsonl"

counter = Counter()
total = 0

with open(path, "r") as f:
    for line in f:
        data = json.loads(line)
        
        label = data["vlm_label_id"]
        counter[label] += 1
        total += 1

print("Total:", total)
print()

for k, v in counter.items():
    print(f"{k}: {v} ({v/total:.2%})")