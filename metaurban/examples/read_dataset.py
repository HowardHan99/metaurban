import numpy as np
from pathlib import Path

dir_path = Path("./recorded_dataset/data")

for f in sorted(dir_path.glob("*.npy"))[:10]:
    data = np.load(f, allow_pickle=True).item()
    print(f.name, "keys:", data.keys())