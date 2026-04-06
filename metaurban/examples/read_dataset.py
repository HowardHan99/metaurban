import os
import random
import numpy as np
import cv2

DATA_DIR = "./recorded_dataset"


def list_files(data_dir):
    rgb_dir = os.path.join(data_dir, "rgb")
    npy_dir = os.path.join(data_dir, "data")

    if not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"RGB folder not found: {rgb_dir}")
    if not os.path.exists(npy_dir):
        raise FileNotFoundError(f"Data folder not found: {npy_dir}")

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy")])

    print(f"Total RGB images: {len(rgb_files)}")
    print(f"Total data files: {len(npy_files)}")

    if len(rgb_files) != len(npy_files):
        print("[WARN] Number of RGB files and NPY files does not match.")

    return rgb_dir, npy_dir, rgb_files, npy_files


def load_sample(idx, rgb_dir, npy_dir, rgb_files, npy_files):
    rgb_path = os.path.join(rgb_dir, rgb_files[idx])
    npy_path = os.path.join(npy_dir, npy_files[idx])

    img = cv2.imread(rgb_path)
    if img is None:
        raise ValueError(f"Failed to read image: {rgb_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data = np.load(npy_path, allow_pickle=True).item()

    return img, data, rgb_path, npy_path


def inspect_one_sample(idx, rgb_dir, npy_dir, rgb_files, npy_files, show_image=False):
    img, data, rgb_path, npy_path = load_sample(idx, rgb_dir, npy_dir, rgb_files, npy_files)

    print("\n" + "=" * 20)
    print(f"SAMPLE INDEX: {idx}")
    print(f"RGB PATH: {rgb_path}")
    print(f"NPY PATH: {npy_path}")

    print("\n===== DATA STRUCTURE =====")
    print("keys:", data.keys())

    print("\n===== ACTION =====")
    print(data["action"])
    try:
        print("action len:", len(data["action"]))
    except Exception:
        print("action len: cannot compute")

    print("\n===== REWARD =====")
    print(data["reward"])

    print("\n===== TERMINAL =====")
    print("terminal:", data["terminal"], "trunc:", data["trunc"])

    print("\n===== STATE TYPE =====")
    state = data["state"]
    print(type(state))

    if isinstance(state, np.ndarray):
        print("\n===== STATE SHAPE =====")
        print(state.shape)

        print("\n===== STATE FIRST 20 VALUES =====")
        print(state[:20])

        print("\n===== STATE STATS =====")
        print("min :", state.min())
        print("max :", state.max())
        print("mean:", state.mean())
        print("std :", state.std())
    else:
        print("\n[WARN] state is not numpy.ndarray")
        print(state)

    print("\n===== IMAGE =====")
    print("image shape:", img.shape)
    print("image dtype:", img.dtype)
    print("image min/max:", img.min(), img.max())

    if show_image:
        cv2.imshow(f"sample_{idx}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def inspect_multiple_samples(rgb_dir, npy_dir, rgb_files, npy_files, num_samples=5):
    n = min(num_samples, len(npy_files))
    print(f"\nInspecting first {n} samples...")

    for idx in range(n):
        img, data, _, _ = load_sample(idx, rgb_dir, npy_dir, rgb_files, npy_files)
        state = data["state"]
        action = data["action"]

        print("\n" + "-" * 50)
        print(f"sample {idx}")
        print("state shape:", state.shape if isinstance(state, np.ndarray) else type(state))
        print("state first 10:", state[:10] if isinstance(state, np.ndarray) else state)
        print("action:", action)
        print("reward:", data["reward"])
        print("terminal/trunc:", data["terminal"], data["trunc"])
        print("image shape:", img.shape)


def analyze_whole_dataset(rgb_dir, npy_dir, rgb_files, npy_files):
    all_states = []
    all_actions = []
    all_rewards = []

    for idx in range(len(npy_files)):
        _, data, _, _ = load_sample(idx, rgb_dir, npy_dir, rgb_files, npy_files)

        state = data["state"]
        action = np.array(data["action"], dtype=np.float32)
        reward = float(data["reward"])

        all_states.append(state)
        all_actions.append(action)
        all_rewards.append(reward)

    all_states = np.array(all_states)
    all_actions = np.array(all_actions)
    all_rewards = np.array(all_rewards)

    print("\n" + "=" * 20)
    print("WHOLE DATASET ANALYSIS")

    print("\n===== DATASET SHAPES =====")
    print("all_states shape :", all_states.shape)
    print("all_actions shape:", all_actions.shape)
    print("all_rewards shape:", all_rewards.shape)

    print("\n===== REWARD STATS =====")
    print("min :", all_rewards.min())
    print("max :", all_rewards.max())
    print("mean:", all_rewards.mean())
    print("std :", all_rewards.std())

    print("\n===== ACTION STATS =====")
    print("action min :", all_actions.min(axis=0))
    print("action max :", all_actions.max(axis=0))
    print("action mean:", all_actions.mean(axis=0))
    print("action std :", all_actions.std(axis=0))

    print("\n===== STATE GLOBAL STATS =====")
    print("state min :", all_states.min())
    print("state max :", all_states.max())
    print("state mean:", all_states.mean())
    print("state std :", all_states.std())

    print("\n===== STATE PER-DIM STD (first 30 dims) =====")
    std_per_dim = all_states.std(axis=0)
    for i in range(min(30, len(std_per_dim))):
        print(f"dim {i:03d}: std={std_per_dim[i]:.6f}")

    print("\n===== STATE PER-DIM MIN/MAX (first 20 dims) =====")
    min_per_dim = all_states.min(axis=0)
    max_per_dim = all_states.max(axis=0)
    for i in range(min(20, len(min_per_dim))):
        print(f"dim {i:03d}: min={min_per_dim[i]:.6f}, max={max_per_dim[i]:.6f}")


def inspect_random_sample(rgb_dir, npy_dir, rgb_files, npy_files):
    idx = random.randint(0, len(npy_files) - 1)
    inspect_one_sample(idx, rgb_dir, npy_dir, rgb_files, npy_files, show_image=False)


def main():
    rgb_dir, npy_dir, rgb_files, npy_files = list_files(DATA_DIR)

    if len(npy_files) == 0:
        print("No data found.")
        return

    inspect_one_sample(
        idx=0,
        rgb_dir=rgb_dir,
        npy_dir=npy_dir,
        rgb_files=rgb_files,
        npy_files=npy_files,
        show_image=False,
    )

    inspect_multiple_samples(
        rgb_dir=rgb_dir,
        npy_dir=npy_dir,
        rgb_files=rgb_files,
        npy_files=npy_files,
        num_samples=5,
    )

    analyze_whole_dataset(
        rgb_dir=rgb_dir,
        npy_dir=npy_dir,
        rgb_files=rgb_files,
        npy_files=npy_files,
    )

    inspect_random_sample(
        rgb_dir=rgb_dir,
        npy_dir=npy_dir,
        rgb_files=rgb_files,
        npy_files=npy_files,
    )


if __name__ == "__main__":
    main()