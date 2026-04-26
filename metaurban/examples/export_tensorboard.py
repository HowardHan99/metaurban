import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def sanitize_filename(name: str) -> str:
    name = name.replace("/", "__")
    name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)
    return name


def find_event_files(logdir: str) -> List[str]:
    event_files = []
    for root, _, files in os.walk(logdir):
        for f in files:
            if "tfevents" in f:
                event_files.append(os.path.join(root, f))
    return sorted(event_files)


def load_scalar_data(event_file: str) -> Dict[str, List[Tuple[int, float]]]:
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={
            event_accumulator.SCALARS: 0,
        },
    )
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    data = {}

    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]

    return data


def merge_scalar_data(all_data: List[Dict[str, List[Tuple[int, float]]]]) -> Dict[str, List[Tuple[int, float]]]:
    merged: Dict[str, List[Tuple[int, float]]] = {}

    for data in all_data:
        for tag, values in data.items():
            if tag not in merged:
                merged[tag] = []
            merged[tag].extend(values)

    for tag in merged:
        merged[tag] = sorted(merged[tag], key=lambda x: x[0])

    return merged


def plot_and_save(tag: str, values: List[Tuple[int, float]], outdir: str) -> None:
    if len(values) == 0:
        return

    steps = [x[0] for x in values]
    vals = [x[1] for x in values]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, vals)
    plt.xlabel("Step")
    plt.ylabel(tag)
    plt.title(tag)
    plt.tight_layout()

    filename = sanitize_filename(tag) + ".png"
    save_path = os.path.join(outdir, filename)
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Export all TensorBoard scalar plots to PNG.")
    parser.add_argument("--logdir", type=str, help="TensorBoard log directory",default="/home/howardhan/metaurban/midterm_logs/SAC_image_state/sac_imgstate_seed0_0415_1149")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for exported figures. Default: <logdir>/exported_scalars",
    )
    args = parser.parse_args()

    logdir = os.path.abspath(os.path.expanduser(args.logdir))
    outdir = (
        os.path.abspath(os.path.expanduser(args.outdir))
        if args.outdir is not None
        else os.path.join(logdir, "exported_scalars")
    )

    os.makedirs(outdir, exist_ok=True)

    event_files = find_event_files(logdir)
    if len(event_files) == 0:
        raise FileNotFoundError(f"No TensorBoard event files found under: {logdir}")

    print(f"Found {len(event_files)} event file(s).")
    print(f"Exporting figures to: {outdir}")

    all_data = []
    for event_file in event_files:
        print(f"Loading: {event_file}")
        all_data.append(load_scalar_data(event_file))

    merged = merge_scalar_data(all_data)

    if len(merged) == 0:
        print("No scalar tags found.")
        return

    print(f"Found {len(merged)} scalar tag(s).")

    for tag, values in merged.items():
        print(f"Saving tag: {tag}")
        plot_and_save(tag, values, outdir)

    print("Done.")


if __name__ == "__main__":
    main()