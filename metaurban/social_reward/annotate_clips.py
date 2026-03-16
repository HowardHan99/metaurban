#!/usr/bin/env python3
"""
annotate_clips.py — CLI entry point for LLM annotation of clip .npz files.

Usage examples
--------------
# Dry-run with mock backend (no API key needed):
python annotate_clips.py \\
    --clips-dir dataset/clips \\
    --backend mock \\
    --out-dir dataset/annotations

# Real run with OpenAI GPT-4o:
OPENAI_API_KEY=sk-... python annotate_clips.py \\
    --clips-dir dataset/clips \\
    --backend openai \\
    --model gpt-4o \\
    --max-frames 6 \\
    --rate-limit-delay 1.0 \\
    --out-dir dataset/annotations

# Google Gemini:
GOOGLE_API_KEY=... python annotate_clips.py \\
    --clips-dir dataset/clips \\
    --backend google \\
    --model gemini-1.5-pro-latest \\
    --out-dir dataset/annotations

After annotation, write reward labels into the clips themselves:
python annotate_clips.py --write-rewards \\
    --clips-dir dataset/clips \\
    --annotations-dir dataset/annotations
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("annotate_clips")


# ---------------------------------------------------------------------------
# Reward-label writer
# ---------------------------------------------------------------------------

def write_rewards_to_clips(clips_dir: Path, annotations_dir: Path) -> int:
    """
    For each annotation JSON in *annotations_dir*, open the matching clip .npz
    and append two new arrays:
      - ``social_penalty``      : scalar float32
      - ``social_present_mask`` : (n_labels,) bool — which labels were flagged

    Returns the number of clips successfully updated.
    """
    from metaurban.social_reward.taxonomy import SOCIAL_ISSUE_DEFINITIONS
    import json

    label_names = list(SOCIAL_ISSUE_DEFINITIONS.keys())
    updated = 0

    for ann_path in sorted(annotations_dir.glob("*.json")):
        try:
            ann_data = json.loads(ann_path.read_text())
        except Exception as exc:
            logger.error("Cannot read %s: %s", ann_path, exc)
            continue

        clip_stem = ann_data.get("clip_id", ann_path.stem)
        clip_path = clips_dir / f"{clip_stem}.npz"
        if not clip_path.exists():
            logger.warning("Clip not found for annotation %s", ann_path.name)
            continue

        # Load existing arrays
        old = np.load(str(clip_path), allow_pickle=False)
        arrays = dict(old)

        # Add / overwrite reward arrays
        penalty = float(ann_data.get("social_penalty", 0.0))
        arrays["social_penalty"] = np.array([penalty], dtype=np.float32)

        present_mask = np.zeros(len(label_names), dtype=bool)
        for entry in ann_data.get("annotations", []):
            if entry.get("present", False) and entry.get("label") in label_names:
                present_mask[label_names.index(entry["label"])] = True
        arrays["social_present_mask"] = present_mask

        # Overwrite the clip file in-place
        np.savez_compressed(str(clip_path), **arrays)
        updated += 1
        logger.debug("Updated %s with penalty=%.4f", clip_path.name, penalty)

    logger.info("Reward labels written to %d clips in %s", updated, clips_dir)
    return updated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Annotate MetaUrban clip files with LLM social issue labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--clips-dir", type=str, required=True,
                   help="Directory containing clip_*.npz files.")
    p.add_argument("--out-dir", type=str, default="dataset/annotations",
                   help="Directory to write annotation JSON files.")
    p.add_argument("--backend", type=str, default="mock",
                   choices=["openai", "google", "mock"],
                   help="LLM backend to use.")
    p.add_argument("--model", type=str, default="",
                   help="Model name (backend-specific). Leave empty for default.")
    p.add_argument("--max-frames", type=int, default=8,
                   help="Max RGB frames to send per clip (vision calls only).")
    p.add_argument("--rate-limit-delay", type=float, default=1.0,
                   help="Seconds to wait between API calls.")
    p.add_argument("--global-lambda", type=float, default=0.5,
                   help="Lambda coefficient in the social penalty formula.")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-annotate clips that already have a JSON file.")
    p.add_argument("--pattern", type=str, default="clip_*.npz",
                   help="Glob pattern for clip files.")
    p.add_argument("--write-rewards", action="store_true",
                   help=(
                       "After annotation, write social_penalty and "
                       "social_present_mask arrays back into each .npz clip."
                   ))
    p.add_argument("--annotations-dir", type=str, default="",
                   help=(
                       "Directory of annotation JSONs for --write-rewards. "
                       "Defaults to --out-dir."
                   ))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    clips_dir = Path(args.clips_dir)
    out_dir   = Path(args.out_dir)

    if not clips_dir.exists():
        logger.error("clips-dir does not exist: %s", clips_dir)
        sys.exit(1)

    if args.write_rewards:
        ann_dir = Path(args.annotations_dir) if args.annotations_dir else out_dir
        if not ann_dir.exists():
            logger.error(
                "--write-rewards requires annotations to exist at %s", ann_dir
            )
            sys.exit(1)
        n = write_rewards_to_clips(clips_dir, ann_dir)
        logger.info("Done. %d clips updated.", n)
        return

    # Normal annotation run
    from metaurban.social_reward.llm_annotator import ClipAnnotator

    annotator = ClipAnnotator(
        backend=args.backend,
        model=args.model,
        max_frames=args.max_frames,
        out_dir=out_dir,
        skip_existing=not args.no_skip_existing,
        global_lambda=args.global_lambda,
    )

    results = annotator.annotate_batch(
        clips_dir=clips_dir,
        pattern=args.pattern,
        rate_limit_delay=args.rate_limit_delay,
    )

    n_flagged = sum(
        1 for r in results
        if any(a.get("present", False) for a in r.get("annotations", []))
    )
    logger.info(
        "Summary: %d clips annotated, %d had at least one social issue.",
        len(results), n_flagged,
    )

    if args.write_rewards:
        write_rewards_to_clips(clips_dir, out_dir)


if __name__ == "__main__":
    main()
