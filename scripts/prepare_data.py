#!/usr/bin/env python3
"""Validate locally supplied datasets against the released split manifests."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from salt.data import read_split
from salt.utils import write_json


COUNTS = {"LEVIR": {"train": 445, "val": 64, "test": 128},
          "WHU3": {"train": 4736, "val": 1036, "test": 2416}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--check-masks", action="store_true", help="Also verify original mask-derived image labels")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {}
    for dataset, splits in COUNTS.items():
        seen = set()
        report[dataset] = {}
        for split, count in splits.items():
            rows = read_split(dataset, split)
            if len(rows) != count:
                raise ValueError(f"{dataset}/{split}: expected {count} samples")
            positive = 0
            for row in rows:
                if row["image_path"] in seen:
                    raise ValueError(f"Overlapping splits: {dataset}/{row['image_path']}")
                seen.add(row["image_path"])
                for field in ("image_path", "mask_path"):
                    path = args.data_root / dataset / row[field]
                    if not path.is_file():
                        raise FileNotFoundError(path)
                if args.check_masks:
                    with Image.open(args.data_root / dataset / row["mask_path"]) as image:
                        label = int((np.asarray(image.convert("L")) > 0).any())
                    if label != int(row["label"]):
                        raise ValueError(f"Image-label mismatch: {dataset}/{row['mask_path']}")
                positive += int(row["label"])
            report[dataset][split] = {"total": count, "positive": positive, "negative": count - positive}
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
