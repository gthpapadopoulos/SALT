#!/usr/bin/env python3
"""Summarize the six SALT evaluations as mean +/- sample standard deviation."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from salt.metrics import METRICS
from salt.utils import write_json


def summarize(root):
    grouped = {}
    for path in sorted(Path(root).glob("*/seed*/result.json")):
        with path.open() as handle:
            record = json.load(handle)
        config = record["config"]
        key = (config["source"], config["target"])
        group = grouped.setdefault(key, {})
        seed = record["seed"]
        if seed in group:
            raise ValueError(f"Duplicate seed {seed} for {key}")
        if record["target_normalization"] or record["target_threshold_search"]:
            raise ValueError(f"Unexpected target calibration: {path}")
        group[seed] = record
    expected = {("LEVIR", "WHU3"), ("WHU3", "LEVIR")}
    if set(grouped) != expected:
        raise ValueError("Both transfer directions are required")
    rows = []
    for (source, target), group in sorted(grouped.items()):
        if set(group) != {111, 222, 333}:
            raise ValueError(f"Missing seeds for {source}->{target}: {set(group)}")
        configs = [group[s]["config"] for s in (111, 222, 333)]
        if any(c != configs[0] for c in configs[1:]):
            raise ValueError("Configuration differs across seeds")
        row = {"source": source, "target": target, "n": 3}
        for metric in METRICS:
            values = [group[seed]["target_metrics"][metric] for seed in (111, 222, 333)]
            if any(v is None for v in values):
                row[f"{metric}_mean"] = row[f"{metric}_std"] = None
            else:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("results/summary"))
    args = parser.parse_args()
    rows = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output.with_suffix(".json"), rows)
    print("Direction | P-AUROC | P-AP | F1 | IoU | I-AUROC | I-AP")
    for row in rows:
        values = []
        for metric in METRICS:
            mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
            values.append("unavailable" if mean is None else f"{mean:.2f} +/- {std:.2f}")
        print(f"{row['source']} -> {row['target']} | " + " | ".join(values))


if __name__ == "__main__":
    main()
