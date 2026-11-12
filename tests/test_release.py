import hashlib
import json
import math
from pathlib import Path

from salt.data import read_split
from salt.utils import load_head


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_counts_and_hashes():
    for dataset, counts in {"LEVIR": (445, 64, 128), "WHU3": (4736, 1036, 2416)}.items():
        for split, count in zip(("train", "val", "test"), counts):
            assert len(read_split(dataset, split)) == count
    hashes = json.loads((ROOT / "splits/sha256.json").read_text())
    for path, digest in hashes.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_six_minimal_checkpoints_and_hashes():
    files = list((ROOT / "checkpoints").glob("*/seed*.pt"))
    assert len(files) == 6
    for path in files:
        head, checkpoint = load_head(path, "cpu")
        assert sum(p.numel() for p in head.parameters()) == 67778
        assert set(checkpoint) == {"state", "config", "seed", "epoch", "source_threshold"}
        assert checkpoint["seed"] in (111, 222, 333)
        assert math.isfinite(checkpoint["source_threshold"])
    hashes = json.loads((ROOT / "checkpoints/sha256.json").read_text())
    for path, digest in hashes.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
