#!/usr/bin/env python3
"""Queue two directions x three seeds with one experiment per selected GPU."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def available_gpus():
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True)
    result = []
    for line in text.splitlines():
        index, used, utilization = (int(value.strip()) for value in line.split(","))
        if used < 1024 and utilization < 10:
            result.append(str(index))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dinov3-repo", type=Path, default=ROOT / "third_party/dinov3")
    parser.add_argument("--gpus", nargs="+", help="Physical IDs; default: currently idle GPUs")
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    gpus = args.gpus or available_gpus()
    if not gpus or len(set(gpus)) != len(gpus):
        parser.error("No idle GPUs found, or duplicate GPU IDs supplied")
    args.output = args.output.resolve()
    jobs = deque()
    for direction in ("levir_to_whu", "whu_to_levir"):
        for seed in (111, 222, 333):
            output = args.output / direction / f"seed{seed}"
            if not (output / "result.json").exists():
                jobs.append((direction, seed, output, "train"))
    running = {}
    failed = False
    common = ["--data-root", str(args.data_root.resolve()), "--weights", str(args.weights.resolve()),
              "--dinov3-repo", str(args.dinov3_repo.resolve()), "--device", "cuda:0"]

    def stop(*_):
        for process, _, _, log in running.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, _, _, log in running.values():
            process.wait()
            log.close()
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while jobs or running:
        for gpu in gpus:
            if gpu in running or not jobs:
                continue
            direction, seed, output, stage = jobs.popleft()
            output.mkdir(parents=True, exist_ok=True)
            if stage == "train" and (output / "train_result.json").exists():
                stage = "evaluate"
            command = [sys.executable, "-u", str(ROOT / f"{stage}.py"), *common, "--output", str(output)]
            if stage == "train":
                command += ["--config", str(ROOT / "configs" / f"{direction}.yaml"), "--seed", str(seed)]
                if (output / "last.pt").exists():
                    command.append("--resume")
            else:
                command += ["--checkpoint", str(output / "best.pt"), "--profile"]
            log = (output / f"{stage}.log").open("a")
            log.write(json.dumps(command) + "\n")
            log.flush()
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                       cwd=ROOT, start_new_session=True)
            running[gpu] = (process, (direction, seed, output, stage), command, log)
            print(f"GPU {gpu}: {direction}, seed {seed}, {stage}", flush=True)
        for gpu, (process, job, _, log) in list(running.items()):
            status = process.poll()
            if status is None:
                continue
            log.close()
            del running[gpu]
            direction, seed, output, stage = job
            if status:
                print(f"FAILED: {direction} seed {seed} {stage}; see {output / (stage + '.log')}", flush=True)
                failed = True
            elif stage == "train":
                jobs.appendleft((direction, seed, output, "evaluate"))
        if running:
            time.sleep(2)
    if failed:
        raise SystemExit(1)
    subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), "--root", str(args.output),
                    "--output", str(args.output / "summary")], check=True)


if __name__ == "__main__":
    main()
