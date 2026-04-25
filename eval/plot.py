"""
eval/plot.py
============
Plot helpers for benchmark JSON outputs and training logs.

Usage
-----
.. code:: bash

    # Compare benchmark JSONs (one bar per agent, three bars per difficulty)
    python -m eval.plot bench --inputs eval/results/random_*.json \\
        eval/results/heuristic_*.json eval/results/oracle_*.json \\
        --out eval/results/bench_compare.png

    # Plot training reward curve from a JSONL training log
    python -m eval.plot train --input training/logs/run.jsonl \\
        --out training/plots/reward_curve.png
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any


def _load(paths: list[str]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for pat in paths:
        for f in sorted(glob.glob(pat)):
            runs.append(json.loads(Path(f).read_text()))
    return runs


def plot_benchmark(inputs: list[str], out: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    runs = _load(inputs)
    if not runs:
        raise SystemExit(f"No benchmark JSONs matched: {inputs}")

    diffs = ["easy", "medium", "hard"]
    agents = [r.get("agent", "unknown") for r in runs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(1, len(runs))
    for i, run in enumerate(runs):
        means: list[float] = []
        errs: list[float] = []
        for d in diffs:
            s = run.get("summary", {}).get(d) or {}
            means.append(float(s.get("mean_reward", 0.0)))
            errs.append(float(s.get("stderr_reward", 0.0)))
        x = [j + i * width for j in range(len(diffs))]
        ax.bar(x, means, width=width, yerr=errs, capsize=3, label=agents[i])

    ax.set_xticks([j + width * (len(runs) - 1) / 2 for j in range(len(diffs))])
    ax.set_xticklabels(diffs)
    ax.set_ylabel("mean episode reward")
    ax.set_title("Oversight Arena — agent comparison")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"[plot] wrote {out}")


def plot_training(jsonl_path: str, out: str, smooth: int = 25) -> None:
    """Render reward curve and per-component stack from a JSONL training log.

    Expected log line schema (one JSON object per line)::

        {"episode": int, "difficulty": str, "total_reward": float,
         "reward_breakdown": {...}, "format_penalty_count": int,
         "deep_inspect_count": int}
    """
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    rows: list[dict[str, Any]] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No log rows in {jsonl_path}")

    eps = [r["episode"] for r in rows]
    rewards = [float(r.get("total_reward", 0.0)) for r in rows]

    def _smooth(xs: list[float], k: int) -> list[float]:
        if k <= 1:
            return xs
        out = []
        for i in range(len(xs)):
            lo = max(0, i - k + 1)
            out.append(statistics.fmean(xs[lo:i + 1]))
        return out

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(eps, rewards, alpha=0.25, label="raw")
    axes[0].plot(eps, _smooth(rewards, smooth), label=f"sma({smooth})")
    axes[0].set_ylabel("episode reward")
    axes[0].set_title("Training reward curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    keys: set[str] = set()
    for r in rows:
        keys.update((r.get("reward_breakdown") or {}).keys())
    for k in sorted(keys):
        series = [
            float((r.get("reward_breakdown") or {}).get(k, 0.0)) for r in rows
        ]
        axes[1].plot(eps, _smooth(series, smooth), label=k)
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel(f"mean component (sma {smooth})")
    axes[1].set_title("Reward components over training")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"[plot] wrote {out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("bench", help="Plot benchmark agent comparison.")
    pb.add_argument("--inputs", nargs="+", required=True)
    pb.add_argument("--out", required=True)

    pt = sub.add_parser("train", help="Plot training reward curve from JSONL log.")
    pt.add_argument("--input", required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--smooth", type=int, default=25)

    args = p.parse_args(argv)
    if args.cmd == "bench":
        plot_benchmark(args.inputs, args.out)
    elif args.cmd == "train":
        plot_training(args.input, args.out, smooth=args.smooth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
