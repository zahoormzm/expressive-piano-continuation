"""Create a standalone training/validation loss chart for recorded scratch runs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/training-curves.png"))
    parser.add_argument("--title")
    args = parser.parse_args()
    figure, axes = plt.subplots(
        1, len(args.runs), figsize=(6 * len(args.runs), 4.5), squeeze=False, sharey=True
    )
    for axis, run in zip(axes[0], args.runs):
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        initial = json.loads((run / "initial-validation.json").read_text())["loss"]
        config = json.loads((run / "resolved-config.json").read_text())
        axis.plot(
            [row["valid_tokens_seen"] / 1e6 for row in rows],
            [row["loss"] for row in rows],
            color="#456b98",
            linewidth=1.2,
            alpha=0.8,
            label="Training update",
        )
        validation = [row for row in rows if "validation_loss" in row]
        axis.plot(
            [0] + [row["valid_tokens_seen"] / 1e6 for row in validation],
            [initial] + [row["validation_loss"] for row in validation],
            color="#c76836",
            marker="o",
            linewidth=1.8,
            label="Fixed validation",
        )
        axis.set(
            title=config["model"]["kind"].upper(),
            xlabel="Valid target tokens processed (millions)",
            ylabel="Mean next-token loss (nats)",
        )
        axis.grid(alpha=0.18)
        axis.legend(frameon=False)
    limited = all(
        json.loads((run / "resolved-config.json").read_text())["data"]["limited_pilot"]
        for run in args.runs
    )
    title = args.title or (
        "Scratch screening pilots on the 32-performance cache"
        if limited
        else "Scratch training on the retained full corpus"
    )
    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
