"""Metric curves over generations, for an image run or a spiral run.

Reads <run_dir>/metrics.json, written by evaluate_images.py for image runs and by
train_recursive_spiral.py for spiral runs, and draws one panel per metric.

Example:
    python plot_metrics.py runs/fashion-mnist_last-gen/20260101_120000
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42})

CURVE_COLOR = "#1f77b4"
CURVE_LW = 4

IMAGE_PANELS = [
    ("fid_inception", "FID Inception\n(features 2048-dim)"),
    ("frechet_pixel", "Frechet pixel\n(no Inception)"),
    ("w2_pixel", "Wasserstein-2 pixel"),
]
SPIRAL_PANELS = [
    ("frechet_distance", "Frechet 2D\n(no Inception, 2-dim)"),
    ("W2", "Wasserstein-2"),
    ("var_ratio", "Variance ratio\n(generated / real)"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--title", default=None, help="figure title (default: taken from the run config)")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_panels(run_dir):
    """Return (generations, [(title, values)]) for either metrics.json layout."""
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)

    if "fid_inception" in metrics:  # written by evaluate_images.py
        gens = sorted(int(g) for g in metrics["fid_inception"])
        return gens, [(title, [metrics[key][str(g)] for g in gens]) for key, title in IMAGE_PANELS]

    # written by train_recursive_spiral.py: one dict of metrics per generation
    gens = sorted(int(g) for g in metrics)
    return gens, [(title, [metrics[str(g)][key] for g in gens]) for key, title in SPIRAL_PANELS]


def main():
    args = parse_args()
    gens, panels = load_panels(args.run_dir)

    title = args.title
    if title is None:
        config_path = os.path.join(args.run_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            title = f"{cfg.get('dataset', 'spiral')} ({cfg.get('protocol', '')})"

    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3))
    for ax, (panel_title, values) in zip(axes, panels):
        ax.plot(gens, values, linewidth=CURVE_LW, color=CURVE_COLOR)
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_title(panel_title, fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    if title:
        fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(top=0.78)

    out = args.out or os.path.join(args.run_dir, "metrics_over_generations.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to {out}")


if __name__ == "__main__":
    main()
