"""Reference / generation 0 / generation N comparison figure for an image run.

Example:
    python plot_samples.py runs/mnist_last-gen/20260101_120000 --gen 100
"""

import argparse
import json
import os

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from diffusion import available_generations, gen_dir, set_seed  # noqa: E402
from train_recursive_images import DATASETS, load_real_data, to_image  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42})

TITLE_SIZE = 40
SUPTITLE_SIZE = 48
DISPLAY_NAME = {"mnist": "MNIST", "fashion-mnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--gen", type=int, default=None, help="last generation to show (default: highest available)")
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--cols", type=int, default=5, help="images per block")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default=None, help="output PDF (default: inside the run directory)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)
    dataset, protocol = cfg["dataset"], cfg["protocol"]

    gens = available_generations(args.run_dir)
    last_gen = args.gen if args.gen is not None else max(gens)
    n = args.rows * args.cols

    set_seed(cfg.get("seed", 0))
    real = load_real_data(dataset, args.data_root, cfg["data_points"], DATASETS[dataset]["channels"])
    gen0 = torch.load(os.path.join(gen_dir(args.run_dir, 0), "synthetic_data.pt"), weights_only=False)
    genN = torch.load(os.path.join(gen_dir(args.run_dir, last_gen), "synthetic_data.pt"),
                      weights_only=False)

    blocks = [(real, "Reference dataset"), (gen0, "Generation 0"), (genN, f"Generation {last_gen}")]

    fig, axes = plt.subplots(args.rows, args.cols * 3,
                             figsize=(args.cols * 3 * 1.75, args.rows * 2.75))
    for i in range(n):
        row, col = divmod(i, args.cols)
        for b, (data, _) in enumerate(blocks):
            ax = axes[row, col + b * args.cols]
            ax.imshow(to_image(data[i]), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

    for b, (_, label) in enumerate(blocks):
        axes[0, 1 + b * args.cols].set_title(label, fontsize=TITLE_SIZE, pad=4)

    for x in (0.335, 0.665):
        fig.add_artist(plt.Line2D([x, x], [0.05, 0.95], transform=fig.transFigure,
                                  color="black", linewidth=2))

    fig.suptitle(DISPLAY_NAME.get(dataset, dataset), fontsize=SUPTITLE_SIZE)
    fig.tight_layout()

    out = args.out or os.path.join(args.run_dir,
                                   f"{dataset}_comparison_{protocol}_gen{last_gen}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to {out}")


if __name__ == "__main__":
    main()
