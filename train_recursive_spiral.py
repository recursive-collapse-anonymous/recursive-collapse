"""Recursive (self-consuming) training of a diffusion model on a 2D Archimedean spiral.

Same two protocols as train_recursive_images.py, on a toy target where the collapse can be
watched directly in data space and measured exactly (Wasserstein-2 by linear
programming, Frechet distance between the fitted Gaussians, variance ratio).

Example:
    python train_recursive_spiral.py --protocol last-gen
    python train_recursive_spiral.py --protocol fixed-budget --generations 100
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib
import numpy as np
import ot
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from diffusion import (DiffusionSchedule, EMA, PROTOCOLS, build_training_set,  # noqa: E402
                    ddim_sample, gen_dir, get_device, set_seed)
from models import MLP2D  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "figure.dpi": 120})

TITLE_SIZE = 24
REF_COLOR = "steelblue"
GEN_COLOR = "coral"
CURVE_COLOR = "#1f77b4"
CURVE_LW = 3
SCATTER_KW = dict(s=10, alpha=0.9, rasterized=True)  # rasterized points, vector axes and text


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", choices=PROTOCOLS, default="last-gen")
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--data-points", type=int, default=1000,
                   help="size of the real dataset and of every synthetic set")
    p.add_argument("--epochs", type=int, default=2000, help="epochs per generation >= 1")
    p.add_argument("--epochs-gen0", type=int, default=None,
                   help="epochs for generation 0 (default: same as --epochs)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ema-decay", type=float, default=0.8)
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--inference-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate of generation 0")
    p.add_argument("--lr-recursive", type=float, default=1e-3,
                   help="learning rate of generations >= 1")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--time-dim", type=int, default=64)
    p.add_argument("--spiral-radius", type=float, default=0.0)
    p.add_argument("--spiral-spacing", type=float, default=1.5)
    p.add_argument("--spiral-turns", type=float, default=2)
    p.add_argument("--eval-samples", type=int, default=1000,
                   help="samples used for the metrics and the scatter plots")
    p.add_argument("--ot-points", type=int, default=5000,
                   help="cap on the number of points per side in the exact optimal transport")
    p.add_argument("--figure-gens", type=int, nargs="*", default=None,
                   help="generations shown in the evolution figure (default: 6 evenly spaced)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="./runs")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--keep-checkpoints", action="store_true")
    return p.parse_args()


def sample_spiral(n, radius, spacing, turns):
    theta = np.linspace(0, 2 * np.pi * turns, n)
    r = radius + spacing * theta
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def compute_metrics(samples, real, n_ot_points, seed):
    """Variance ratio, exact Wasserstein-2 and 2D Frechet distance to the real data."""
    metrics = {
        "var_x": float(np.var(samples[:, 0])),
        "var_y": float(np.var(samples[:, 1])),
    }
    metrics["var_total"] = metrics["var_x"] + metrics["var_y"]
    real_var = float(np.var(real[:, 0]) + np.var(real[:, 1]))
    metrics["var_ratio"] = metrics["var_total"] / real_var if real_var > 0 else 0.0

    eigvals = np.linalg.eigvalsh(np.cov(samples.T))
    metrics["cov_eigenval_0"] = float(eigvals[0])
    metrics["cov_eigenval_1"] = float(eigvals[1])

    # Exact optimal transport. Both marginals are subsampled: past a few thousand
    # points per side ot.emd2 stops on its iteration limit and returns a cost that
    # is too small. The subsampling uses its own generator and does not touch the
    # global numpy stream.
    rng = np.random.default_rng(seed)
    n_ot = min(len(real), len(samples), n_ot_points)
    real_ot = real[rng.choice(len(real), n_ot, replace=False)] if len(real) > n_ot else real
    samp_ot = samples[rng.choice(len(samples), n_ot, replace=False)] if len(samples) > n_ot else samples

    a = np.ones(len(real_ot)) / len(real_ot)
    b = np.ones(len(samp_ot)) / len(samp_ot)
    M = ot.dist(real_ot, samp_ot, metric="sqeuclidean")
    cost, log = ot.emd2(a, b, M, numItermax=1_000_000, log=True)
    if log.get("warning") is not None:
        print(f"    [warning] ot.emd2: {log['warning']}")
    metrics["W2"] = float(np.sqrt(max(cost, 0.0)))
    metrics["n_ot_points"] = int(n_ot)

    # Frechet distance between the Gaussians fitted on the two clouds.
    diff = real.mean(axis=0) - samples.mean(axis=0)
    Sr, Sg = np.cov(real, rowvar=False), np.cov(samples, rowvar=False)
    covmean = sqrtm(Sr @ Sg)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    metrics["frechet_distance"] = float(diff @ diff + np.trace(Sr + Sg - 2 * covmean))

    return metrics


def axis_limits(real, margin=0.08):
    """Square limits computed once on the real data.

    Fixed limits are required: with autoscaling every scatter plot is rescaled to
    its own cloud and the variance collapse becomes invisible across generations.
    """
    lo, hi = real.min(axis=0), real.max(axis=0)
    span = float((hi - lo).max())
    ctr = (hi + lo) / 2.0
    half = span * (0.5 + margin)
    return (ctr[0] - half, ctr[0] + half), (ctr[1] - half, ctr[1] + half)


def style_axis(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")


def train_one_generation(model, optimizer, ema, train_data, gen, n_epochs, args, schedule):
    loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True)
    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        for (x,) in loader:
            x = x.to(args.device)
            t = torch.randint(0, schedule.T, (x.shape[0],), device=args.device)
            xt, noise = schedule.add_noise(x, t)
            loss = F.mse_loss(model(xt, t), noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            epoch_losses.append(loss.item())
        losses.append(float(np.mean(epoch_losses)))
        if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            print(f"  gen {gen:3d} | epoch {epoch + 1:4d}/{n_epochs} | loss {losses[-1]:.6f}",
                  flush=True)
    return losses


def main():
    args = parse_args()
    args.device = get_device(args.device)
    args.epochs_gen0 = args.epochs if args.epochs_gen0 is None else args.epochs_gen0

    run_dir = os.path.join(args.out, f"spiral_{args.protocol}",
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Device: {args.device}\nRun directory: {run_dir}")

    prefix = f"spiral_{args.protocol}"

    def save_figure(fig, name, subdir=None):
        target = subdir or fig_dir
        os.makedirs(target, exist_ok=True)
        path = os.path.join(target, f"{name}.pdf")
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return path

    set_seed(args.seed)
    schedule = DiffusionSchedule(args.diffusion_steps, args.device)

    real_np = sample_spiral(args.data_points, args.spiral_radius,
                            args.spiral_spacing, args.spiral_turns)
    real_data = torch.tensor(real_np, dtype=torch.float32)
    np.save(os.path.join(run_dir, "real_data.npy"), real_np)
    xlim, ylim = axis_limits(real_np)

    if not args.no_figures:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(real_np[:, 0], real_np[:, 1], c=REF_COLOR, **SCATTER_KW)
        ax.set_title("Reference dataset", fontsize=TITLE_SIZE)
        style_axis(ax, xlim, ylim)
        save_figure(fig, f"{prefix}_real_data")

    all_losses, all_metrics, synthetic = {}, {}, {}

    for gen in range(args.generations + 1):
        print("=" * 60)
        print(f"GENERATION {gen} ({args.protocol})")
        print("=" * 60)

        # Non-incremental protocol: a new model, optimizer and EMA at every generation.
        model = MLP2D(data_dim=2, hidden_dim=args.hidden_dim, time_dim=args.time_dim).to(args.device)
        lr = args.lr if gen == 0 else args.lr_recursive
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        ema = EMA(model, decay=args.ema_decay)
        if gen == 0:
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        if gen == 0:
            train_data, info = real_data, {"train_source": "real data", "pool_size": len(real_data),
                                           "real_fraction_in_pool": 1.0}
        else:
            train_data, info = build_training_set(args.protocol, real_data, synthetic,
                                                  gen, args.data_points)
        print(f"  training set: {len(train_data):,} samples ({info['train_source']}, "
              f"real fraction in pool {info['real_fraction_in_pool']:.1%})")

        out_dir = gen_dir(run_dir, gen, create=True)
        n_epochs = args.epochs_gen0 if gen == 0 else args.epochs
        losses = train_one_generation(model, optimizer, ema, train_data, gen, n_epochs,
                                      args, schedule)
        all_losses[gen] = losses

        synthetic[gen] = ddim_sample(ema.ema_model, args.data_points, (2,), schedule,
                                     args.inference_steps, args.device)
        torch.save(synthetic[gen], os.path.join(out_dir, "synthetic_data.pt"))
        np.save(os.path.join(out_dir, "synthetic_data.npy"), synthetic[gen].numpy())

        eval_samples = ddim_sample(ema.ema_model, args.eval_samples, (2,), schedule,
                                   args.inference_steps, args.device).numpy()
        all_metrics[gen] = compute_metrics(eval_samples, real_np, args.ot_points, args.seed)
        print(f"  var_ratio={all_metrics[gen]['var_ratio']:.3f} | "
              f"W2={all_metrics[gen]['W2']:.4f} | "
              f"Frechet={all_metrics[gen]['frechet_distance']:.4f}")

        if args.keep_checkpoints:
            torch.save({"generation": gen, "model_state_dict": model.state_dict(),
                        "ema_state_dict": ema.ema_model.state_dict(), "losses": losses},
                       os.path.join(out_dir, f"checkpoint_gen{gen}.pt"))

        info.update({"generation": gen, "final_loss": losses[-1], "metrics": all_metrics[gen]})
        with open(os.path.join(out_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=2)

        if not args.no_figures:
            reference = real_np if gen == 0 else synthetic[0].numpy()
            ref_label = "Reference dataset" if gen == 0 else "Generation 0"
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].scatter(reference[:, 0], reference[:, 1], c=REF_COLOR, **SCATTER_KW)
            axes[0].set_title(ref_label, fontsize=TITLE_SIZE)
            style_axis(axes[0], xlim, ylim)
            axes[1].scatter(eval_samples[:, 0], eval_samples[:, 1], c=GEN_COLOR, **SCATTER_KW)
            axes[1].set_title(f"Generation {gen}", fontsize=TITLE_SIZE)
            style_axis(axes[1], xlim, ylim)
            fig.suptitle("Spiral", fontsize=TITLE_SIZE)
            fig.tight_layout()
            save_figure(fig, f"{prefix}_samples_gen_{gen:03d}")

        with open(os.path.join(run_dir, "all_losses.json"), "w") as f:
            json.dump(all_losses, f, indent=2)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump({str(k): v for k, v in all_metrics.items()}, f, indent=2)

    if args.no_figures:
        print(f"Done. Results written to {run_dir}")
        return

    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.cm.viridis
    for k, (gen, losses_g) in enumerate(sorted(all_losses.items())):
        ax.plot(losses_g, lw=1.2, color=cmap(k / max(len(all_losses) - 1, 1)))
    ax.set_xlabel("Epoch (within each generation)")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss per generation (dark to light: generation 0 to N)")
    save_figure(fig, f"{prefix}_all_losses")

    # Evolution of the samples across generations
    if args.figure_gens:
        gens_to_show = [g for g in args.figure_gens if g in synthetic]
    else:
        candidates = sorted({0, 1, args.generations // 4, args.generations // 2,
                             3 * args.generations // 4, args.generations})
        gens_to_show = [g for g in candidates if g in synthetic]

    ncols = len(gens_to_show) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.4))
    axes = np.atleast_1d(axes).flatten()
    axes[0].scatter(real_np[:, 0], real_np[:, 1], c=REF_COLOR, **SCATTER_KW)
    axes[0].set_title("Reference dataset", fontsize=TITLE_SIZE)
    style_axis(axes[0], xlim, ylim)
    for i, g in enumerate(gens_to_show):
        s = synthetic[g].numpy()
        axes[i + 1].scatter(s[:, 0], s[:, 1], c=GEN_COLOR, **SCATTER_KW)
        axes[i + 1].set_title(f"Generation {g}", fontsize=TITLE_SIZE)
        style_axis(axes[i + 1], xlim, ylim)
        axes[i + 1].set_yticklabels([])
    fig.suptitle("Spiral", fontsize=TITLE_SIZE)
    fig.tight_layout()
    save_figure(fig, f"{prefix}_collapse_evolution")

    # Metrics across generations
    gens = sorted(all_metrics)
    panels = [
        ("Frechet 2D\n(no Inception, 2-dim)", [all_metrics[g]["frechet_distance"] for g in gens]),
        ("Wasserstein-2", [all_metrics[g]["W2"] for g in gens]),
        ("Variance ratio\n(generated / real)", [all_metrics[g]["var_ratio"] for g in gens]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (title, values) in zip(axes, panels):
        ax.plot(gens, values, color=CURVE_COLOR, lw=CURVE_LW)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    fig.suptitle("Spiral", fontsize=TITLE_SIZE)
    fig.tight_layout()
    save_figure(fig, f"{prefix}_metrics_over_generations")

    # Final loss per generation
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(gens, [all_losses[g][-1] for g in gens], color=CURVE_COLOR, lw=CURVE_LW)
    ax.set_title("Final loss per generation")
    ax.set_xlabel("Generation")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    fig.suptitle("Spiral", fontsize=TITLE_SIZE)
    fig.tight_layout()
    save_figure(fig, f"{prefix}_final_loss_per_generation")

    print(f"Done. Results written to {run_dir}\nFigures: {fig_dir}")


if __name__ == "__main__":
    main()
