"""Recursive (self-consuming) training of a diffusion model on MNIST, Fashion-MNIST or CIFAR-10.

Generation 0 is trained from scratch on `--data-points` real images. Every later
generation trains a freshly initialized model, on data selected according to the
protocol (see common.build_training_set), and produces `--data-points` synthetic
images with its EMA weights.

Example:
    python train_recursive_images.py --dataset mnist --protocol last-gen
    python train_recursive_images.py --dataset cifar10 --protocol fixed-budget
"""

import argparse
import json
import os
import time
from datetime import datetime

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from diffusion import (DiffusionSchedule, EMA, PROTOCOLS, build_training_set,  # noqa: E402
                    ddim_sample, gen_dir, get_device, set_seed)
from models import build_model  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "figure.dpi": 120})

# Per-dataset settings used for the experiments reported in the paper.
DATASETS = {
    "mnist": dict(
        torchvision_class="MNIST", channels=1, image_size=28,
        model="unet-small", model_kwargs=dict(channels=1, base_dim=32, time_dim=64),
        lr=9e-4, lr_recursive=1e-4, inference_steps=100,
    ),
    "fashion-mnist": dict(
        torchvision_class="FashionMNIST", channels=1, image_size=28,
        model="unet-small", model_kwargs=dict(channels=1, base_dim=32, time_dim=64),
        lr=9e-4, lr_recursive=1e-4, inference_steps=100,
    ),
    "cifar10": dict(
        torchvision_class="CIFAR10", channels=3, image_size=32,
        model="unet-attention", model_kwargs=dict(channels=3, base_dim=8, time_dim=32),
        lr=5e-4, lr_recursive=5e-4, inference_steps=200,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASETS), default="mnist")
    p.add_argument("--protocol", choices=PROTOCOLS, default="last-gen")
    p.add_argument("--generations", type=int, default=100,
                   help="number of generations after generation 0")
    p.add_argument("--data-points", type=int, default=10_000,
                   help="size of the real dataset and of every synthetic set")
    p.add_argument("--epochs", type=int, default=1000, help="epochs per generation >= 1")
    p.add_argument("--epochs-gen0", type=int, default=None,
                   help="epochs for generation 0 (default: same as --epochs)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ema-decay", type=float, default=0.8)
    p.add_argument("--diffusion-steps", type=int, default=1000, help="length T of the schedule")
    p.add_argument("--inference-steps", type=int, default=None,
                   help="DDIM steps (default: dataset-specific)")
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate of generation 0 (default: dataset-specific)")
    p.add_argument("--lr-recursive", type=float, default=None,
                   help="learning rate of generations >= 1 (default: dataset-specific)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="./data", help="where torchvision stores the dataset")
    p.add_argument("--out", default="./runs", help="root directory for run outputs")
    p.add_argument("--device", default="auto")
    p.add_argument("--sample-every", type=int, default=0,
                   help="also plot samples every N epochs inside a generation (0 = only at the end)")
    p.add_argument("--no-figures", action="store_true", help="skip all figures")
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="keep a checkpoint per generation (needed by evaluate_images.py --from checkpoints)")
    return p.parse_args()


def load_real_data(name, data_root, n, channels):
    cfg = DATASETS[name]
    mean_std = (0.5,) * channels
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_std, mean_std),
    ])
    dataset = getattr(datasets, cfg["torchvision_class"])(
        data_root, train=True, download=True, transform=transform)
    indices = torch.randperm(len(dataset))[:n].tolist()
    return torch.stack([dataset[i][0] for i in indices])


def to_image(x):
    """(C, H, W) in [-1, 1] -> array displayable by imshow."""
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    return x[0].numpy() if x.shape[0] == 1 else x.permute(1, 2, 0).numpy()


def plot_samples(reference, generated, title, path, ncols=8):
    n = min(len(reference), len(generated), ncols * 2)
    nrows = max(1, n // ncols)
    fig, axes = plt.subplots(nrows, 2 * ncols, figsize=(2 * ncols * 1.1, nrows * 1.2))
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].imshow(to_image(reference[i]), cmap="gray", vmin=0, vmax=1)
        axes[r, c].axis("off")
        axes[r, c + ncols].imshow(to_image(generated[i]), cmap="gray", vmin=0, vmax=1)
        axes[r, c + ncols].axis("off")
    axes[0, ncols // 2].set_title("Reference", fontsize=11)
    axes[0, ncols + ncols // 2].set_title("Generated", fontsize=11)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def train_one_generation(model, optimizer, ema, train_data, gen, args, schedule,
                         out_dir, reference):
    loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True)
    sample_shape = (DATASETS[args.dataset]["channels"],) + (DATASETS[args.dataset]["image_size"],) * 2
    losses = []

    for epoch in range(args.epochs_gen0 if gen == 0 else args.epochs):
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
        n_epochs = args.epochs_gen0 if gen == 0 else args.epochs
        print(f"  gen {gen:3d} | epoch {epoch + 1:4d}/{n_epochs} | loss {losses[-1]:.4f}", flush=True)

        last_epoch = epoch == n_epochs - 1
        intermediate = args.sample_every and (epoch + 1) % args.sample_every == 0
        if args.no_figures or not (last_epoch or intermediate):
            continue

        samples_dir = os.path.join(out_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        preview = ddim_sample(ema.ema_model, 16, sample_shape, schedule,
                              args.inference_steps, args.device)
        plot_samples(reference, preview,
                     f"generation {gen} | epoch {epoch + 1} | loss {losses[-1]:.4f}",
                     os.path.join(samples_dir, f"epoch_{epoch + 1:04d}.pdf"))

    if not args.no_figures:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(range(1, len(losses) + 1), losses)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training loss (generation {gen})")
        fig.savefig(os.path.join(out_dir, "loss_curve.pdf"), bbox_inches="tight")
        plt.close(fig)

    if args.keep_checkpoints:
        torch.save({"generation": gen,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.ema_model.state_dict(),
                    "losses": losses},
                   os.path.join(out_dir, f"checkpoint_gen{gen}.pt"))

    return losses


def main():
    args = parse_args()
    cfg = DATASETS[args.dataset]
    args.device = get_device(args.device)
    args.epochs_gen0 = args.epochs if args.epochs_gen0 is None else args.epochs_gen0
    args.inference_steps = args.inference_steps or cfg["inference_steps"]
    args.lr = args.lr or cfg["lr"]
    args.lr_recursive = args.lr_recursive or cfg["lr_recursive"]

    run_dir = os.path.join(args.out, f"{args.dataset}_{args.protocol}",
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Device: {args.device}\nRun directory: {run_dir}")

    set_seed(args.seed)
    schedule = DiffusionSchedule(args.diffusion_steps, args.device)
    sample_shape = (cfg["channels"],) + (cfg["image_size"],) * 2

    real_data = load_real_data(args.dataset, args.data_root, args.data_points, cfg["channels"])
    print(f"Real data: {tuple(real_data.shape)}")

    all_losses, synthetic = {}, {}
    reference = real_data[:16].clone()

    for gen in range(args.generations + 1):
        print("=" * 60)
        print(f"GENERATION {gen} ({args.protocol})")
        print("=" * 60)

        # Non-incremental protocol: a new model, optimizer and EMA at every generation.
        model = build_model(cfg["model"], **cfg["model_kwargs"]).to(args.device)
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
        print(f"  training set: {len(train_data):,} samples "
              f"({info['train_source']}, real fraction in pool "
              f"{info['real_fraction_in_pool']:.1%})")

        out_dir = gen_dir(run_dir, gen, create=True)
        losses = train_one_generation(model, optimizer, ema, train_data, gen, args,
                                      schedule, out_dir, reference)
        all_losses[gen] = losses

        start = time.perf_counter()
        synthetic[gen] = ddim_sample(ema.ema_model, args.data_points, sample_shape,
                                     schedule, args.inference_steps, args.device)
        torch.save(synthetic[gen], os.path.join(out_dir, "synthetic_data.pt"))
        print(f"  sampled {args.data_points} images in {time.perf_counter() - start:.1f} s")

        info.update({"generation": gen, "final_loss": losses[-1]})
        with open(os.path.join(out_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=2)

        if gen == 0:
            reference = synthetic[0][:16].clone()
        if args.protocol == "last-gen":
            synthetic.pop(gen - 1, None)  # only the previous generation is ever reused

        with open(os.path.join(run_dir, "all_losses.json"), "w") as f:
            json.dump(all_losses, f, indent=2)

    print(f"Done. Results written to {run_dir}")


if __name__ == "__main__":
    main()
