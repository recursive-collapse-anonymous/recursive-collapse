"""Three collapse metrics per generation for a MNIST / Fashion-MNIST / CIFAR-10 run.

For every generation of a run directory produced by train_recursive_images.py, compares the
generated images to the real training set with:

  1. FID           Frechet distance between Gaussians fitted on Inception-V3
                   features (2048-dim). Downloads the Inception weights on first use.
  2. Frechet pixel Same formula applied directly to the flattened images.
  3. W2 pixel      Exact optimal transport between the two point clouds (POT),
                   on a subsample of --w2-subsample images per side.

For two Gaussians the Frechet distance equals the squared W2 distance between
them (Dowson & Landau, 1982); comparing the three curves shows how much of the
measured collapse depends on the representation and on the Gaussian assumption.

Results are written to <run_dir>/metrics.json, read by plot_metrics.py.

Example:
    python evaluate_images.py runs/mnist_last-gen/20260101_120000
"""

import argparse
import json
import os

import numpy as np
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torchvision import models

from diffusion import DiffusionSchedule, available_generations, ddim_sample, gen_dir, get_device
from models import build_model
from train_recursive_images import DATASETS, load_real_data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--source", choices=("samples", "checkpoints"), default="samples",
                   help="use the synthetic_data.pt of each generation, or re-sample from its checkpoint")
    p.add_argument("--generations", type=int, nargs="*", default=None,
                   help="subset of generations to evaluate (default: all available)")
    p.add_argument("--w2-subsample", type=int, default=2000,
                   help="images per side in the exact optimal transport")
    p.add_argument("--batch-size", type=int, default=64, help="batch size for Inception features")
    p.add_argument("--sampling-seed", type=int, default=42, help="only used with --source checkpoints")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def load_run_config(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        return json.load(f)


def build_inception(device):
    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inception.fc = nn.Identity()
    inception.eval()
    return inception.to(device)


@torch.no_grad()
def inception_features(inception, images, device, batch_size):
    """images: (N, C, H, W) in [-1, 1] -> (N, 2048) features."""
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        batch = (batch + 1.0) / 2.0
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        f = inception(batch)
        if isinstance(f, tuple):
            f = f[0]
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def frechet_distance(feats_a, feats_b, eps=1e-6):
    """||mu_a - mu_b||^2 + Tr(Sa + Sb - 2 (Sa Sb)^{1/2}); any feature dimension."""
    mu_a, mu_b = feats_a.mean(0), feats_b.mean(0)
    Sa = np.cov(feats_a, rowvar=False)
    Sb = np.cov(feats_b, rowvar=False)
    diff = mu_a - mu_b

    covmean, _ = sqrtm(Sa @ Sb, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(Sa.shape[0]) * eps
        covmean = sqrtm((Sa + offset) @ (Sb + offset))
    if np.iscomplexobj(covmean):
        if np.abs(covmean.imag).max() > 1e-3:
            print(f"    [warning] sqrtm imaginary part (max={np.abs(covmean.imag).max():.2e})")
        covmean = covmean.real
    return float(diff @ diff + np.trace(Sa + Sb - 2 * covmean))


def wasserstein2(X, Y, n_subsample, seed=0):
    """Exact W2 between two point clouds, subsampled to keep the LP tractable."""
    rng = np.random.default_rng(seed)
    n1, n2 = min(n_subsample, len(X)), min(n_subsample, len(Y))
    X_sub = X[rng.choice(len(X), n1, replace=False)]
    Y_sub = Y[rng.choice(len(Y), n2, replace=False)]
    a, b = np.ones(n1) / n1, np.ones(n2) / n2
    M = ot.dist(X_sub, Y_sub, metric="sqeuclidean")
    cost = ot.emd2(a, b, M, numItermax=1_000_000)
    return float(np.sqrt(max(cost, 0.0)))


def main():
    args = parse_args()
    device = get_device(args.device)
    cfg = load_run_config(args.run_dir)
    dataset = cfg["dataset"]
    spec = DATASETS[dataset]
    print(f"Run: {args.run_dir}\nDataset: {dataset} | protocol: {cfg['protocol']} | device: {device}")

    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    real_data = load_real_data(dataset, args.data_root, cfg["data_points"], spec["channels"])
    real_pixels = real_data.view(len(real_data), -1).numpy().astype(np.float64)

    inception = build_inception(device)
    print("Extracting Inception features of the real data...")
    real_feats = inception_features(inception, real_data, device, args.batch_size)

    gens = args.generations
    if gens is None:
        gens = available_generations(args.run_dir)
        if args.source == "checkpoints":
            gens = [g for g in gens
                    if os.path.exists(os.path.join(gen_dir(args.run_dir, g), f"checkpoint_gen{g}.pt"))]
    print(f"Generations to evaluate: {gens}")

    schedule = DiffusionSchedule(cfg["diffusion_steps"], device)
    sample_shape = (spec["channels"],) + (spec["image_size"],) * 2

    results = {"fid_inception": {}, "frechet_pixel": {}, "w2_pixel": {}}
    for gen in gens:
        out_dir = gen_dir(args.run_dir, gen)
        if args.source == "samples":
            samples = torch.load(os.path.join(out_dir, "synthetic_data.pt"), weights_only=False)
        else:
            ckpt = torch.load(os.path.join(out_dir, f"checkpoint_gen{gen}.pt"),
                              map_location=device, weights_only=False)
            model = build_model(spec["model"], **spec["model_kwargs"]).to(device)
            model.load_state_dict(ckpt["ema_state_dict"])
            model.eval()
            torch.manual_seed(args.sampling_seed)
            samples = ddim_sample(model, cfg["data_points"], sample_shape, schedule,
                                  cfg["inference_steps"], device)
            del model

        gen_feats = inception_features(inception, samples, device, args.batch_size)
        gen_pixels = samples.view(len(samples), -1).numpy().astype(np.float64)

        results["fid_inception"][str(gen)] = frechet_distance(real_feats, gen_feats)
        results["frechet_pixel"][str(gen)] = frechet_distance(real_pixels, gen_pixels)
        results["w2_pixel"][str(gen)] = wasserstein2(real_pixels, gen_pixels, args.w2_subsample)
        print(f"  gen {gen:3d} | FID {results['fid_inception'][str(gen)]:8.2f} "
              f"| Frechet pixel {results['frechet_pixel'][str(gen)]:9.2f} "
              f"| W2 pixel {results['w2_pixel'][str(gen)]:7.3f}", flush=True)

        del samples, gen_feats, gen_pixels
        if device == "cuda":
            torch.cuda.empty_cache()

    results["config"] = {"source": args.source, "w2_subsample": args.w2_subsample,
                         "n_samples": cfg["data_points"], "dataset": dataset,
                         "protocol": cfg["protocol"]}
    out_path = os.path.join(args.run_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics written to {out_path}")


if __name__ == "__main__":
    main()
