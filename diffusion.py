"""Shared building blocks: seeding, diffusion schedule, EMA, DDIM sampler, protocols."""

import os
from copy import deepcopy

import numpy as np
import torch

PROTOCOLS = ("last-gen", "fixed-budget")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


class DiffusionSchedule:
    """Linear beta schedule, as in Ho et al. (2020)."""

    def __init__(self, T, device, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.device = device
        self.beta = torch.linspace(beta_start, beta_end, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x0, t):
        """Forward diffusion q(x_t | x_0). Works for 2D and image tensors."""
        noise = torch.randn_like(x0)
        shape = (-1,) + (1,) * (x0.dim() - 1)
        a = self.alpha_hat[t].view(shape)
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise, noise


@torch.inference_mode()
def ddim_sample(model, n, sample_shape, schedule, num_steps, device, batch_size=16384):
    """Deterministic DDIM sampling (eta = 0), returned on CPU.

    sample_shape is the shape of a single sample, e.g. (1, 28, 28) or (2,).
    """
    timesteps = torch.linspace(schedule.T - 1, 0, num_steps, dtype=torch.long, device=device)

    all_samples = []
    remaining = n
    while remaining > 0:
        b = min(remaining, batch_size)
        x = torch.randn(b, *sample_shape, device=device)

        for i, t in enumerate(timesteps):
            t = int(t.item())
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
            eps = model(x, t_tensor)

            alpha_t = schedule.alpha_hat[t]
            x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)

            if i == len(timesteps) - 1:
                x = x0
                break

            alpha_next = schedule.alpha_hat[int(timesteps[i + 1].item())]
            x = torch.sqrt(alpha_next) * x0 + torch.sqrt(1 - alpha_next) * eps

        all_samples.append(x.cpu())
        remaining -= b

    return torch.cat(all_samples, dim=0)


class EMA:
    """Exponential moving average of the model weights; sampling always uses these."""

    def __init__(self, model, decay):
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


def build_training_set(protocol, real_data, synthetic, gen, budget):
    """Training set of generation `gen` >= 1, for one of the two protocols.

    last-gen      : train on D_{gen-1} only, the synthetic set of the previous
                    generation. Real data never re-enters the loop.
    fixed-budget  : draw `budget` samples uniformly from the pool
                    D_real u D_0 u ... u D_{gen-1}. The pool grows but the
                    training set does not, so the expected real fraction decays
                    like 1 / (gen + 1).

    Returns (train_data, info) where info is written to the generation's info.json.
    """
    if protocol == "last-gen":
        train_data = synthetic[gen - 1]
        info = {
            "train_source": f"synthetic_gen_{gen - 1}",
            "pool_size": len(train_data),
            "real_fraction_in_pool": 0.0,
        }
    elif protocol == "fixed-budget":
        pool = torch.cat([real_data] + [synthetic[g] for g in range(gen)], dim=0)
        idx = torch.randperm(len(pool))[:budget]
        train_data = pool[idx]
        info = {
            "train_source": "uniform subsample of (real + all previous synthetic)",
            "pool_size": len(pool),
            "real_fraction_in_pool": len(real_data) / len(pool),
        }
    else:
        raise ValueError(f"unknown protocol {protocol!r}, expected one of {PROTOCOLS}")

    return train_data, info


def gen_dir(run_dir, gen, create=False):
    """Directory of one generation. Accepts both gen_00 and gen_000 namings."""
    candidates = [os.path.join(run_dir, f"gen_{gen:02d}"),
                  os.path.join(run_dir, f"gen_{gen:03d}")]
    for path in candidates:
        if os.path.isdir(path):
            return path
    if create:
        os.makedirs(candidates[0], exist_ok=True)
        return candidates[0]
    return candidates[0]


def available_generations(run_dir, filename="synthetic_data.pt"):
    """Generation indices for which `filename` exists under run_dir."""
    gens = []
    for name in sorted(os.listdir(run_dir)):
        if not name.startswith("gen_"):
            continue
        if os.path.exists(os.path.join(run_dir, name, filename)):
            gens.append(int(name.split("_")[1]))
    return sorted(gens)
