# Feature Selective Model Collapse in Diffusion Models: Total Replacement versus Fixed-Budget Training

Code for the experiments of the paper *Model collapse in recursive generative
diffusion models* (anonymous submission). A diffusion model is trained on real
data, generates a synthetic dataset, a new model is trained on that synthetic
dataset, and so on for up to 100 generations. The scripts reproduce the collapse
curves and the sample grids of the paper on a 2D spiral, MNIST, Fashion-MNIST and
CIFAR-10.

## Protocols

Both protocols are *non-incremental*: at every generation a **new** model,
optimizer and EMA are initialized from scratch. The previous generation's weights
are never reused, and the only thing that carries over is the data.

| `--protocol` | Training set of generation *n* ≥ 1 | Real data after generation 0 |
|---|---|---|
| `last-gen` | the synthetic set produced by generation *n − 1* | never reused |
| `fixed-budget` | a uniform subsample of fixed size drawn from the pool D<sub>real</sub> ∪ D<sub>0</sub> ∪ … ∪ D<sub>n−1</sub> | present in the pool, expected fraction ≈ 1 / (n + 1) |

The training set has the same size at every generation (`--data-points`), so
under `fixed-budget` the pool grows while the budget does not: the real fraction
decays to zero without any data ever being deleted. Both protocols are
implemented in a single function, `build_training_set` in `diffusion.py`.

## Files

| File | Purpose |
|---|---|
| `diffusion.py` | diffusion schedule, EMA, DDIM sampler, the two protocols |
| `models.py` | the three noise predictors (two U-Nets, one MLP) |
| `train_recursive_images.py` | recursive training on MNIST, Fashion-MNIST or CIFAR-10 |
| `train_recursive_spiral.py` | recursive training on the 2D spiral, metrics computed inline |
| `evaluate_images.py` | FID, Frechet pixel and Wasserstein-2 per generation for an image run |
| `plot_samples.py` | reference / generation 0 / generation *N* comparison figure |
| `plot_metrics.py` | metric curves over generations, for image and spiral runs |
| `reproduce.sh` | every run of the paper, one line per experiment |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A GPU is not required for the spiral, but is for the image experiments.

## Quick check

A two-generation, two-epoch run finishes in under a minute on CPU and exercises
the whole pipeline:

```bash
python train_recursive_spiral.py --protocol last-gen --generations 2 --epochs 20 --data-points 200
python train_recursive_images.py --dataset mnist --protocol last-gen \
    --generations 2 --epochs 2 --data-points 512
```

## Reproducing the paper

Default hyperparameters are the ones used in the paper, so the dataset and the
protocol are the only arguments needed. `reproduce.sh` lists the eight runs, each
preceded by the name of the experiment:

```bash
python train_recursive_spiral.py --protocol last-gen
python train_recursive_spiral.py --protocol fixed-budget
python train_recursive_images.py --dataset mnist         --protocol last-gen
python train_recursive_images.py --dataset mnist         --protocol fixed-budget
python train_recursive_images.py --dataset fashion-mnist --protocol last-gen
python train_recursive_images.py --dataset fashion-mnist --protocol fixed-budget
python train_recursive_images.py --dataset cifar10       --protocol last-gen
python train_recursive_images.py --dataset cifar10       --protocol fixed-budget
```

Each run prints its output directory on the first line. Then, for an image run:

```bash
python evaluate_images.py runs/mnist_last-gen/<run_id>   # writes metrics.json
python plot_metrics.py    runs/mnist_last-gen/<run_id>
python plot_samples.py    runs/mnist_last-gen/<run_id> --gen 100
```

Spiral runs compute their metrics and write their figures during the run, so only
`plot_metrics.py` is useful afterwards.

## Hyperparameters

Shared: linear β schedule with T = 1000, deterministic DDIM sampling, Adam,
batch size 256, EMA decay 0.8, sampling always from the EMA weights.

| | spiral | MNIST / Fashion-MNIST | CIFAR-10 |
|---|---|---|---|
| network | MLP, 3 hidden layers of 256 | U-Net, base width 32 | U-Net, base width 8, self-attention at 4×4 |
| parameters | 155,522 | 339,617 | 195,211 |
| dataset size | 1,000 | 10,000 | 10,000 |
| epochs per generation | 2,000 | 1,000 | 1,000 |
| generations | 20 | 100 | 100 |
| DDIM steps | 100 | 100 | 200 |
| learning rate, generation 0 | 1e-3 | 9e-4 | 5e-4 |
| learning rate, generations ≥ 1 | 1e-3 | 1e-4 | 5e-4 |

The learning rate of generations ≥ 1 is a separate argument (`--lr-recursive`)
because those generations are cold-started; it is not a fine-tuning rate, since
no weights are inherited.

## Output layout

```
runs/<dataset>_<protocol>/<timestamp>/
    config.json               every argument of the run
    all_losses.json           loss per epoch, per generation
    metrics.json              written by evaluate_images.py (or by train_recursive_spiral.py)
    gen_00/
        synthetic_data.pt     the generation's synthetic dataset
        info.json             training set, pool size, real fraction, final loss
        loss_curve.pdf
        samples/              sample grids
        checkpoint_gen0.pt    only with --keep-checkpoints
    gen_01/
    ...
```

## Notes

- **Storage and memory.** Every generation writes its full synthetic dataset.
  For 100 generations of CIFAR-10 at 10,000 images this is about 120 GB in
  `float32`; `--keep-checkpoints` adds one checkpoint per generation. Under
  `fixed-budget` the whole pool is held in memory to be subsampled, which is the
  same order of magnitude; under `last-gen` only the previous generation is kept.
- **Runtime.** The image runs are cluster jobs: 101 generations × 1,000 epochs.
  Reduce `--generations` and `--epochs` for anything exploratory.
- **FID.** `evaluate_images.py` downloads the Inception-V3 ImageNet weights on
  first use. By default it computes the metrics on each generation's saved
  `synthetic_data.pt`; `--source checkpoints` re-samples from the saved EMA
  weights instead, and needs a run made with `--keep-checkpoints`.
- **Wasserstein-2.** Exact optimal transport is solved by linear programming,
  which is why both marginals are subsampled (`--w2-subsample`, `--ot-points`).
  Without that cap `ot.emd2` stops on its iteration limit and silently returns a
  cost that is too small.
- **Determinism.** `--seed` controls data sampling, weight initialization and the
  diffusion noise. Runs remain sensitive to GPU non-determinism in cuDNN.
