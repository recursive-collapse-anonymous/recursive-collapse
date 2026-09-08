#!/usr/bin/env bash
# Every run reported in the paper, one line per experiment.
# Default hyperparameters are the ones used in the paper, so the dataset and the
# protocol are the only arguments needed. Run the lines you need rather than the
# whole file: each image run is a cluster job of 101 generations x 1000 epochs.

set -e

# --- 2D spiral -------------------------------------------------------------
# spiral_last_gen
python train_recursive_spiral.py --protocol last-gen
# spiral_fixed_budget
python train_recursive_spiral.py --protocol fixed-budget

# --- MNIST -----------------------------------------------------------------
# mnist_last_gen
python train_recursive_images.py --dataset mnist --protocol last-gen
# mnist_fixed_budget
python train_recursive_images.py --dataset mnist --protocol fixed-budget

# --- Fashion-MNIST ---------------------------------------------------------
# fashion_mnist_last_gen
python train_recursive_images.py --dataset fashion-mnist --protocol last-gen
# fashion_mnist_fixed_budget
python train_recursive_images.py --dataset fashion-mnist --protocol fixed-budget

# --- CIFAR-10 --------------------------------------------------------------
# cifar10_last_gen
python train_recursive_images.py --dataset cifar10 --protocol last-gen
# cifar10_fixed_budget
python train_recursive_images.py --dataset cifar10 --protocol fixed-budget

# --- Metrics and figures for one image run ---------------------------------
# Replace <run_dir> by the directory printed at the start of the run,
# e.g. runs/mnist_last-gen/20260101_120000
#
#   python evaluate_images.py <run_dir>            # writes <run_dir>/metrics.json
#   python plot_metrics.py    <run_dir>
#   python plot_samples.py    <run_dir> --gen 100
#
# Spiral runs compute their metrics and write their figures during the run.
