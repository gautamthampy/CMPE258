# CMPE 258 Deep Learning Homework Submission

This repository contains four new PyTorch tasks added in the style of the CoderGym `pytorch_task_v1` protocol.

## Structure

- `MLtasks/ml_tasks.json`: task definitions and protocol metadata
- `MLtasks/tasks/cnn_lvl2b_fashionmnist_adamw/task.py`
- `MLtasks/tasks/ae_lvl2b_conv_cifar_denoise/task.py`
- `MLtasks/tasks/rnn_lvl2b_gru_imdb_sentiment/task.py`
- `MLtasks/tasks/mlp_lvl2b_mnist_mixup_labelsmooth/task.py`

## Requirements

- Python 3.10+
- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `scikit-learn`

## Run

From `MLtasks/`:

```bash
python tasks/cnn_lvl2b_fashionmnist_adamw/task.py
python tasks/ae_lvl2b_conv_cifar_denoise/task.py
python tasks/rnn_lvl2b_gru_imdb_sentiment/task.py
python tasks/mlp_lvl2b_mnist_mixup_labelsmooth/task.py
```

Each script trains, evaluates on train and validation splits, prints metrics, asserts quality thresholds, saves artifacts under `MLtasks/output/tasks/<task_id>/`, and exits with `0` on success or non-zero on failure.

## New Tasks Summary

1. `cnn_lvl2b_fashionmnist_adamw`: FashionMNIST CNN with AdamW and cosine annealing.
2. `ae_lvl2b_conv_cifar_denoise`: Convolutional denoising autoencoder on CIFAR-10.
3. `rnn_lvl2b_gru_imdb_sentiment`: GRU sentiment classifier with gradient clipping on a synthetic IMDB-like corpus.
4. `mlp_lvl2b_mnist_mixup_labelsmooth`: MNIST MLP with mixup, label smoothing, and learning-rate warmup.