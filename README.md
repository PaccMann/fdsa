[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Fully Differentiable Set Autoencoder (fdsa)

A fully differentiable set autoencoder for encoding sets inspired by ["The Set Autoencoder: Unsupervised Representation Learning for Sets "](https://openreview.net/forum?id=r1tJKuyRZ). The model makes use of an
encoder from ["Order Matters: Sequence to sequence for sets"](https://arxiv.org/abs/1511.06391) and the decoder is a slightly modified version of the one in ["The Set Autoencoder: Unsupervised Representation Learning for Sets "](https://openreview.net/forum?id=r1tJKuyRZ). To efficiently match the reconstructions of the autoencoder to their corresponding inputs to create a differentiable loss function, three architectures were developed and evaluated that could approximate the assignment problem and thus act as an end-to-end
set matching network. The package includes code for these networks as well as baseline implementations of the set autoencoder fitted with the Hungarian matching algorithm and the Gale-Shapley algorithm.

## Installation

Create a conda environment:

```console
conda env create -f conda.yml
```

Activate the environment:

```console
conda activate fdsa
```

Install:

```console
pip install .
```

### development

Install in editable mode for development:

```sh
pip install --user -e .
```

## Examples

For some examples on how to use `fdsa` see [here](./examples)

## Citation

If you use `fdsa` in your projects, please cite as:
```bib
Nikita Janakarajan, Jannis Born, and Matteo Manica. 2022. A Fully Differ-entiable Set Autoencoder. InProceedings of the 28th ACM SIGKDD Confer-ence on Knowledge Discovery and Data Mining (KDD ’22), August 14–18,2022, Washington, DC, USA.ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3534678.3539153
```