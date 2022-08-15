[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Fully Differentiable Set Autoencoder (fdsa)

A fully differentiable set autoencoder for encoding sets. [Paper @KDD 2022](https://dl.acm.org/doi/10.1145/3534678.3539153).


The work is inspired by ["The Set Autoencoder: Unsupervised Representation Learning for Sets "](https://openreview.net/forum?id=r1tJKuyRZ). The model makes use of an
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

If you use `fdsa` in your projects, please cite:


```bib
@inproceedings{10.1145/3534678.3539153,
  author = {Janakarajan, Nikita and Born, Jannis and Manica, Matteo},
  title = {A Fully Differentiable Set Autoencoder},
  year = {2022},
  isbn = {9781450393850},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3534678.3539153},
  doi = {10.1145/3534678.3539153},
  booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages = {3061–3071},
  numpages = {11},
  keywords = {set matching network, multi-modality, autoencoders, sets},
  location = {Washington DC, USA},
  series = {KDD '22}
}
```
