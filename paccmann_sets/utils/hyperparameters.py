"""Script containing factories for various hyperparameters"""
import torch
import torch.distributions as d
import torch.nn as nn
from brc_pytorch.layers import (
    BistableRecurrentCell, NeuromodulatedBistableRecurrentCell
)
from paccmann_sets.utils.layers.peephole_lstm import PeepholeLSTMCell
from paccmann_sets.utils.mapper import MapperSetsAE
from pytoda.datasets.utils.wrappers import WrapperCDist, WrapperKLDiv

METRIC_FUNCTION_FACTORY = {'p-norm': WrapperCDist, 'KL': WrapperKLDiv}

MAPPER_FUNCTION_FACTORY = {
    'HM': MapperSetsAE('HM').get_assignment_matrix_hm,
    'GS': MapperSetsAE('GS').get_assignment_matrix_gs
}

DISTRIBUTION_FUNCTION_FACTORY = {
    'normal': d.normal.Normal,
    'multinormal': d.multivariate_normal.MultivariateNormal,
    'beta': d.beta.Beta,
    'uniform': d.uniform.Uniform,
    'bernoulli': d.bernoulli.Bernoulli
}

RNN_CELL_FACTORY = {
    'LSTM': nn.LSTMCell,
    'GRU': nn.GRUCell,
    'BRC': BistableRecurrentCell,
    'nBRC': NeuromodulatedBistableRecurrentCell,
    'pLSTM': PeepholeLSTMCell
}

RNN_FACTORY = {'LSTM': nn.LSTM, 'GRU': nn.GRU}

ACTIVATION_FN_FACTORY = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'tanh': nn.Tanh(),
    'lrelu': nn.LeakyReLU(),
    'elu': nn.ELU(),
    'softmax1': nn.Softmax(dim=1),
    'softmax2': nn.Softmax(dim=2)
}

POOLING_FN_FACTORY = {
    'avg': nn.AvgPool2d,
    'adaptive_avg': nn.AdaptiveAvgPool2d,
    'max': nn.MaxPool2d
}

LR_SCHEDULER_FACTORY = {
    'step': torch.optim.lr_scheduler.StepLR,
    'exp': torch.optim.lr_scheduler.ExponentialLR,
    'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau
}
