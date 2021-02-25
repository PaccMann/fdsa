import math
from typing import Tuple

import torch
import torch.nn as nn


class PeepholeLSTMCell(nn.Module):
    """LSTM with peephole connections."""

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = True, *args, **kwargs
    ) -> None:
        """Constructor.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (dict): Number of hidden units in linear and
                encoder layers.
            bias (bool): Whether to include bias. Defaults to True.
        """
        super(PeepholeLSTMCell, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.weights_x = nn.Parameter(
            torch.FloatTensor(self.input_size, self.hidden_size * 4)
        )
        self.weights_h = nn.Parameter(
            torch.FloatTensor(self.hidden_size, self.hidden_size * 4)
        )
        self.weights_c = nn.Parameter(
            torch.FloatTensor(self.hidden_size, self.hidden_size * 3)
        )

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.hidden_size * 4))
        else:
            self.bias = torch.zeros((self.hidden_size * 4))

        self.init_params()

    def init_params(self) -> None:
        """Uniform Xavier initialisation of weights."""

        std_dev = math.sqrt(1 / self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std_dev, std_dev)

    def init_hidden(self, batch_size: int) -> Tuple:
        """Initialise hidden and cell states.

        Args:
            batch_size (int) : batch size used for training.

        Returns:
            Tuple: a tuple containing the hidden state and cell state
                both initialized to zeros.
        """
        hidden_state = torch.FloatTensor(torch.zeros((batch_size, self.hidden_size))
                                         ).to(self.device)
        cell_state = torch.FloatTensor(torch.zeros((batch_size, self.hidden_size))
                                       ).to(self.device)

        return hidden_state, cell_state

    def forward(self, data_t: torch.Tensor, states: Tuple) -> Tuple:
        """Single LSTM cell.

        Args:
            data_t (torch.Tensor): Element at step t with shape
                [batch_size, self.input_size]
            states (Tuple): Tuple of the internal states of the recurrent cell:
                hidden_state (torch.Tensor): Hidden state of the LSTM cell of
                    shape [batch_size, self.hidden_size]
                cell_state (torch.Tensor): Cell state of the LSTM cell of shape
                    [batch_size, self.hidden_size]

        Returns:
            Tuple: A tuple containing the hidden state and cell state
                after the element is processed.
        """
        hidden_state, cell_state = states

        linear_xh = torch.matmul(data_t, self.weights_x) + torch.matmul(
            hidden_state, self.weights_h
        ) + self.bias

        linear_cxh = linear_xh[:, :self.hidden_size * 2] + torch.matmul(
            cell_state, self.weights_c[:, :self.hidden_size * 2]
        )

        forget_prob = torch.sigmoid(linear_cxh[:, :self.hidden_size])

        input_prob = torch.sigmoid(linear_cxh[:, self.hidden_size:self.hidden_size * 2])

        candidates = torch.tanh(linear_xh[:, self.hidden_size * 2:self.hidden_size * 3])

        new_cell_state = forget_prob * cell_state + input_prob * candidates

        linear_output = (
            linear_xh[:, self.hidden_size * 3:self.hidden_size * 4] + torch.matmul(
                new_cell_state,
                self.weights_c[:, self.hidden_size * 2:self.hidden_size * 3]
            )
        )

        output_gate = torch.sigmoid(linear_output)

        new_hidden_state = torch.tanh(new_cell_state) * output_gate

        return new_hidden_state, new_cell_state
