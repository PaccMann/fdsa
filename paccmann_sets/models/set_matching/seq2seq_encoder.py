from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from paccmann_sets.models.set_matching.selectrnn import SelectRNN


class Seq2SeqEncoder(nn.Module):
    """Encoder component of Sequence to Sequence model."""

    def __init__(self, params: dict, device: torch.device) -> None:
        """Constructor.

        Args:
            params (dict): Dictionary containing all the parameters necessary
                for the encoder. Example:
                cell (str): The RNN cell to use as a decoder.
                input_size (int): The dimension/size of the input element.
                max_length (int): The maximum length of the set.
                hidden_size (int): The hidden/cell state dimensionality.
                bidirectional (str): True if RNN should be bidirectional.
            device (torch.device): Device on which operations are run.
        """

        super(Seq2SeqEncoder, self).__init__()
        self.parameters = deepcopy(params)
        self.parameters.update({'max_length': 2 * params['max_length'] + 1})
        self.parameters.update({'bidirectional': 'True'})
        self.model = SelectRNN(self.parameters, device)()

    def forward(self, x: torch.Tensor) -> Tuple:
        """Forward pass of encoder.

        Args:
            x (torch.Tensor): Input tensor to be encoded.
                Shape: [input_length, batch_size, dim]. The tensor passed as
                input is the concatenation of the two sets connected by a unit
                length tensor of the same batch size and dim.

        Returns:
            Tuple: Tuple of the hidden state from all steps and hidden state
                from the last step only.
        """

        return self.model(x)
