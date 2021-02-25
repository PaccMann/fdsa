"""Implementation of Set autoencoders encoder."""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from paccmann_sets.utils.hyperparameters import RNN_CELL_FACTORY


class EncoderSetsAE(nn.Module):
    """Encoder Implementation of Sets Autoencoder"""

    def __init__(self, **params) -> None:
        """Constructor.

        Args:
            params(dict): A json file containing hyperparameters for the
                encoder. Example keys are:
                cell (str): Recurrent cell to be used as the encoder in the set
                    autoencoder. Defaults to 'pLSTM'.
                input_size (int): Number of input features. Defaults to 128.
                hidden_size_linear (int): Number of hidden units in the linear
                    layer. Defaults to 256.
                hidden_size_encoder (int): Number of hidden units in the encoder.
                    Defaults to 256.
            
        """
        super(EncoderSetsAE, self).__init__()

        self.input_size = params.get('input_size', 128)
        self.hidden_size_linear = params.get('hidden_size_linear', 256)
        self.hidden_size_encoder = params.get('hidden_size_encoder', 256)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cell = params.get('cell', 'pLSTM')

        self.rnn = RNN_CELL_FACTORY[
            self.cell](self.hidden_size_linear, self.hidden_size_encoder)

        # Maps each element of a set to a memory slot.
        self.memory_mapping = nn.Linear(self.input_size, self.hidden_size_linear)

    def forward(self, data: torch.Tensor, states: Tuple = None) -> Tuple:
        """Generates encoding for sets.

        Args:
            data (torch.Tensor): Padded data matrix of shape
                [batch_size, sequence_length, self.input_size].
            states (Tuple): Tuple of the initial hidden state and cell state
                tensors, intialised at 0s.
        Returns:
            Tuple: A tuple containing the cell state, hidden state
                and read vector after the last element has been processed.
        """

        batch_size, sequence_length, _ = data.size()

        read_vector = torch.zeros((batch_size, self.hidden_size_encoder)
                                  ).to(self.device)

        if states is None:
            if 'LSTM' in self.cell:

                hidden_state, cell_state = torch.zeros_like(
                    read_vector
                ), torch.zeros_like(read_vector)

            else:

                hidden_state = torch.zeros_like(read_vector)

        memory_slots = self.memory_mapping(data)

        for _ in range(sequence_length):

            if 'LSTM' in self.cell:
                new_hidden_state, new_cell_state = self.rnn(
                    read_vector, (hidden_state, cell_state)
                )

                cell_state = new_cell_state

            else:
                new_hidden_state = self.rnn(read_vector, hidden_state)

            scalar_scores = torch.einsum('abc,ac->ab', (memory_slots, new_hidden_state))

            attention_weights = torch.softmax(scalar_scores, dim=1)

            if _ == 0:
                assert scalar_scores.size() == torch.Size(
                    [batch_size, sequence_length]
                ), 'Incorrect dimensions.'
                assert np.allclose(
                    np.sum(attention_weights.detach().cpu().numpy()), batch_size
                ), 'Weights for each set do not sum to 1.'

            read_vector = torch.einsum('ab,abc->ac',
                                       (attention_weights, memory_slots)).to(
                                           self.device
                                       )

            hidden_state = new_hidden_state

        if 'LSTM' in self.cell:
            return cell_state, hidden_state, read_vector
        else:
            return hidden_state, read_vector
