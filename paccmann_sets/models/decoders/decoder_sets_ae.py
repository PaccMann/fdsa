from typing import Tuple

import torch
import torch.nn as nn
from paccmann_sets.utils.hyperparameters import RNN_CELL_FACTORY


class DecoderSetsAE(nn.Module):
    """Decoder from Sets AutoEncoder."""

    def __init__(self, **params) -> None:
        """Constructor.

        Args:
            params (dict): A dictionary containing parameters required
                to build the decoder. Example:
                hidden_size_encoder(int): Hidden state dimension of the encoder.
                    Defaults to 256.
                input_size(int): Input feature size. Defaults to 128.
                hidden_size_decoder(int): Hidden state dimension of the decoder.
                    Defaults to 256.
                loss(str): Loss function to optimise. Defaults to 'CrossEntropy'.
                cell(str): Recurrent cell type to use as a decoder. Defaults to
                    'pLSTM'.
        """
        super(DecoderSetsAE, self).__init__()

        self.input_size = params.get('hidden_size_encoder', 256)
        self.output_dim = params.get('input_size', 128)
        self.hidden_size_decoder = params.get('hidden_size_decoder', 256)
        self.loss = params.get('loss', 'CrossEntropy')
        self.cell = params.get('cell', 'pLSTM')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rnn = RNN_CELL_FACTORY[self.cell
                                    ](self.input_size, self.hidden_size_decoder)

        self.output_layer = nn.Linear(self.hidden_size_decoder, self.output_dim)

        if self.loss == 'CrossEntropy':
            self.prob_layer = nn.Linear(self.hidden_size_decoder, 2)

        elif self.loss == 'BCELogits':
            self.prob_layer = nn.Linear(self.hidden_size_decoder, 1)

    def forward(self, encoder_output: Tuple, length: int, p: int = 2) -> Tuple:
        """Executes batch processing of the decoder.

        Args:
            encoder_output (Tuple): Tuple of cell_state,hidden_state and
                read_vector from the last step of the encoder, all having shape
                [batch_size x hidden_size].
            length (int): Maximum sequence length of the current batch.
            p (int, optional): the p-norm to use when calculating the
                cost matrix. Defaults to 2.

        Returns:
            Tuple: A tuple containing the mapped outputs and their member
                probabilities.
        """

        if 'LSTM' in self.cell:
            cell_state, hidden_state, read_vector = encoder_output
        else:
            hidden_state, read_vector = encoder_output

        stacked_outputs = []
        member_probabilities = []

        read_vector0 = torch.zeros_like(read_vector)

        for i in range(length):

            if 'LSTM' in self.cell:
                new_hidden, new_cell = self.rnn(
                    read_vector.to(self.device),
                    (hidden_state.to(self.device), cell_state.to(self.device))
                )

                cell_state = new_cell

            else:
                new_hidden = self.rnn(
                    read_vector.to(self.device), hidden_state.to(self.device)
                )

            output = self.output_layer(new_hidden)
            member_probability = self.prob_layer(new_hidden)

            stacked_outputs.append(output)
            member_probabilities.append(member_probability)

            hidden_state = new_hidden
            read_vector = read_vector0

        stacked_outputs = torch.stack(stacked_outputs).permute(1, 0, 2)
        member_probabilities = torch.stack(member_probabilities).permute(1, 0, 2)

        return stacked_outputs, member_probabilities
