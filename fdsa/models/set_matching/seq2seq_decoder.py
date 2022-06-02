from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fdsa.models.set_matching.dnn import DNNSetMatching
from fdsa.models.set_matching.selectrnn import SelectRNN


class Seq2SeqDecoder(nn.Module):
    """Decoder component of the Sequence to Sequence model."""

    def __init__(self, params: dict, device: torch.device) -> None:
        """Constructor.

        Args:
            params (dict): Dictionary containing parameters necessary for the
                decoder. Example:
                cell (str): The RNN cell to use as a decoder. Defaults to 'GRU'.
                input_size (int): The dimension/size of the input element.
                    Defaults to 128.
                max_length (int): The maximum length of the set.
                    Defaults to 5.
                hidden_size (int): The hidden/cell state dimensionality.
                    Defaults to 512.
                fc_layers (int): The number of fully connected layers.
                    Defaults to 2.
                fc_units (List(int)): Number of hidden units for each FC layer.
                    Defaults to [128,5].
                fc_activation (str): Activation function to apply on each FC
                    layer. ['lrelu', 'None'].
            device (torch.device): Device on which operations are run.
                          
        """

        super(Seq2SeqDecoder, self).__init__()

        self.cell = params.get('cell', 'GRU')
        self.input_size = params.get('input_size', 128)
        self.max_length = params.get('max_length', 5)

        self.params = deepcopy(params)

        self.hidden_size = params.get('hidden_size', 512)

        self.params_fc = dict(
            {
                'input_size': self.hidden_size,
                'fc_layers': params.get('fc_layers', 2),
                'fc_units': params.get('fc_units', [128, 5]),
                'fc_activation': params.get('fc_activation', ['lrelu', 'None'])
            }
        )

        self.params.update({'input_size': self.hidden_size, 'bidirectional': 'False'})

        self.model = SelectRNN(self.params, device)()

        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.max_length),
            nn.Softmax(1)
        )
        self.add_attention = nn.Linear(
            2 * self.hidden_size + self.input_size, self.hidden_size
        )

        self.fc = DNNSetMatching(self.params_fc)

    def forward(
        self, y: torch.Tensor, hidden_state: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> Tuple:
        """Forward pass of decoder.

        Args:
            y (torch.Tensor): Element of reference set. Shape: [1,batch_size,dim]
            hidden_state (torch.Tensor): The last hidden state from the encoder.
                Shape: [num_layers * num_directions, batch_size, hidden_size]
            encoder_outputs (torch.Tensor): All hidden states from the encoder.
                Shape:[set_length, batch_size, num_directions * hidden_size]

        Returns:
            Tuple: Tuple containing the output, new hidden state and attention
                weights used to compute the output for that element.
        """

        if 'LSTM' in self.cell:
            dec_hidden = (
                hidden_state[0][0].unsqueeze(0), hidden_state[1][0].unsqueeze(0)
            )
            hidden_state_n = hidden_state[0][-1].squeeze(0)
        else:
            dec_hidden = hidden_state[0].unsqueeze(0)
            hidden_state_n = hidden_state[-1].squeeze(0)

        attention_weights = self.attention(torch.cat((y, hidden_state_n),
                                                     1)).unsqueeze(2).permute(0, 2, 1)

        attn_applied = torch.matmul(
            attention_weights,
            encoder_outputs[-self.max_length:, :, :].permute(1, 0, 2)
        )

        output = torch.cat((y, attn_applied.squeeze_()), 1)

        output = self.add_attention(output).unsqueeze_(0)

        output = F.relu(output)

        output, hidden_state = self.model(output, dec_hidden)

        output = self.fc(output)

        return output, hidden_state, attention_weights.permute(1, 0, 2)
