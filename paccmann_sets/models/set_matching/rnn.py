import torch
import torch.nn as nn

from paccmann_sets.utils.hyperparameters import RNN_CELL_FACTORY
from paccmann_sets.utils.hyperparameters import RNN_FACTORY
from paccmann_sets.utils.hyperparameters import ACTIVATION_FN_FACTORY
from brc_pytorch.layers import MultiLayerBase
from paccmann_sets.utils.layers.select_item import SelectItem


class RNNSetMatching(nn.Module):
    """Generalisable RNN module to allow for flexibility in architecture."""

    def __init__(
        self,
        params: dict,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        """Constructor.

        Args:
            params (dict) with the following keys:
                input_size (int): The dimension/size of the input element.
                    Defaults to 128.
                max_length (int): The maximum length of the set.
                    Defaults to 5.
                cell (str): RNN cell to use. One of 'LSTM','GRU','BRC' or
                    'nBRC'. Defaults to 'GRU'.
                layers (int) : The number of RNN layers required. Defaults to 1.
                hidden_size (int): The hidden/cell state dimensionality.
                    Defaults to 512.
                bidirectional (str): 'True' or 'False' for a bidirectional RNN.
                    Defaults to False.
                batch_first (str): 'True' or 'False' to indicate if batch_size
                    comes first or set_length. Defaults to False.
                return_sequences (str): 'True' returns hidden state at all time
                    steps form the last layer. 'False' returns the hidden state
                    of the last time step from the last layer. Defaults to True.
                fc_layers (List[(int)]): Number of fully connected layers
                    required after the RNN. Defaults to 2.
                fc_units (List[(int)]): List of hidden units for each fully
                    connected layer. Defaults to [128,5].
                fc_activation (List[(str)]): Activation function to apply after
                    each fully connected layer. See utils/hyperparameter.py
                    for options. Defaults to ['lrelu', 'None'].
            device (torch.device): Device on which the model is executed.
                Defaults to CPU.
        """
        super(RNNSetMatching, self).__init__()

        self.input_size = params.get('input_size', 128)
        self.seq_len = params.get('max_length', 5)
        self.cell = params.get('cell', 'GRU')
        self.rnn_layers = params.get('layers', 1)
        self.hidden_size = params.get('hidden_size', 512)
        self.bidirectional = eval(params.get('bidirectional', 'False'))
        self.batch_first = eval(params.get('batch_first', 'False'))
        self.return_sequences = eval(params.get('return_sequences', 'True'))
        self.fc_layers = params.get('fc_layers', 2)
        self.fc_units = params.get('fc_units', [128, 5])
        self.fc_activation = params.get('fc_activation', ['lrelu', 'None'])

        self.device = device

        modules_rnn = []
        modules_fc = []

        num_directions = 2 if self.bidirectional else 1

        if self.cell == 'BRC' or self.cell == 'nBRC':

            inner_input_dimensions = num_directions * self.hidden_size

            recurrent_layers = [
                RNN_CELL_FACTORY[self.cell](self.input_size, self.hidden_size)
            ]

            for _ in range(self.rnn_layers - 1):
                recurrent_layers.append(
                    RNN_CELL_FACTORY[self.cell]
                    (inner_input_dimensions, self.hidden_size)
                )

            rnn = MultiLayerBase(
                mode=self.cell,
                cells=recurrent_layers,
                hidden_size=self.hidden_size,
                batch_first=self.batch_first,
                bidirectional=self.bidirectional,
                return_sequences=self.return_sequences,
                device=self.device
            )

            modules_rnn.append(rnn)

            if self.return_sequences:
                modules_rnn.append(SelectItem(0, -self.seq_len, self.batch_first))

            if self.fc_layers is not None:
                if self.bidirectional:
                    hidden_units = [2 * self.hidden_size] + self.fc_units
                else:
                    hidden_units = [self.hidden_size] + self.fc_units

                for layer in range(self.fc_layers):
                    modules_fc.append(
                        nn.Linear(hidden_units[layer], hidden_units[layer + 1])
                    )
                    if self.fc_activation[layer] != 'None':

                        modules_fc.append(
                            ACTIVATION_FN_FACTORY[self.fc_activation[layer]]
                        )

        else:
            # need to subset if only the last time step of last layer is needed
            rnn = RNN_FACTORY[self.cell](
                self.input_size,
                self.hidden_size,
                num_layers=self.rnn_layers,
                batch_first=self.batch_first,
                bidirectional=self.bidirectional,
            )

            modules_rnn.append(rnn)

            if self.return_sequences:
                modules_rnn.append(SelectItem(0, -self.seq_len, self.batch_first))
            else:
                modules_rnn.append(SelectItem(0, -1, self.batch_first))

            if self.fc_layers is not None:
                if self.bidirectional:
                    hidden_units = [2 * self.hidden_size] + self.fc_units
                else:
                    hidden_units = [self.hidden_size] + self.fc_units

                for layer in range(self.fc_layers):
                    modules_fc.append(
                        nn.Linear(hidden_units[layer], hidden_units[layer + 1])
                    )
                    if self.fc_activation[layer] != 'None':
                        modules_fc.append(
                            ACTIVATION_FN_FACTORY[self.fc_activation[layer]]
                        )

        self.rnn = nn.Sequential(*modules_rnn)
        self.fc = nn.Sequential(*modules_fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes input through specified network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, *,fc_units[-1]].
        """
        x = self.rnn(x)
        x = self.fc(x)

        return x
