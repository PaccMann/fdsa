import torch
import torch.nn as nn
from brc_pytorch.layers import MultiLayerBase
from fdsa.utils.hyperparameters import RNN_CELL_FACTORY, RNN_FACTORY


class SelectRNN(nn.Module):
    """Allows to switch between torch implemented GRU/LSTM and in-house
        implementation of multilayer BRC/nBRC"""

    def __init__(
        self,
        params: dict,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        """Constructor.

        Args:
            params (dict): Dictionary of parameters necessary for the RNN.
            Example:
                input_size (int): The dimension/size of the input element.
                    Defaults to 128.
                max_length (int): The maximum length of the set. Defaults to 5.
                cell (str): The RNN cell to use as a decoder. Defaults to GRU.
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
            device (torch.device): Device on which operations are run.
                Defaults to CPU.
        """

        super(SelectRNN, self).__init__()

        self.input_size = params.get('input_size', 128)
        self.seq_len = params.get('max_length', 5)
        self.cell = params.get('cell', 'GRU')
        self.rnn_layers = params.get('layers', 1)
        self.hidden_size = params.get('hidden_size', 512)
        self.bidirectional = eval(params.get('bidirectional', 'False'))
        self.batch_first = eval(params.get('batch_first', 'False'))
        self.return_sequences = eval(params.get('return_sequences', 'True'))
        self.device = device

        num_directions = 2 if self.bidirectional else 1

        if self.cell == 'BRC' or self.cell == 'nBRC':

            recurrent_layers = [
                RNN_CELL_FACTORY[self.cell](self.input_size, self.hidden_size)
            ]

            inner_input_dimensions = num_directions * self.hidden_size

            for _ in range(self.rnn_layers - 1):
                recurrent_layers.append(
                    RNN_CELL_FACTORY[self.cell]
                    (inner_input_dimensions, self.hidden_size)
                )

            self.rnn = MultiLayerBase(
                mode=self.cell,
                cells=recurrent_layers,
                hidden_size=self.hidden_size,
                batch_first=self.batch_first,
                bidirectional=self.bidirectional,
                return_sequences=self.return_sequences,
                device=self.device
            )

        else:

            self.rnn = RNN_FACTORY[self.cell](
                self.input_size,
                self.hidden_size,
                num_layers=self.rnn_layers,
                batch_first=self.batch_first,
                bidirectional=self.bidirectional
            )

    def forward(self):
        """Returns the chosen model."""
        return self.rnn
