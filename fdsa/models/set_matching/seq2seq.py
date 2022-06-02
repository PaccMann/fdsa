from typing import Tuple

import torch
import torch.nn as nn
from fdsa.models.set_matching.seq2seq_decoder import Seq2SeqDecoder
from fdsa.models.set_matching.seq2seq_encoder import Seq2SeqEncoder


class Seq2Seq(nn.Module):
    """Complete Sequence to Sequence Model"""

    def __init__(
        self,
        params: dict,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """Constructor.

        Args:
            params (dict): Dictionary containing all the parameters necessary
                for seq2seq encoder and decoder. See Seq2SeqEncoder and
                Seq2SeqDecoder for more details.
            device (torch.device): Device on which operations are run.
                Defaults to CPU.
        """

        super(Seq2Seq, self).__init__()

        self.device = device
        self.max_length = params.get('max_length', 5)
        self.cell = params.get('cell', 'GRU')
        self.batch_first = eval(params.get('batch_first', 'False'))

        self.encoder = Seq2SeqEncoder(params, self.device)
        self.decoder = Seq2SeqDecoder(params, self.device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple:
        """Computes output and attention weights of seq2seq model.

        Args:
            x (torch.Tensor): Input tensor to be encoded. The input set for
                matching is the concatenation of the two sets to be matched with
                a connecting token. Of the two sets, one is deemed to be the
                reference set while the other set is reordered accordingly.
                Shape : [2*set_length+1,batch_size,dim]
            y (torch.Tensor): The reference set for aligning the other set.
                Shape : [set_length,batch_size,dim]

        Returns:
            Tuple: Tuple of the outputs associated with each element of the
                reference set and attention weights used to compute outputs.
        NOTE: This model always assumes batch_first = False. 
        TODO: Make this class, and subsequently the encoder and decoder classes
            flexible to handle batch_first = True as well.
        """

        length, batch_size, dim = x.size()

        outputs = torch.zeros(self.max_length, batch_size,
                              self.max_length).to(self.device)

        attention_weights = torch.zeros(
            self.max_length, dim, batch_size, self.max_length
        ).to(self.device)

        encoder_outputs, encoder_hn = self.encoder(x)

        decoder_hidden = encoder_hn

        for i in range(y.size(0)):

            output, hidden_state, attn_wts = self.decoder(
                y[i], decoder_hidden, encoder_outputs
            )

            outputs[i] = output
            attention_weights[i] = attn_wts
            decoder_hidden = hidden_state

        return outputs, attention_weights
