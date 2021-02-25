from typing import Tuple

import torch
import torch.nn as nn
from paccmann_sets.models.decoders.decoder_sets_ae import DecoderSetsAE
from paccmann_sets.models.encoders.encoder_sets_ae import EncoderSetsAE


class SetsAE(nn.Module):

    def __init__(
        self,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
        **params
    ) -> None:
        """Constructor.

        Args:
            device (torch.device): Device on which to run the model.
                Defaults to CPU.
            params (dict): Dictionary of parameters to pass into the encoder and
                decoder. See EncoderSetsAE and DecoderSetsAE for examples.
        """
        super().__init__()
        self.encoder = EncoderSetsAE(**params)

        self.decoder = DecoderSetsAE(**params)

    def forward(
        self, inputs: torch.Tensor, max_length: int, batch_lengths: torch.Tensor
    ) -> Tuple:
        """Forward pass of the Set AutoEncoder.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, max_length, dim]
            max_length (int): Maximum set length of the curent input tensor.
            batch_lengths (torch.Tensor): Lengths of all sets in the batch.

        Returns:
            Tuple: Tuple containing the predicted outputs and its probabilities.
        """
        encoder_output = self.encoder(inputs)

        output, probablities = self.decoder(encoder_output, max_length)

        return output, probablities
