import torch
import torch.nn as nn


class SetAELoss(nn.Module):
    """Class to navigate between using a cross-entropy loss for L^{eos} or the binary
        logits loss from PyTorch."""

    def __init__(
        self,
        loss: str,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        """Constructor.

            Args:
                loss (str): Loss function to optimise.
                device (torch.device): Device on which the model should run.
                    Defaults to CPU.
            """
        self.loss = loss
        self.device = device
        self.loss_dict = {'CrossEntropy': self.ce_loss, 'BCELogits': self.bce_loss}
        if loss not in self.loss_dict.keys():
            raise NameError(
                f'Invalid loss ({loss}). Choose one from {self.loss_dict.keys()}'
            )

    def rmse_loss(
        self, inputs: torch.Tensor, mapped_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Computes the similarity loss using RMSE.

            Args:
                inputs (torch.Tensor): Input tensor of shape
                    [batch_size x sequence_length x input_size].
                mapped_outputs (torch.Tensor): Outputs ordered in correspondence with
                    the inputs.

            Returns:
                torch.Tensor: Similarity loss value for the given batch.
            NOTE: loss assumes batch_first is True.
            """
        loss_sim = nn.MSELoss()
        loss_similarity = torch.sqrt(loss_sim(inputs, mapped_outputs))

        return loss_similarity

    def ce_loss(
        self, member_probabilities: torch.Tensor, batch_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Computes the cross-entropy loss function for member probability.

            Args:
                member_probabilities (torch.Tensor): Probability that an output is a
                    member of the set.
                batch_lengths (torch.Tensor): Tensor of lengths of each set in the
                    batch.

            Returns:
                torch.Tensor: L^{eos} value for the given batch.
            NOTE: loss assumes batch_first is True.
            """
        batch_size, length, dim = member_probabilities.size()

        loss_mem = nn.CrossEntropyLoss()
        true_probabilities = torch.ones_like(
            member_probabilities[:, :, 0], dtype=torch.int64
        )
        mask = torch.zeros(
            batch_size, length + 1, dtype=true_probabilities.dtype, device=self.device
        )

        mask[torch.arange(batch_size), batch_lengths] = 1
        mask = mask.cumsum(dim=1)[:, :-1]
        true_probabilities = true_probabilities * (1 - mask)

        member_probabilities = member_probabilities.permute(0, 2, 1)
        loss_member = loss_mem(member_probabilities, true_probabilities)

        return loss_member

    def bce_loss(
        self, member_probabilities: torch.Tensor, batch_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Computes the cross-entropy loss function for member probability using the
            BCELogits Loss function.

            Args:
                member_probabilities (torch.Tensor): Probability that an output is
                    a member of the set.
                batch_lengths (torch.Tensor): Tensor of lengths of each set in the
                    batch.

            Returns:
                torch.Tensor: L^{eos} value for the given batch.
            NOTE: loss assumes batch_first is True.
            """
        batch_size, length, dim = member_probabilities.size()

        loss_mem = nn.BCEWithLogitsLoss()
        true_probabilities = torch.ones_like(member_probabilities.squeeze_())
        mask = torch.zeros(
            batch_size, length + 1, dtype=true_probabilities.dtype, device=self.device
        )

        mask[torch.arange(batch_size), batch_lengths] = 1
        mask = mask.cumsum(dim=1)[:, :-1]

        true_probabilities = true_probabilities * (1 - mask)
        loss_member = loss_mem(member_probabilities, true_probabilities)

        return loss_member

    def forward(
        self, inputs: torch.Tensor, mapped_outputs: torch.Tensor,
        member_probabilities: torch.Tensor, batch_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Computes the total loss by summing similarity and member probability losses.


            Args:
                inputs (torch.Tensor): Input tensor of shape
                    [batch_size x sequence_length x input_size].
                mapped_outputs (torch.Tensor): Outputs ordered in correspondence
                    with inputs.
                member_probabilities (torch.Tensor): Probability that an output is
                    a member of the set.
                batch_lengths (torch.Tensor): Tensor of lengths of each set in the
                    batch.

            Returns:
                torch.Tensor: Loss value for the given batch.
            NOTE: loss assumes batch_first is True.
            """
        similarity = self.rmse_loss(inputs, mapped_outputs)
        membership = self.loss_dict[self.loss](member_probabilities, batch_lengths)
        return similarity + membership
