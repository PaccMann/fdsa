from typing import Tuple

# import lap
import numpy as np
import torch
import torch.nn as nn
from fdsa.utils.gale_shapley import GaleShapley
from scipy.optimize import linear_sum_assignment


class MapperSetsAE(nn.Module):
    """Mapping Algorithm for Sets AutoEncoder."""

    def __init__(
        self,
        matcher='HM',
        p: int = 2,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        """Constructor.

        Args:
            matcher (string): The matching algorithm to use.
                One of 'HM' (Munkres version of the Hungarian algorithm),
                or 'GS' (Gale-Shapley algorithm).
                Defaults to 'HM'.
            p (int, optional): the p-norm to use when calculating the
                cost matrix. Defaults to 2.
            device (torch.device): Device on which to run the model.
                Defaults to CPU.
        """
        super(MapperSetsAE, self).__init__()
        self.p = p
        self.matcher = matcher
        self.method = dict(
            {
                'HM': self.get_assignment_matrix_hm,
                'GS': self.get_assignment_matrix_gs
            }
        )

        self.device = device

    def get_assignment_matrix_hm(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """Runs the Munkres version of the Hungarian algorithm.

        Args:
            cost_matrix (torch.Tensor): A 2-D tensor that represents the cost
                of matching a row (input) and column (output). Has dimensions
                N x M, where N is the length of inputs and M the length of
                outputs.

        Returns:
            Tuple: Tuple of 2-D binary matrix with the same dimensions as the
                cost matrix, where 1 represents a match and 0 otherwise, and
                row-wise nonzero indices of the matrix.
        """
        matrix = torch.zeros_like(cost_matrix)
        rows, cols = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        matrix[rows, cols] = 1

        return torch.as_tensor(matrix), cols

    # def get_assignment_matrix_vj(
    #     self, cost_matrix: torch.Tensor
    # ) -> torch.Tensor:
    #     """Runs the Jonker-Volgenant algorithm.

    #     Args:
    #         cost_matrix (torch.Tensor): A 2-D tensor that represents the cost
    #             of matching a row (input) and column (output). Has dimensions
    #             N x M, where N is the length of inputs and M the length of
    #             outputs.

    #     Returns:
    #         Tuple: Tuple of 2-D binary matrix with the same dimensions as the
    #             cost matrix, where 1 represents a match and 0 otherwise, and
    #             row-wise nonzero indices of the matrix.
    #     """
    #     matrix = torch.zeros_like(cost_matrix)
    #     cost, cols, rows = lap.lapjv(
    #         cost_matrix.detach().cpu().numpy(), extend_cost=True
    #     )
    #     matrix[range(len(cols)), cols] = 1

    #     return torch.as_tensor(matrix), cols

    def get_assignment_matrix_gs(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """Runs the Gale-Shapley Stable Marriage algorithm.

        Args:
            cost_matrix (torch.Tensor): A 2-D tensor that represents the cost
                of matching a row (input) and column (output). Has dimensions
                N x M, where N is the length of inputs and M the length of
                outputs.

        Returns:
            Tuple: Tuple of 2-D binary matrix with the same dimensions as the
                cost matrix, where 1 represents a match and 0 otherwise, and
                row-wise nonzero indices of the matrix.
        """

        gs = GaleShapley(cost_matrix.size()[0], cost_matrix.size()[1])
        binary_matrix = gs.compute(cost_matrix)
        rows, cols = np.nonzero(binary_matrix)
        return binary_matrix, cols

    def output_mapping(
        self, outputs: torch.Tensor, match_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Orders the outputs based on the match matrix.

        Args:
            outputs (torch.Tensor): The set of outputs generated by the decoder.
            match_matrix (torch.Tensor):  A 2-D binary matrix, where 1
                represents a match and 0 otherwise.
                Has the same dimensions as the cost matrix.
        Returns:
            torch.Tensor: Outputs ordered in correspondence with inputs.
        """
        return (match_matrix[..., None] * outputs[None, ...]).sum(dim=1)

    def forward(
        self, inputs: torch.Tensor, stacked_outputs: torch.Tensor,
        member_probabilities: torch.Tensor
    ) -> Tuple:
        """Computes cost matrix and performs a matching between inputs and outputs.

        Args:
            inputs (torch.Tensor): Input tensor of shape
                [batch_size x sequence_length x input_size].
            stacked_outputs (torch.Tensor): Reconstructed elements from the
                decoder with shape [batch_size, max_length, input_size].
            member_probabilities (torch.Tensor): Probabilities describing the
                likelihood of elements belonging to the set.

        Returns:
            Tuple: Tuple of the outputs and their membership probabilities
                reordered with respect to the input.
        """

        in_batch_size, input_length, input_size = inputs.size()
        out_batch_size, output_length, output_size = stacked_outputs.size()
        mapped_outputs = []

        with torch.no_grad():
            cost_matrices = list(
                map(torch.cdist, inputs, stacked_outputs, [self.p] * in_batch_size)
            )

            match_matrices, cols = map(
                list, zip(*map(self.method[self.matcher], cost_matrices))
            )

        mapped_outputs = list(map(self.output_mapping, stacked_outputs, match_matrices))

        mapped_prob = list(
            map(self.output_mapping, member_probabilities, match_matrices)
        )

        return torch.stack(mapped_outputs), torch.stack(mapped_prob), np.stack(cols)
