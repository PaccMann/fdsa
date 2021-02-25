from typing import Tuple

import torch


class GaleShapley:
    """2D implementation of Gale-Shapley"""

    def __init__(self, input_length: int, output_length: int) -> None:
        """Constructor.

        Args:
            input_length (int): Number of elements in the first set.
            output_length (int): Number of elements in the second set.
        """

        self.inp_len = input_length
        self.out_len = output_length

    def get_ranking(self, pairwise_distance: torch.Tensor) -> Tuple:
        """Retrieve the ranks of elements in order of preference of each set in
            the set pair.

        Args:
            pairwise_distance (torch.Tensor): A 2D tensor of pairwise distances
                between the elements of the two sets.

        Returns:
            Tuple: Tuple of the ranks of elements of one set in order of
                preference of the other set, for both sets.
        """
        sorted_input, indices_input = torch.sort(pairwise_distance)
        sorted_output, indices_output = torch.sort(pairwise_distance.t())

        return indices_input, indices_output

    def compute(
        self, cost_matrix: torch.Tensor
    ) -> torch.Tensor:  # Execute for one input at a time
        """Compute the Gale-Shapley assignment matrix for the first set as the
            "proposer".

        Args:
            cost_matrix (torch.Tensor): A 2D tensor containing the costs of
                assigning an element of the first set to an element of the
                second set. The cost is usually given by the pairwise distance.

        Returns:
            torch.Tensor: Binary 2D tensor of assignments of the second set to
                the first set in the pair.
        """

        match_matrix = torch.zeros((self.inp_len, self.out_len))

        preference_inputs, preference_outputs = self.get_ranking(cost_matrix)

        singles = torch.tensor(range(self.inp_len))

        while singles.nelement() != 0:

            for i in singles:  # i is input id

                for j in range(self.out_len):  # j is output ranking

                    output_preferred = preference_inputs[i, j]  # output id

                    if 1 in match_matrix[:, output_preferred]:

                        # get input id it is matched to
                        current_match = torch.where(
                            match_matrix[:, output_preferred] == 1
                        )[0]

                        # get matched input rank from output preferences

                        rank_current_match = torch.where(
                            preference_outputs[output_preferred, :] == current_match
                        )[0]

                        # get potential input rank from output preferences

                        rank_potential = torch.where(
                            preference_outputs[output_preferred, :] == i
                        )[0]

                        if rank_potential < rank_current_match:
                            match_matrix[i, output_preferred] = 1
                            match_matrix[current_match, output_preferred] = 0

                            singles = torch.cat((singles, current_match))
                            singles = singles[singles != i]

                        else:
                            continue

                    else:
                        match_matrix[i, output_preferred] = 1
                        singles = singles[singles != i]

                    break

        return match_matrix
