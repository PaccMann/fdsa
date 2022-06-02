import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical


class SetMatchLoss(nn.Module):

    def __init__(
        self,
        loss: str,
        ce_type: str = 'row',
        temperature: float = None,
        iterations: int = None
    ) -> None:
        """Constructor.

        Args:
            loss (str): Loss function to use. See dictionary below for keys.
            temperature (float, optional): Temperature to apply to logits.
                Defaults to None.
            iterations (int, optional): Number of iterations for sinkhorn
                normalisation. Defaults to None.
        """
        super(SetMatchLoss, self).__init__()
        self.loss = loss
        self.temp = temperature
        self.n = iterations
        self.ce_type = ce_type

        self.loss_func = dict(
            {
                'ce_rowcol': self.crossentropy_rowcol,
                'ce_row': self.crossentropy_row,
                'ce_l1': self.crossentropy_l1,
                'ce_l2': self.crossentropy_l2,
                'ce_l1l2': self.crossentropy_l1l2,
                'ce_l1l2_penalty': self.crossentropy_l1l2_penalty,
                'kl_div': self.kl_div_loss,
                'mask_loss': self.mask_loss,
                'unique_max': self.unique_max_mask_loss,
                'sinkhorn': self.sinkhorn_loss,
                'kl_dist': self.distance_loss
            }
        )

    def similarity(
        self, predictions: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        """Computes the L1-loss between predicted probabilities and true binary
           matrix.

        Args:
            predictions (torch.Tensor): Probability matrix with shape
                [batch_size,set_length,set_length].
            target_mask (torch.Tensor): True binary permutation matrix with
                shape [batch_size,set_length,set_length].

        Returns:
            torch.Tensor: The scalar loss value.
        """
        loss = nn.L1Loss()
        return loss(predictions, target_mask.float())

    def crossentropy_row(
        self, predictions: torch.Tensor, target12: torch.Tensor, target21: torch.Tensor
    ) -> torch.Tensor:
        """Computes row-wise cross entropy loss.

        Args:
            predictions (torch.Tensor): Logits from the last FC layer with shape
                [batch_size,set_length,set_length].
            target12 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set1 vs set2.
            target21 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set2 vs set1.

        Returns:
            torch.Tensor: The scalar loss value.
        """

        row_constraint = F.log_softmax(predictions / self.temp, dim=2)

        row_loss = F.nll_loss(row_constraint.permute(0, 2, 1), target21)

        return row_loss

    def crossentropy_rowcol(
        self, predictions: torch.Tensor, target12: torch.Tensor, target21: torch.Tensor
    ) -> torch.Tensor:
        """Computes row and column-wise cross entropy loss.

        Args:
            predictions (torch.Tensor): Logits from the last FC layer with shape
                [batch_size,set_length,set_length].
            target12 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set1 vs set2.
            target21 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set2 vs set1.
        TODO: make this modular, write a function for column cross entropy and
            sum row and column CE functions to get rowcol CE.

        Returns:
            torch.Tensor: The scalar loss value.
        """

        row_constraint = F.log_softmax(predictions / self.temp, dim=2)
        col_constraint = F.log_softmax(predictions / self.temp, dim=1)

        row_loss = F.nll_loss(row_constraint.permute(0, 2, 1), target21)
        col_loss = F.nll_loss(col_constraint, target12)

        return row_loss + col_loss

    def crossentropy_l1(
        self, predictions: torch.Tensor, target12: torch.Tensor, target21: torch.Tensor
    ) -> torch.Tensor:
        """Computes row and column-wise cross entropy loss and an additional L1
           similarity loss between predicted row-wise probabilities and one hot
           embedding of target21.

        Args:
            predictions (torch.Tensor): Logits from the last FC layer with shape
                [batch_size,set_length,set_length].
            target12 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set1 vs set2.
            target21 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set2 vs set1.
        TODO: Make similarity flexible to allow easy switching between row and
            column L1 loss using arguments.
        Returns:
            torch.Tensor: The scalar loss value.
        """

        if self.ce_type == 'rowcol':
            ce_loss = self.crossentropy_rowcol(predictions, target12, target21)
        elif self.ce_type == 'row':
            ce_loss = self.crossentropy_row(predictions, target12, target21)

        true_mask21 = F.one_hot(target21).float()

        row_constraint = F.log_softmax(predictions / self.temp, dim=2)

        similarity = self.similarity(row_constraint, true_mask21)

        return ce_loss + similarity

    def crossentropy_l2(
        self, predictions: torch.Tensor, target12: torch.Tensor, target21: torch.Tensor
    ) -> torch.Tensor:
        """Samples from row and column probability distributions and
            computes the L2-norm between their sampled probabilties. Loss is a
            sum of crossentropy_rowcol and the L2-norm.

        Args:
            predictions (torch.Tensor): Logits from the last FC layer with shape
                [batch_size,set_length,set_length].
            target12 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set1 vs set2.
            target21 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set2 vs set1.

        Returns:
            torch.Tensor: The scalar loss value.
        """

        if self.ce_type == 'rowcol':
            ce_loss = self.crossentropy_rowcol(predictions, target12, target21)
        elif self.ce_type == 'row':
            ce_loss = self.crossentropy_row(predictions, target12, target21)

        row_softmax = F.softmax(predictions / self.temp, dim=2)
        col_softmax = F.softmax(predictions / self.temp, dim=1).permute(0, 2, 1)

        row_mask = OneHotCategorical(row_softmax).sample()

        col_mask = OneHotCategorical(col_softmax).sample()

        row_filtered = row_softmax * row_mask
        col_filtered = col_softmax * col_mask

        loss = nn.MSELoss()

        l2_norm = torch.sqrt(loss(row_filtered, col_filtered.permute(0, 2, 1)))

        return ce_loss + l2_norm

    def crossentropy_l1l2(
        self, predictions: torch.Tensor, target12: torch.Tensor, target21: torch.Tensor
    ) -> torch.Tensor:
        """Computes L1 and L2 penalties in addition to cross entropy loss.

        Args:
            predictions (torch.Tensor): Logits from the last FC layer with shape
                [batch_size,set_length,set_length].
            target12 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set1 vs set2.
            target21 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set2 vs set1.

        Returns:
            torch.Tensor: The scalar loss value.
        """

        ce_loss = self.crossentropy_l2(predictions, target12, target21)

        true_mask21 = F.one_hot(target21).float()

        row_constraint = F.log_softmax(predictions / self.temp, dim=2)

        similarity = self.similarity(row_constraint, true_mask21)

        return ce_loss + similarity

    def kl_div_loss(
        self, predictions: torch.Tensor, target12: torch.Tensor, target21: torch.Tensor
    ) -> torch.Tensor:
        """Computes the KL-Divergence between log softmax of logits and binary
            matrix of true targets both row and column-wise.

        Args:
            predictions (torch.Tensor): Logits from the last FC layer with shape
                [batch_size,set_length,set_length].
            target12 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set1 vs set2.
            target21 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set2 vs set1.

        Returns:
            torch.Tensor: The scalar loss value.
        """

        # output vs input
        row_constraint = F.log_softmax(predictions, dim=2)
        # input vs output
        col_constraint = F.log_softmax(predictions, dim=1).permute(0, 2, 1)

        # input vs output permutation matrix
        one_hot_target12 = F.one_hot(target12).type(torch.float32)
        # output vs input permutation matrix
        one_hot_target21 = F.one_hot(target21).type(torch.float32)

        loss = nn.KLDivLoss(reduction='batchmean')

        return loss(row_constraint,
                    one_hot_target21) + loss(col_constraint, one_hot_target12)

    def mask_loss(
        self, predictions: torch.Tensor, target12: torch.Tensor, target21: torch.Tensor
    ) -> torch.Tensor:
        """Computes the row wise Cross Entropy Loss and the L1-norm between the
           the column-wise sum of the predicted permutation matrix and a
           vector of ones.

        Args:
            predictions (torch.Tensor): Logits from the last FC layer with shape
                [batch_size,set_length,set_length].
            target12 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set1 vs set2.
            target21 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set2 vs set1.

        Returns:
            torch.Tensor: The scalar loss value.
        """
        true_mask = F.one_hot(target21)
        softmax_predictions = F.softmax(predictions, 2)

        softmax_argmax = torch.argmax(softmax_predictions, 2)
        pred_mask = F.one_hot(softmax_argmax, true_mask.size(2)).float()

        loss_fn = nn.CrossEntropyLoss()

        ce = loss_fn(predictions.permute(0, 2, 1), target21)
        col_constraint = torch.ones(
            true_mask.size(0), true_mask.size(2), device=target21.device
        )

        constrained_loss = nn.L1Loss()(torch.sum(pred_mask, dim=1), col_constraint)

        return ce + constrained_loss

    def sinkhorn_normalisation(self, predictions: torch.Tensor) -> torch.Tensor:
        """Returns a doubly stochastic matrix by Sinkhorn-Knopp normalisation.

        Args:
            predictions (torch.Tensor): Logits from the last FC layer with shape
                [batch_size,set_length,set_length].

        Returns:
            torch.Tensor: Doubly stochastic matrix (DSM) such that row and
                column sum to 1. Shape: [batch_size,set_length,set_length].
        """

        def row_norm(predictions):
            """Performs row-wise normalisation."""

            return F.normalize(predictions, p=1, dim=2)

        def col_norm(predictions):
            """Performs column-wise normalisation."""

            return F.normalize(predictions, p=1, dim=1)

        positive_matrix = F.relu(predictions)
        sinkhorn = positive_matrix + 1e-9

        if self.n is not None:
            for i in range(self.n):
                sinkhorn = col_norm(row_norm(sinkhorn))

        return sinkhorn

    def sinkhorn_loss(
        self, predictions: torch.Tensor, target12: torch.Tensor, target21: torch.Tensor
    ) -> torch.Tensor:
        """Computes KL-divergence between the DSM and true permutation matrix
            both row and column-wise.

        Args:
            predictions (torch.Tensor): Logits from the last FC layer with shape
                [batch_size,set_length,set_length].
            target12 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set1 vs set2.
            target21 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set2 vs set1.

        Returns:
            torch.Tensor: The scalar loss value.
        """
        doubly_stochastic_matrix = self.sinkhorn_normalisation(predictions)

        true_mask21 = F.one_hot(target21).float()
        true_mask12 = F.one_hot(target12).float()
        loss_fn = nn.KLDivLoss(reduction='batchmean')

        log_dsm = torch.log(doubly_stochastic_matrix)

        loss = loss_fn(log_dsm,
                       true_mask21) + loss_fn(log_dsm.permute(0, 2, 1), true_mask12)

        return loss

    def l1l2_penalty(self, predictions: torch.Tensor) -> torch.Tensor:
        """Computes L1-L2 matrix penalty as shown in AutoShuffle Net.
            https://arxiv.org/pdf/1901.08624.pdf

        Args:
            predictions (torch.Tensor): Logits from the last FC layer with shape
                [batch_size,set_length,set_length].

        Returns:
            torch.Tensor: The scalar penalty value.
        """

        row_softmax = F.softmax(predictions / self.temp, dim=2)
        col_softmax = F.softmax(predictions / self.temp, dim=1)

        row_l1 = torch.norm(row_softmax, p=1, dim=2)
        row_l2 = torch.norm(row_softmax, p=2, dim=2)

        col_l1 = torch.norm(col_softmax, p=1, dim=1)
        col_l2 = torch.norm(col_softmax, p=2, dim=1)

        p = torch.sum(row_l1 - row_l2) + torch.sum(col_l1 - col_l2)

        return p

    def crossentropy_l1l2_penalty(
        self, predictions: torch.Tensor, target12: torch.Tensor, target21: torch.Tensor
    ) -> torch.Tensor:
        """Computes cross entropy loss in addition to the L1-L2 matrix penalty.

        Args:
            predictions (torch.Tensor): Logits from the last FC layer with shape
                [batch_size,set_length,set_length].
            target12 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set1 vs set2.
            target21 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set2 vs set1.
            

        Returns:
            torch.Tensor: The scalar loss value.
        """

        if self.ce_type == 'rowcol':
            ce_loss = self.crossentropy_rowcol(predictions, target12, target21)
        elif self.ce_type == 'row':
            ce_loss = self.crossentropy_row(predictions, target12, target21)

        penalty = self.l1l2_penalty(predictions)

        return ce_loss + penalty

    def unique_max_mask_loss(
        self, predictions: torch.Tensor, target12: torch.Tensor, target21: torch.Tensor
    ) -> torch.Tensor:
        """Combines the maximum probabilties row and column-wise into a single
            sparse probabilty matrix, and then uses the column-wise max to
            predict target12. This prediction is then evaluated using Cross
            Entropy Loss.

        Args:
            predictions (torch.Tensor): Logits from the last FC layer with shape
                [batch_size,set_length,set_length].
            target12 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set1 vs set2.
            target21 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set2 vs set1.

        Returns:
            torch.Tensor: The scalar loss value.
        """

        classes = target12.size()[1]
        row_softmax = F.softmax(predictions, 2)
        col_softmax = F.softmax(predictions, 1)

        row_argmax = torch.argmax(row_softmax, 2)
        col_argmax = torch.argmax(col_softmax, 1)

        row_mask = F.one_hot(row_argmax, classes)
        # Transpose to get the correct out vs in orientation
        col_mask = F.one_hot(col_argmax, classes).permute(0, 2, 1)

        combined_sparse_probabilities = row_softmax * row_mask + col_softmax * col_mask

        # orientation of combined tensor is out vs in
        predicted_class12 = torch.argmax(combined_sparse_probabilities, 1)
        # since prediction is along 1st dim, we need to transpose one_hot
        # to get the correct mask in out vs in orientation
        predicted_mask12 = F.one_hot(predicted_class12, classes).permute(0, 2, 1)

        final_predictions12 = combined_sparse_probabilities * predicted_mask12

        # final_predictions are of orientation out vs in where probs are along 1st dim
        # so no need of permuting when calculating nll loss with target12
        # because for multi-dim the shape reqd is batch x class_prob x seq_len

        ce = nn.CrossEntropyLoss()

        loss = ce(final_predictions12, target12)

        return loss

    def forward(
        self, predictions: torch.Tensor, target12: torch.Tensor, target21: torch.Tensor
    ) -> torch.Tensor:
        """Returns the desired loss value.

        Args:
            predictions (torch.Tensor): Logits from the last FC layer with shape
                [batch_size,set_length,set_length].
            target12 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set1 vs set2.
            target21 (torch.Tensor): True matching indices such that a one hot
                embedding produces the permutation matrix for set2 vs set1.

        Returns:
            torch.Tensor: The scalar loss value.
        """

        return self.loss_func[self.loss](predictions, target12, target21)
