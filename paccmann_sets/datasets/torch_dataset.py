from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ToySetsDataset(Dataset):
    """Dataset class for loading the Shapes and Galaxy data."""

    def __init__(
        self,
        csv_file: str,
        features: List[str],
        target: List[str],
        identifiers: List[str] = ['ID', 'OBJID']
    ) -> None:
        """Constructor.

        Args:
            csv_file (string): Path to the csv file containing data.
            features (list(string)): Column names of features.
            target (list(string)): Column name(s) of target.
            identifiers (list(string), optional): Column names associated
                with ID. Defaults to ['ID','OBJID'] (according to galaxy data).

        """
        self.data = pd.read_csv(csv_file, dtype=np.float32)

        self.identifiers = self.data[identifiers]
        self.x = self.data[features]
        self.y = self.data[target]
        self.unique_id = pd.unique(self.identifiers['ID'])

    def __len__(self) -> int:
        """Length of the data.

        Returns:
            int: Length of the data.
        """
        return len(self.unique_id)

    def __getitem__(self, idx: torch.Tensor) -> Tuple:
        """Returns all elements belonging to one set.

        Args:
            idx (tensor): Index of set to be sampled.

        Returns:
            Tuple: Tuple of tensor of features and corresponding targets subset
                by set ID and not element ID, and length of feature tensor.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id_ = self.unique_id[idx]

        x = torch.from_numpy(self.x[self.identifiers['ID'].isin([id_])].values)
        # y = torch.from_numpy(self.y[self.identifiers['ID'].isin([id_])].values)

        return x


class SetsDataset(Dataset):
    """Dataset class to load set data from a file."""

    def __init__(
        self,
        dataset_path: str,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        self.dataset = torch.load(dataset_path).to(device)

    def __len__(self):
        """Get length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """Gets item from the dataset at the given index."""

        if torch.is_tensor(index):
            index = index.tolist()

        return self.dataset[index, :, :]


class Collate:
    """Class to pad data based on maximum set length in a batch."""

    def __init__(
        self,
        max_length: int,
        input_dim: int,
        padding_value: int,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        """Constructor.

        Args:
            max_length (int): Maximum length of the set as required.
            input_dim (int): Size of the input elements in the set.
            padding_value (int): Numerical value to pad the set.
            device (torch.device, optional): Device on which the data is stored.
                Defaults to CPU.
        """

        self.max_length = max_length
        self.dim = input_dim
        self.pad_val = padding_value
        self.device = device

    def __call__(self, batch) -> Tuple:
        """Padding function that returns the padded sets, and true lengths of each set
            in the batch.

        Args:
            batch (object): Batch object from the DataLoader.

        Returns:
            Tuple: Tuple of padded tensors of sets and tensors of set lengths.
        """

        lengths = list(map(len, batch))
        batch_size = len(batch)

        padded_seqs = torch.full(
            (batch_size, self.max_length, self.dim), self.pad_val, device=self.device
        )

        for i, l in enumerate(lengths):
            padded_seqs[i, 0:l, :] = batch[i][0:l, :]

        return padded_seqs, torch.tensor(lengths)
