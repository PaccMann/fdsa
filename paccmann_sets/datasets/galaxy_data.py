from typing import Tuple

import pandas as pd
from astropy.table import Table


class Galaxy:
    """Generates a dataset of galaxy clusters.
    """

    def __init__(self, fits_filepath):
        """Constructor.

        Args:
            fits_filepath (string): Path to the fits file.
        """
        self.path = fits_filepath

    @staticmethod
    def save_csv(data, filepath):
        """Save dataframe as a csv file.

        Args:
            data (dataframe): Dataframe to be saved as csv.
            filepath (string): Path where the csv file is to be saved.
        """

        assert type(data) == pd.DataFrame

        data.to_csv(filepath)

    def data_galaxy(self) -> Tuple:
        """Extracts data from fits file.

        Returns:
            list_of_dataframes (List): Each element in the list is a N(c) x 19
                matrix corresponding to a cluster c that describes the galaxies
                in that cluster. N(c) is the number of galaxies in cluster c.
                The 19 columns represent 2 IDs and 17 features.
            list_of_targets (List): Each element in the list is a N(c) x 3
                matrix corresponding to a cluster c. The 3 columns represent
                cluster ID, object ID, and the true spectrometric value.
        """

        data = Table.read(self.path, format='fits')
        dataframe = data.to_pandas()

        return dataframe
