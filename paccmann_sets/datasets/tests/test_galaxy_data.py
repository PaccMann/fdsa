"""Testing Galaxy Data"""
import os
import shutil
import tempfile

import pandas as pd
import requests
from paccmann_sets.datasets.galaxy_data import Galaxy


def test_data_galaxy():
    """Test data_galaxy."""

    directory = tempfile.mkdtemp()

    url = (
        'http://risa.stanford.edu/redmapper/v6.3/' +
        'redmapper_dr8_public_v6.3_members.fits.gz'
    )

    filename = url.split("/")[-1]
    filepath = os.path.join(directory, filename)
    with open(filepath, "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    data = Galaxy(filepath).data_galaxy()

    assert type(data) == pd.DataFrame
    assert len(pd.unique(data['ID'])) == 26111
    assert data.shape[1] == 22

    shutil.rmtree(directory)
