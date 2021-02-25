"""Testing Shapes"""
import numpy as np
import pandas as pd
from paccmann_sets.datasets.shapes_data import Shapes


def test_data_random():
    """Test data_random. """
    sample_size = 10
    dim = 3
    column_names = [['x'], ['x', 'y'], ['x', 'y', 'z']]
    data = Shapes().data_random(dim, sample_size)

    assert type(data) == pd.DataFrame
    assert len(pd.unique(data['ID'])) == sample_size
    assert data.shape[1] == (dim + 2)
    assert data['label'].dtype == int
    assert data['ID'].dtype == int

    for i in column_names[dim - 1]:
        assert data[i].dtype == np.float64


def test_datapoints_circle():
    """Test datapoints_circle."""
    sample_size = 10
    sample_id = 1

    data = Shapes().datapoints_circle(sample_size, sample_id)

    assert type(data) == pd.DataFrame
    assert len(data) == sample_size
    assert data.shape[1] == 4
    assert data['x'].dtype == np.float64
    assert data['y'].dtype == np.float64
    assert data['label'].dtype == int
    assert data['ID'].dtype == int

    cx, cy, radius = Shapes().datapoints_circle(1, 1, use='square')

    assert type(cx) == int
    assert type(cy) == int
    assert type(radius) == int

    cross_start, centre_x, centre_y = Shapes().datapoints_circle(1, 1, use='cross')

    assert len(cross_start) == 1
    assert cross_start.dtype == int
    assert type(centre_x) == int
    assert type(centre_y) == int


def test_datapoints_square():
    """Test datapoints_square."""
    sample_size = 10
    sample_id = 1

    data = Shapes().datapoints_square(sample_size, sample_id)

    assert type(data) == pd.DataFrame
    assert len(data) == sample_size
    assert data.shape[1] == 4
    assert data['x'].dtype == np.float64
    assert data['y'].dtype == np.float64
    assert data['label'].dtype == int
    assert data['ID'].dtype == int


def test_datapoints_cross():
    """Test datapoints_cross."""
    sample_size = 10
    sample_id = 1

    data = Shapes().datapoints_cross(sample_size, sample_id)

    assert type(data) == pd.DataFrame
    assert len(data) == sample_size
    assert data.shape[1] == 4
    assert data['x'].dtype == np.float64
    assert data['y'].dtype == np.float64
    assert data['label'].dtype == int
    assert data['ID'].dtype == int


def test_data_shapes():
    """Test data_shapes."""
    sample_size = 200
    data = Shapes().data_shapes(sample_size)

    assert type(data) == pd.DataFrame
    assert len(pd.unique(data['ID'])) == sample_size
    assert data.shape[1] == 4
    assert data['x'].dtype == np.float64
    assert data['y'].dtype == np.float64
    assert data['label'].dtype == int
    assert data['ID'].dtype == int
