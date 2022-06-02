from math import cos, pi, sin
from typing import List, Tuple

import numpy as np
import pandas as pd
from skimage import draw


class Shapes:
    """Generates a dataset of shapes with varying sizes and rotations.
    """

    def __init__(
        self,
        min_x: float = -0.5,
        max_x: float = 0.5,
        min_y: float = -0.5,
        max_y: float = 0.5,
        min_boundary: int = 1000,
        max_boundary: int = 9000,
        img_size: int = 10000,
        seed: int = None
    ):
        """Constructor.

        Args:
            min_x (float, optional): Lower boundary of the x axis.
                Defaults to -0.5.
            max_x (float, optional): Upper boundary of the x axis.
                Defaults to 0.5.
            min_y (float, optional): Lower boundary of the x axis.
                Defaults to -0.5.
            max_y (float, optional): Upper boundary of the y axis.
                Defaults to 0.5.
            min_boundary (int, optional): Lower boundary of the sampling area
                for x and y axes. Defaults to 1000.
            max_boundary (int, optional): Upper boundary of the sampling area
                for x and y axes. Defaults to 9000.
            img_size (int, optional): Image resolution. Defaults to 10000.

            NOTE: The min and max boundaries of x and y axes represent the
                region where the shapes are required to be situated.
                The sampling area boundaries define regions from which the
                centres of the shapes are sampled to ensure the above min/max
                boundaries are respected.
                The shape is first generated as an image; Image resolution
                controls how accurately the points are translated to the
                required region.
        """

        self.max_x = max_x
        self.min_x = min_x
        self.max_y = max_y
        self.min_y = min_y
        self.min_boundary = min_boundary
        self.max_boundary = max_boundary
        self.img_size = img_size
        if seed:
            np.random.seed(42)

    @staticmethod
    def save_csv(data: pd.DataFrame, filepath: str) -> None:
        """Save dataframe as a csv file.

        Args:
            data (dataframe): Dataframe to be saved as csv.
            filepath (string): Path where the csv file is to be saved.
        """

        assert type(data) == pd.DataFrame

        data.to_csv(filepath)

    def generate_random(self, dim: int, sample_id: int) -> np.ndarray:
        """Generates a set of data points that are sampled uniformly at random.
        Args:
            dim (int): Dimension of the data.
            sample_id(int): ID of the set.
        Returns:
            np.ndarray(float64) : dim-D array of the coordinates
        """
        k = np.random.randint(1, 17)
        element_size = tuple([k, dim])
        column_names = [['x'], ['x', 'y'], ['x', 'y', 'z']]

        label_id = pd.DataFrame({'label': [dim] * k, 'ID': [sample_id] * k})

        return pd.concat(
            [
                pd.DataFrame(
                    np.random.uniform(self.min_x, self.max_x, size=element_size),
                    columns=column_names[dim - 1]
                ), label_id
            ],
            axis=1
        )

    def data_random(self, dimension: int, sample_size: int) -> np.ndarray:
        """Generates a random uniformly distributed dataset.

        Args:
            dimension (int): dimension of elements of the dataset.
            sample_size (int): Length of the dataset.

        Returns:
            dataset (np.ndarray(float64)): array of dimension-D sets with
                varying lengths.
        """
        dataset = pd.concat(
            [self.generate_random(dimension, i) for i in range(1, sample_size + 1)],
            sort=False
        )

        return dataset

    def get_max_radius(self, x: int, y: int, min_radius: int) -> int:
        """Generates a radius for the bounding circle based on maximum and
        minimum possible radii.

        Args:
            x (int): x coordinate of the centre pixel of the circle.
            y (int): y coordinate of the centre pixel of the circle.
            min_radius (int): Fixed minimum radius of the
                bounding circle in pixels.

        Returns:
            radius (int): The radius chosen at random from a calculated range.
        """

        r_max_x = min(np.sqrt((0 - x)**2), np.sqrt((self.img_size - x)**2))
        r_max_y = min(np.sqrt((0 - y)**2), np.sqrt((self.img_size - y)**2))

        r_lim = min(r_max_x, r_max_y)

        radius = np.random.randint(min_radius, int(r_lim))

        return radius

    def scale(self, value, old_min, old_max, new_min, new_max) -> np.float64:
        """Translates a given coordinate from one scale range to another.

        Args:
            value (int or float): The 1-D coordinate to be translated.
            old_min (int or float): The minimum value of the old range.
            old_max (int or float): The maximum value of the old range.
            new_min (int or float): The minimum value of the new range.
            new_max (int or float): The maximum value of the new range.

        Returns:
            (int or float): Value of the coordinate in the new range.
        """
        return ((value - old_min) *
                (new_max - new_min)) / ((old_max - old_min)) + new_min

    def datapoints_circle(
        self, set_length: int, sample_id: int, min_radius: int = 100, use: str = None
    ):
        """Generates a set of datapoints sampled from the circumference of a
        circle.

        Args:
            set_length (int): The number of elements required in the set.
            sample_id(int): ID of the set.
            min_radius (int, optional): The minimum radius of the
                bounding circle in pixels. Defaults to 100.
            use (string, optional): "square" or "cross" signifies the shape it
                is used for as a bounding circle. Defaults to None.

        Returns:
            datapoints (np.ndarray): Sampled 2D points from the circumference
                of the circle of dtype int if used as a bounding circle,
                float otherwise.
            radius (int): Radius of the circle.
            x (int): The x coordinate of the centre pixel of the circle.
                Returned only when use = 'cross'.
            y (int): The y coordinate of the centre pixel of the circle.
                Returned only when use = 'cross'.
        """

        x = np.random.randint(self.min_boundary, self.max_boundary)
        y = np.random.randint(self.min_boundary, self.max_boundary)

        radius = self.get_max_radius(x, y, min_radius)

        rr, cc = draw.circle_perimeter(
            x, y, radius, shape=(self.img_size, self.img_size)
        )

        if use == 'square':
            return x, y, radius

        elif use == 'cross':

            # uncomment the following lines to add rotations

            # points = np.array(list(zip(rr, cc)))

            # sampling_indices = np.random.randint(0, len(points), set_length)
            # datapoints = points[sampling_indices]

            # comment the following lines to add rotations

            rr0 = int(x + radius * cos(pi / 2))
            cc0 = int(y + radius * sin(pi / 2))
            datapoints = np.array(list(zip([rr0], [cc0])))

            return datapoints, x, y

        else:
            scaled_rr = self.scale(rr, 0, self.img_size, self.min_x, self.max_x)
            scaled_cc = self.scale(cc, 0, self.img_size, self.min_y, self.max_y)

            assert all(np.abs(scaled_rr) <= self.max_x)
            assert all(np.abs(scaled_cc) <= self.max_y)

            points = pd.DataFrame(
                {
                    'x': scaled_rr,
                    'y': scaled_cc,
                    'label': [0] * len(rr),
                    'ID': [sample_id] * len(rr)
                }
            )

            sampling_indices = np.random.randint(0, len(points), set_length)
            datapoints = points.iloc[sampling_indices, :].reset_index(drop=True)

            return datapoints

    def datapoints_square(
        self, set_length: int, sample_id: int, min_radius: int = 100
    ) -> Tuple:
        """Generates a set of datapoints sampled from the perimeter of a
        square.

        Args:
            set_length (int): The number of elements required in the set.
            sample_id(int): ID of the set.
            min_radius (int, optional): Fixed minimum radius of the bounding
                circle in pixels. Defaults to 100.

        Returns:
            datapoints (np.ndarray(float)): Sampled 2D points from the
                perimeter of the square.
            side (float): Length of the square generated.
        """

        cx, cy, radius = self.datapoints_circle(1, sample_id, use='square')

        side = np.sqrt(radius * radius * 2)
        half_side = side * 0.5

        # uncomment the following line to add rotations

        # angle = np.random.random_sample() * pi / 2

        r, c = draw.rectangle_perimeter(
            (cx - half_side, cy - half_side),
            extent=(side, side),
            shape=(self.img_size, self.img_size)
        )

        scaled_r = self.scale(r, 0, self.img_size, self.min_x, self.max_x)
        scaled_c = self.scale(c, 0, self.img_size, self.min_y, self.max_y)

        # uncomment the following lines to add rotations

        # mean_r = (max(scaled_r) + min(scaled_r)) / 2
        # mean_c = (max(scaled_c) + min(scaled_c)) / 2

        # rot_r = (scaled_r - mean_r) * cos(angle) - (scaled_c - mean_c
        #                                             ) * sin(angle) + mean_r
        # rot_c = (scaled_r - mean_r) * sin(angle) + (scaled_c - mean_c
        #                                             ) * cos(angle) + mean_c

        # assert all(np.abs(rot_r) <= self.max_x)
        # assert all(np.abs(rot_c) <= self.max_y)

        # points = pd.DataFrame(
        #     {
        #         'x': rot_r,
        #         'y': rot_c,
        #         'label': [1] * len(rot_r),
        #         'ID': [sample_id] * len(rot_r)
        #     }
        # )

        # comment the following lines to add rotations

        points = pd.DataFrame(
            {
                'x': scaled_r,
                'y': scaled_c,
                'label': [1] * len(scaled_r),
                'ID': [sample_id] * len(scaled_r)
            }
        )

        sampling_indices = np.random.randint(0, len(points), set_length)
        datapoints = points.iloc[sampling_indices, :].reset_index(drop=True)

        return datapoints

    def datapoints_cross(
        self, set_length: int, sample_id: int, min_radius: int = 100
    ) -> Tuple:
        """Generates a set of datapoints sampled from a cross.

        Args:
            set_length (int): The number of elements required in the set.
            sample_id(int): ID of the set.
            min_radius (int, optional): Fixed minimum radius of the bounding
                circle in pixels. Defaults to 100.

        Returns:
            datapoints (np.ndarray(float)): Sampled 2D points from the cross.
            radius (int): Radius of the bounding circle in pixels.
        """

        start_point, centre_x, centre_y = self.datapoints_circle(
            1, sample_id, use='cross'
        )

        x1 = 2 * centre_x - start_point[0, 0]
        y1 = 2 * centre_y - start_point[0, 1]

        assert x1.dtype == int
        assert y1.dtype == int

        rr, cc = draw.line(start_point[0, 0], start_point[0, 1], x1, y1)

        scaled_rr = self.scale(rr, 0, self.img_size, self.min_x, self.max_x)
        scaled_cc = self.scale(cc, 0, self.img_size, self.min_y, self.max_y)

        assert all(np.abs(scaled_rr) <= self.max_x)
        assert all(np.abs(scaled_cc) <= self.max_y)

        mean_rr = (max(scaled_rr) + min(scaled_rr)) / 2
        mean_cc = (max(scaled_cc) + min(scaled_cc)) / 2

        rot_rr = (
            (scaled_rr - mean_rr) * cos(pi / 2) - (scaled_cc - mean_cc) * sin(pi / 2)
        ) + mean_rr
        rot_cc = (
            (scaled_rr - mean_rr) * sin(pi / 2) + (scaled_cc - mean_cc) * cos(pi / 2)
        ) + mean_cc

        # uncomment the following lines to add rotations

        # assert all(np.abs(rot_rr) <= self.max_x)
        # assert all(np.abs(rot_cc) <= self.max_y)

        # points = pd.DataFrame(
        #     {
        #         'x': np.append(scaled_rr, rot_rr),
        #         'y': np.append(scaled_cc, rot_cc),
        #         'label': [2] * (2 * len(rr)),
        #         'ID': [sample_id] * (2 * len(rr))
        #     }
        # )

        # comment the following lines to add rotations

        points = pd.DataFrame(
            {
                'x': np.append(scaled_rr, rot_rr),
                'y': np.append(scaled_cc, rot_cc),
                'label': [2] * (2 * len(rr)),
                'ID': [sample_id] * (2 * len(rr))
            }
        )

        sampling_indices = np.random.randint(0, len(points), set_length)
        datapoints = points.iloc[sampling_indices, :].reset_index(drop=True)

        return datapoints

    def generate_shapes(self, shapes_list: List, sample_id: int) -> np.array:
        """Generates a shape at random.

        Args:
            shapes_list (list[objects]): List of function names for generating
                different shapes.
            sample_id(int): ID of the set.

        Returns:
            np.ndarray: 2D array of sampled datapoints from the randomly
                chosen shape.
        """
        set_length = np.random.randint(10, 34)

        return np.random.choice(shapes_list)(set_length, sample_id)

    def data_shapes(self, sample_size: int) -> List:
        """Generates a dataset of randomly chosen shapes of varying sizes and
        orientations.

        Args:
            sample_size (int): Length of the dataset.

        Returns:
            dataset (list[float]): Each item is a shape and its coordinates.

            NOTE: dataset is an array of 2D variable length arrays.
        """

        shapes_list = [
            self.datapoints_circle, self.datapoints_square, self.datapoints_cross
        ]

        dataset = pd.concat(
            [self.generate_shapes(shapes_list, i) for i in range(1, sample_size + 1)],
            sort=False
        )

        return dataset
