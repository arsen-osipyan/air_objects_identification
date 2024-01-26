import numpy as np
import pandas as pd
from typing import NoReturn

from .model import Model
from .airenv import AirEnv


class RadarSystem(Model):

    def __init__(self, position: np.array, detection_radius: float, error: float, air_env: AirEnv = None) -> NoReturn:
        """
        Initializes RadarSystem
        :param position: position in R^3 (meters)
        :param detection_radius: detection radius (meters)
        :param error: detection absolute error for each dimension (meters)
        """
        super().__init__()

        self.__position = np.array(position, dtype=float)
        self.__detection_radius = detection_radius
        self.__error = error

        self.__air_env = air_env

        self.__data_dtypes = {
            'time': 'int64',
            'x': 'float64',
            'x_err': 'float64',
            'y': 'float64',
            'y_err': 'float64',
            'z': 'float64',
            'z_err': 'float64',
            'id': 'int64'
        }
        self.__data = pd.DataFrame(columns=list(self.__data_dtypes.keys())).astype(self.__data_dtypes)

    def trigger(self, **kwargs) -> NoReturn:
        """
        Runs detect() method
        :return:
        """
        super().trigger(**kwargs)

        self.detect()

    def get_detections(self) -> pd.DataFrame:
        """
        Gives all collected detections
        :return: dataframe with detections
        """
        return self.__data

    def detect(self) -> NoReturn:
        """
        Detects all visible AirObject-s, adds error and updates detections dataframe
        """
        detections = self.__air_env.air_objects_dataframe()

        detections['time'] = self.time.get()
        detections['x_err'] = self.__error
        detections['y_err'] = self.__error
        detections['z_err'] = self.__error

        p = self.__position
        r = self.__detection_radius
        detections['is_observed'] = detections.apply(
            lambda row: np.sqrt((row['x'] - p[0])**2 + (row['y'] - p[1])**2 + (row['z'] - p[2])**2) <= r,
            axis=1
        )

        detections = detections[detections['is_observed']]

        detections['x'] = detections['x'] + np.random.uniform(-self.__error, self.__error, len(detections))
        detections['y'] = detections['y'] + np.random.uniform(-self.__error, self.__error, len(detections))
        detections['z'] = detections['z'] + np.random.uniform(-self.__error, self.__error, len(detections))

        detections.drop(columns=['is_observed'], inplace=True)

        self.__concat_data(detections)

    def __is_observed(self, position: np.array) -> bool:
        """
        Checks if provided point could be observed from this RadarSystem
        :param position: point position in R^3 (meters)
        :return: is this point observed from RadarSystem
        """
        distance = np.sqrt(np.sum([(position[i] - self.__position[i])**2 for i in range(3)]))
        return distance <= self.__detection_radius

    def __concat_data(self, df: pd.DataFrame) -> NoReturn:
        df = df[list(self.__data_dtypes.keys())].astype(self.__data_dtypes)
        if len(self.__data) == 0:
            self.__data = df
        else:
            df.index += len(self.__data)
            self.__data = pd.concat([self.__data, df])

    def set_air_environment(self, air_env: AirEnv = None) -> NoReturn:
        self.__air_env = air_env

    def __repr__(self) -> str:
        return '<RadarSystem: position={}, detection_radius={}, error={}>'.format(
            self.__position, self.__detection_radius, self.__error
        )

    def __str__(self) -> str:
        return self.__repr__()
