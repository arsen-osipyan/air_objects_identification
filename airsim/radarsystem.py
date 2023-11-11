import numpy as np
import pandas as pd
from .model import Model
from typing import NoReturn
from .airenv import AirEnv


class RadarSystem(Model):

    def __init__(self, position: np.array, detection_radius: float, error: float) -> NoReturn:
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

        self.__air_env = None

        self.__data = pd.DataFrame(
            columns=['detection_time', 'air_object_id', 'detection_id', 'detection_error',
                     'air_object_x', 'air_object_y', 'air_object_z']
        )

    def trigger(self) -> NoReturn:
        """
        Runs detect() method
        :return:
        """
        super().trigger()

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

        detections.rename(columns={
            'id': 'air_object_id',
            'x': 'air_object_x',
            'y': 'air_object_y',
            'z': 'air_object_z'
        }, inplace=True)

        detections['detection_time'] = self.time.get()
        detections['detection_id'] = range(len(detections))
        detections['detection_error'] = self.__error

        detections['is_observed'] = self.__is_observed(
            np.array([detections['air_object_x'], detections['air_object_y'], detections['air_object_z']]))

        detections = detections[detections['is_observed']]

        detections['air_object_x'] = detections['air_object_x'] + self.__error * np.random.randn(len(detections))
        detections['air_object_y'] = detections['air_object_y'] + self.__error * np.random.randn(len(detections))
        detections['air_object_z'] = detections['air_object_z'] + self.__error * np.random.randn(len(detections))

        if len(self.__data) == 0:
            self.__data = detections[['detection_time', 'air_object_id', 'detection_id', 'detection_error',
                                      'air_object_x', 'air_object_y', 'air_object_z']]
        else:
            self.__data = pd.concat([self.__data, detections])

    def __is_observed(self, position: np.array) -> bool:
        """
        Checks if provided point could be observed from this RadarSystem
        :param position: point position in R^3 (meters)
        :return: is this point observed from RadarSystem
        """
        distance = np.sqrt(np.sum([(position[i] - self.__position[i])**2 for i in range(3)]))
        return distance <= self.__detection_radius

    def attach_air_environment(self, air_env: AirEnv) -> NoReturn:
        """
        Attaches AirEnv if wasn't attached yet
        :param air_env: AirEnv to attach
        """
        if self.__air_env is not None:
            raise RuntimeError('AirEnv already attached to RadarSystem.')
        self.__air_env = air_env

    def detach_air_environment(self) -> NoReturn:
        """
        Detaches AirEnv if it was attached
        """
        if self.__air_env is None:
            raise RuntimeError('AirEnv is not attached to RadarSystem.')
        self.__air_env = None

    def __repr__(self) -> str:
        return '<RadarSystem: position={}, detection_radius={}, error={}>'.format(
            self.__position, self.__detection_radius, self.__error
        )

    def __str__(self) -> str:
        return self.__repr__()
