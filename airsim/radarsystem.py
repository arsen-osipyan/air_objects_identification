import numpy as np
import pandas as pd
from .model import Model
from typing import NoReturn
from .airenv import AirEnv


class RadarSystem(Model):

    def __init__(self, position: np.array, detection_radius: float, error: float) -> NoReturn:
        super().__init__()

        self.__position = position
        self.__detection_radius = detection_radius
        self.__error = error

        self.__air_env = None

        self.__data = pd.DataFrame(
            columns=['detection_time', 'detection_ao_id', 'detection_id', 'detection_error',
                     'detection_x', 'detection_y', 'detection_z', 'is_observed']
        )

    def trigger(self) -> NoReturn:
        super().trigger()

        self.detect()

    def get_detections(self) -> pd.DataFrame:
        return self.__data

    def detect(self) -> NoReturn:
        detections = self.__air_env.air_objects_dataframe()

        detections.rename(columns={
            'id': 'detection_ao_id',
            'x': 'detection_x',
            'y': 'detection_y',
            'z': 'detection_z'
        }, inplace=True)

        detections['detection_time'] = self.time.get()
        detections['detection_id'] = range(len(detections))
        detections['detection_error'] = self.__error

        detections['is_observed'] = self.__is_observed(
            np.array([detections['detection_x'], detections['detection_y'], detections['detection_z']]))

        detections = detections[detections['is_observed']]

        detections['detection_x'] = detections['detection_x'] + self.__error * np.random.randn(len(detections))
        detections['detection_y'] = detections['detection_y'] + self.__error * np.random.randn(len(detections))
        detections['detection_z'] = detections['detection_z'] + self.__error * np.random.randn(len(detections))

        if len(self.__data) == 0:
            self.__data = detections[['detection_time', 'detection_ao_id', 'detection_id', 'detection_error',
                                      'detection_x', 'detection_y', 'detection_z', 'is_observed']]
        else:
            self.__data = pd.concat([self.__data, detections])

    def __is_observed(self, position: np.array) -> bool:
        distance = np.sqrt(np.sum([(position[i] - self.__position[i])**2 for i in range(3)]))
        return distance <= self.__detection_radius

    def attach_air_environment(self, air_env: AirEnv) -> NoReturn:
        if self.__air_env is not None:
            raise RuntimeError('AirEnv already attached to RadarSystem.')
        self.__air_env = air_env

    def detach_air_environment(self) -> NoReturn:
        if self.__air_env is None:
            raise RuntimeError('AirEnv is not attached to RadarSystem.')
        self.__air_env = None

    def __repr__(self) -> str:
        return '<RadarSystem: position={}, detection_radius={}, error={}>'.format(
            self.__position, self.__detection_radius, self.__error
        )

    def __str__(self) -> str:
        return self.__repr__()
