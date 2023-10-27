import numpy as np
import pandas as pd
from .model import Model


class RadarSystem(Model):

    def __init__(self, position, detection_radius, error):
        """
        Initializing RadarSystem with position (in m), detection_radius (in m) and error (in m)
        :param position: np.array([_, _, _])
        :param detection_radius: float
        :param error: float
        """
        super().__init__()

        self.__position = position
        self.__detection_radius = detection_radius
        self.__error = error

        self.__air_env = None

        self.__data = pd.DataFrame(
            columns=['detection_time', 'detection_id', 'detection_error',
                     'detection_x', 'detection_y', 'detection_z', 'is_observed']
        )

    def trigger(self):
        """
        Running detect
        """
        super().trigger()

        self.detect()

    def get_detections(self):
        return self.__data

    def detect(self):
        detections = self.__air_env.air_objects()

        detections['detection_time'] = self.time.get()
        detections['detection_id'] = range(len(detections))
        detections['detection_error'] = self.__error

        detections['is_observed'] = self.__is_observed(np.array([detections['x'], detections['y'], detections['z']]))

        detections = detections[detections['is_observed']]

        detections['x'] = detections['x'] + self.__error * np.random.randn(len(detections))
        detections['y'] = detections['y'] + self.__error * np.random.randn(len(detections))
        detections['z'] = detections['z'] + self.__error * np.random.randn(len(detections))

        detections = detections.rename(columns={
            'x': 'detection_x',
            'y': 'detection_y',
            'z': 'detection_z'
        })

        if len(self.__data) == 0:
            self.__data = detections[['detection_time', 'detection_id', 'detection_error',
                                      'detection_x', 'detection_y', 'detection_z', 'is_observed']]
        else:
            self.__data = pd.concat([self.__data, detections])

    def __is_observed(self, position):
        distance = np.sqrt(np.sum([(position[i] - self.__position[i])**2 for i in range(3)]))
        return distance <= self.__detection_radius

    def attach_air_environment(self, air_env):
        if self.__air_env is not None:
            raise RuntimeError('Air environment already attached to radar system.')
        self.__air_env = air_env

    def detach_air_environment(self):
        if self.__air_env is None:
            raise RuntimeError('Radar system has no air environment attached.')
        self.__air_env = None

    def __repr__(self):
        return '<RadarSystem: position={}, detection_radius={}, error={}>'.format(
            self.__position, self.__detection_radius, self.__error
        )

    def __str__(self):
        return self.__repr__()
