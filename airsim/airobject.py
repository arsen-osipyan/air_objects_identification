import numpy as np
from typing import NoReturn, Callable

from .model import Model


class AirObject(Model):

    def __init__(self, track: Callable[[int], np.array]) -> NoReturn:
        super().__init__()

        # Checking if track function returns 3-dimensional numpy array
        if track(self.time.get()).shape != (3,):
            raise RuntimeError(f'Track function should return numpy array with (3,) shape.')

        self.__track = track

    def trigger(self, **kwargs) -> NoReturn:
        super().trigger(**kwargs)

    def position(self) -> np.array:
        return list(map(float, self.__track(self.time.get())))

    def __repr__(self) -> str:
        return '<AirObject: position={}>'.format(self.position())

    def __str__(self) -> str:
        return self.__repr__()
