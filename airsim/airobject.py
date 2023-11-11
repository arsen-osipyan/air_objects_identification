import numpy as np
from .model import Model
from typing import NoReturn


class AirObject(Model):

    def __init__(self, position: np.array, velocity: np.array, acceleration: np.array) -> NoReturn:
        """
        Initializes AirObject
        :param position: position in R^3 (meters)
        :param velocity: velocity in R^3 (meters / seconds)
        :param acceleration: acceleration in R^3 (meters / seconds^2)
        """
        super().__init__()
        self.__position = np.array(position, dtype=float)
        self.__velocity = np.array(velocity, dtype=float)
        self.__acceleration = np.array(acceleration, dtype=float)

    def trigger(self) -> NoReturn:
        """
        Runs move() method with time interval from last trigger() call
        """
        super().trigger()

        self.move(self.time.get() - self.prev_trigger_call_time())

    def move(self, dt: float) -> NoReturn:
        """
        Updates AirObject position and velocity
        :param dt: time interval in seconds
        """
        self.__position += self.__velocity * dt + self.__acceleration * dt ** 2 / 2
        self.__velocity += self.__acceleration * dt

    def position(self) -> np.array:
        return self.__position

    def velocity(self) -> np.array:
        return self.__velocity

    def acceleration(self) -> np.array:
        return self.__acceleration

    def set_position(self, position: np.array) -> NoReturn:
        self.__position = position

    def set_velocity(self, velocity: np.array) -> NoReturn:
        self.__velocity = velocity

    def set_acceleration(self, acceleration: np.array) -> NoReturn:
        self.__acceleration = acceleration

    def __repr__(self) -> str:
        return '<AirObject: position={}, velocity={}, acceleration={}>'.format(
            self.__position, self.__velocity, self.__acceleration
        )

    def __str__(self) -> str:
        return self.__repr__()
