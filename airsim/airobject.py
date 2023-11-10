from .model import Model
from typing import List, NoReturn


class AirObject(Model):

    def __init__(self, position, velocity, acceleration):
        """
        Initializing AirObject with position (in m), velocity (in m/s) and acceleration (in m/s^2)
        :param position: np.array([_, _, _])
        :param velocity: np.array([_, _, _])
        :param acceleration: np.array([_, _, _])
        """
        super().__init__()
        self.__position = position
        self.__velocity = velocity
        self.__acceleration = acceleration

    def trigger(self):
        """
        Moving air object for the time passed from last trigger() call
        """
        super().trigger()

        self.move(self.time.get() - self.prev_trigger_call_time())

    def move(self, dt):
        """
        Moving air object for dt seconds
        :param dt: float
        """
        self.__position += self.__velocity * dt + self.__acceleration * dt ** 2 / 2
        self.__velocity += self.__acceleration * dt

    def position(self):
        return self.__position

    def velocity(self):
        return self.__velocity

    def acceleration(self):
        return self.__acceleration

    def set_position(self, position):
        self.__position = position

    def set_velocity(self, velocity):
        self.__velocity = velocity

    def set_acceleration(self, acceleration):
        self.__acceleration = acceleration

    def __repr__(self):
        return '<AirObject: position={}, velocity={}, acceleration={}>'.format(
            self.__position, self.__velocity, self.__acceleration
        )

    def __str__(self):
        return self.__repr__()
