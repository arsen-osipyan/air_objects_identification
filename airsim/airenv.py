from .model import Model
from .airobject import AirObject
import pandas as pd
from typing import NoReturn


class AirEnv(Model):

    def __init__(self) -> NoReturn:
        """
        Initializes AirEnv
        """
        super().__init__()
        self.__air_objects = dict()
        self.__air_object_id_incr = 0

        self.__public_ids = False

    def trigger(self) -> NoReturn:
        """
        Runs trigger() method of all attached AirObject-s
        """
        super().trigger()

        for k in self.__air_objects.keys():
            self.__air_objects[k].trigger()

    def is_attached(self, air_object: AirObject) -> bool:
        """
        Checks if provided AirObject is attached to AirEnv
        :param air_object: AirObject to check
        :return: attachment status
        """
        return air_object in self.__air_objects.values()

    def attach_air_object(self, air_object: AirObject) -> int:
        """
        Attaches AirObject if it wasn't attached
        :param air_object: AirObject to attach
        :return: AirObject inner id
        """
        if self.is_attached(air_object):
            raise RuntimeError('AirObject already attached to AirEnv.')
        self.__air_objects[self.__air_object_id_incr] = air_object
        self.__air_object_id_incr += 1
        return self.__air_object_id_incr - 1

    def detach_air_object(self, air_object: AirObject) -> int:
        """
        Detaches AirObject if it was attached
        :param air_object: AirObject to detach
        :return: AirObject inner id
        """
        if self.is_attached(air_object):
            raise RuntimeError('AirObject is not attached to AirEnv.')
        for k, v in self.__air_objects.items():
            if v == air_object:
                self.__air_objects.pop(k, None)
                return k

    def air_objects_dataframe(self) -> pd.DataFrame:
        """
        Gives all AirObject-s positions in current time
        :return: dataframe with AirObject-s positions and ids (optional)
        """
        data = pd.DataFrame(columns=['id', 'x', 'y', 'z'])
        for k in self.__air_objects.keys():
            data.loc[len(data)] = {
                'id': k if self.__public_ids else -1,
                'x': self.__air_objects[k].position()[0],
                'y': self.__air_objects[k].position()[1],
                'z': self.__air_objects[k].position()[2]
            }
        return data

    def set_public_ids(self, public_ids: bool) -> NoReturn:
        """
        Sets public_ids parameter to show/hide actual AirObject-s ids
        :param public_ids: logical value to set
        """
        self.__public_ids = public_ids

    def __repr__(self) -> str:
        return '<AirEnv: len(air_objects)={}>'.format(len(self.__air_objects))

    def __str__(self) -> str:
        return self.__repr__()
