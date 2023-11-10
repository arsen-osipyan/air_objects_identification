from .model import Model
from .airobject import AirObject
import pandas as pd
from typing import NoReturn


class AirEnv(Model):

    def __init__(self) -> NoReturn:
        super().__init__()
        self.__air_objects = dict()
        self.__air_object_id_incr = 0

        self.__public_ids = False

    def trigger(self) -> NoReturn:
        super().trigger()

        for k in self.__air_objects.keys():
            self.__air_objects[k].trigger()

    def is_attached(self, air_object: AirObject) -> bool:
        return air_object in self.__air_objects.values()

    def attach_air_object(self, air_object: AirObject) -> int:
        if self.is_attached(air_object):
            raise RuntimeError('AirObject already attached to AirEnv.')
        self.__air_objects[self.__air_object_id_incr] = air_object
        self.__air_object_id_incr += 1
        return self.__air_object_id_incr - 1

    def detach_air_object(self, air_object: AirObject) -> int:
        if self.is_attached(air_object):
            raise RuntimeError('AirObject is not attached to AirEnv.')
        for k, v in self.__air_objects.items():
            if v == air_object:
                self.__air_objects.pop(k, None)
                return k

    def air_objects_dataframe(self) -> pd.DataFrame:
        data = pd.DataFrame(columns=['id', 'x', 'y', 'z'])
        for k in self.__air_objects.keys():
            data.loc[len(data)] = {
                'id': k if self.__public_ids else None,
                'x': self.__air_objects[k].position()[0],
                'y': self.__air_objects[k].position()[1],
                'z': self.__air_objects[k].position()[2]
            }
        return data

    def set_public_ids(self, public_ids: bool) -> NoReturn:
        self.__public_ids = public_ids

    def __repr__(self) -> str:
        return '<AirEnv: air_objects={}>'.format(self.__air_objects)

    def __str__(self) -> str:
        return self.__repr__()
