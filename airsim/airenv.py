import pandas as pd
from typing import NoReturn, List

from .model import Model
from .airobject import AirObject


class AirEnv(Model):

    def __init__(self, air_objects: List[AirObject] = None) -> NoReturn:
        super().__init__()
        self.__air_objects = dict()
        self.__air_object_id_incr = 0

        if air_objects is not None:
            self.__air_objects = {i: air_objects[i] for i in range(len(air_objects))}
            self.__air_object_id_incr = len(air_objects)

    def trigger(self) -> NoReturn:
        """
        Вызов метода trigger() для всех ВО
        """
        for ao_id, ao in self.__air_objects.items():
            ao.trigger()

    def is_attached(self, air_object: AirObject) -> bool:
        return air_object in self.__air_objects.values()

    def attach_air_object(self, air_object: AirObject) -> int:
        if self.is_attached(air_object):
            raise RuntimeError('AirObject already attached to AirEnv.')
        self.__air_objects[self.__air_object_id_incr] = air_object
        self.__air_object_id_incr += 1
        return self.__air_object_id_incr - 1

    def detach_air_object(self, air_object: AirObject) -> int:
        if not self.is_attached(air_object):
            raise RuntimeError('AirObject is not attached to AirEnv.')
        for k, v in self.__air_objects.items():
            if v == air_object:
                self.__air_objects.pop(k, None)
                return k

    def air_objects_dataframe(self) -> pd.DataFrame:
        """
        Для текущего момента модельного времени формируется таблица положений всех ВО
        :return: pd.DataFrame - таблица
        """
        data = pd.DataFrame(columns=['id', 'x', 'y', 'z'])
        for ao_id, ao in self.__air_objects.items():
            data.loc[len(data)] = {
                'id': ao_id,
                'x': ao.position()[0],
                'y': ao.position()[1],
                'z': ao.position()[2]
            }
        return data

    def __repr__(self) -> str:
        return '<AirEnv: len(air_objects)={}>'.format(len(self.__air_objects))

    def __str__(self) -> str:
        return self.__repr__()
