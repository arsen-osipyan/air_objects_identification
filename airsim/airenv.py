from .model import Model
import pandas as pd


class AirEnv(Model):

    def __init__(self):
        super().__init__()
        self.__air_objects = []

    def trigger(self):
        """
        Trigger all air objects in list
        """
        super().trigger()

        for i in range(len(self.__air_objects)):
            self.__air_objects[i].trigger()

    def attach_air_object(self, air_object):
        """
        Attach air object if it doesn't exist in air objects list
        :param air_object: AirObject
        """
        if air_object in self.__air_objects:
            raise RuntimeError(f'AirObject already attached to this AirEnv.')
        self.__air_objects.append(air_object)

    def detach_air_object(self, air_object):
        """
        Detach air object if it exists in air objects list
        :param air_object: AirObject
        """
        if air_object not in self.__air_objects:
            raise RuntimeError(f'AirObject is not attached to this AirEnv.')
        return self.__air_objects.remove(air_object)

    def air_objects(self):
        """
        Get current air objects positions
        :return: pd.DataFrame(['x', 'y', 'z'])
        """
        data = pd.DataFrame(columns=['x', 'y', 'z'])
        for ao in self.__air_objects:
            data.loc[len(data)] = {
                'x': ao.position()[0],
                'y': ao.position()[1],
                'z': ao.position()[2]
            }
        return data

    def __repr__(self):
        return '<AirEnv: air_objects={}>'.format(self.__air_objects)

    def __str__(self):
        return self.__repr__()
