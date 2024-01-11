import pandas as pd
from .model import Model
from typing import NoReturn
from .radarsystem import RadarSystem


class ControlPoint(Model):

    def __init__(self) -> NoReturn:
        """
        Initializes ControlPoint
        """
        super().__init__()
        self.__radar_systems = dict()
        self.__radar_system_id_incr = 0

        self.__data = pd.DataFrame(
            columns=['load_time', 'load_time_id', 'radar_system_id', 'detection_time', 'air_object_id',
                     'detection_id', 'detection_error', 'air_object_x', 'air_object_y', 'air_object_z']
        )
        self.__last_load_time = None
        self.__load_time_id_incr = 0
    
    def trigger(self) -> NoReturn:
        """
        Runs upload_data() method
        """
        super().trigger()

        self.upload_data()
        # self.identify_air_objects_algorithm()
        # self.identify_air_objects_rnn()

    def identify_air_objects_algorithm(self) -> NoReturn:
        """
        Fills None values in data in column 'detection_ao_id' using algorithm
        """
        pass

    def identify_air_objects_rnn(self) -> NoReturn:
        """
        Fills None values in data in column 'detection_ao_id' using RNN
        """
        pass

    def upload_data(self) -> NoReturn:
        """
        Uploads data from last load time to inner dataframe
        """
        current_time = self.time.get()

        for k, v in self.__radar_systems.items():
            rs_data = v.get_detections()
            if self.__last_load_time is not None:
                rs_data = rs_data[rs_data['detection_time'] > self.__last_load_time]
            if len(rs_data) != 0:
                rs_data.loc[:, ['load_time']] = current_time
                rs_data.loc[:, ['load_time_id']] = self.__load_time_id_incr
                rs_data.loc[:, ['radar_system_id']] = int(k)
                self.__concat_data(rs_data)

        self.__last_load_time = current_time
        self.__load_time_id_incr += 1

    def __concat_data(self, df: pd.DataFrame) -> NoReturn:
        """
        Concats inner dataframe and provided and replaces it if inner is empty
        :param df: dataframe to concat with
        """
        df = df[list(self.__data.columns)]
        if len(self.__data) == 0:
            self.__data = df
        else:
            df.index += len(self.__data)
            self.__data = pd.concat([self.__data, df])

    def is_attached(self, radar_system: RadarSystem) -> bool:
        """
        Checks if provided RadarSystem is attached to ControlPoint
        :param radar_system: RadarSystem to check
        :return: attachment status
        """
        return radar_system in self.__radar_systems.values()

    def attach_radar_system(self, radar_system: RadarSystem) -> int:
        """
        Attaches RadarSystem if it wasn't attached
        :param radar_system: RadarSystem to attach
        :return: RadarSystem inner id
        """
        if self.is_attached(radar_system):
            raise RuntimeError('RadarSystem already attached to ControlPoint.')
        self.__radar_systems[self.__radar_system_id_incr] = radar_system
        self.__radar_system_id_incr += 1
        return self.__radar_system_id_incr - 1

    def detach_radar_systems(self, radar_system: RadarSystem) -> int:
        """
        Detaches RadarSystem if it was attached
        :param radar_system: RadarSystem to detach
        :return: RadarSystem inner id
        """
        if self.is_attached(radar_system):
            raise RuntimeError('RadarSystem is not attached to ControlPoint.')
        for k, v in self.__radar_systems.items():
            if v == radar_system:
                self.__radar_systems.pop(k, None)
                return k

    def get_data(self) -> pd.DataFrame:
        return self.__data

    def __repr__(self) -> str:
        return '<ControlPoint: radar_systems={}>'.format(
            self.__radar_systems
        )

    def __str__(self) -> str:
        return self.__repr__()
