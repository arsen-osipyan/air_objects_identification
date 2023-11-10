import pandas as pd
from .model import Model
from typing import NoReturn
from .radarsystem import RadarSystem


class ControlPoint(Model):

    def __init__(self) -> NoReturn:
        super().__init__()
        self.__radar_systems = dict()
        self.__radar_system_id_incr = 0

        self.__data = pd.DataFrame(
            columns=['load_time', 'radar_system_id', 'detection_time', 'detection_ao_id', 'detection_id',
                     'detection_error', 'detection_x', 'detection_y', 'detection_z']
        )
        self.__last_load_time = None
    
    def trigger(self) -> NoReturn:
        super().trigger()

        self.upload_data()
        self.identify_air_objects_algorithm()
        self.identify_air_objects_rnn()

    def identify_air_objects_algorithm(self) -> NoReturn:
        pass

    def identify_air_objects_rnn(self) -> NoReturn:
        pass

    def upload_data(self) -> NoReturn:
        current_time = self.time.get()

        for k, v in self.__radar_systems.items():
            rs_data = v.get_detections()
            if self.__last_load_time is not None:
                rs_data = rs_data[rs_data['detection_time'] > self.__last_load_time]
            if len(rs_data) != 0:
                rs_data.loc[:, ['load_time']] = current_time
                rs_data.loc[:, ['radar_system_id']] = k
                self.concat_data(rs_data)

        self.__last_load_time = current_time

    def concat_data(self, df: pd.DataFrame) -> NoReturn:
        df = df[list(self.__data.columns)]
        if len(self.__data) == 0:
            self.__data = df
        else:
            self.__data = pd.concat([self.__data, df])

    def is_attached(self, radar_system: RadarSystem) -> bool:
        return radar_system in self.__radar_systems.values()

    def attach_radar_system(self, radar_system: RadarSystem) -> int:
        if self.is_attached(radar_system):
            raise RuntimeError('RadarSystem already attached to ControlPoint.')
        self.__radar_systems[self.__radar_system_id_incr] = radar_system
        self.__radar_system_id_incr += 1
        return self.__radar_system_id_incr - 1

    def detach_radar_systems(self, radar_system: RadarSystem) -> int:
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
