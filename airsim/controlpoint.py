import pandas as pd
from typing import NoReturn, List

from .model import Model
from .radarsystem import RadarSystem


class ControlPoint(Model):

    def __init__(self, radar_systems: List[RadarSystem] = None) -> NoReturn:
        """
        Initializes ControlPoint
        """
        super().__init__()
        self.__radar_systems = dict()
        self.__radar_system_id_incr = 0

        if radar_systems is not None:
            self.__radar_systems = {i: radar_systems[i] for i in range(len(radar_systems))}
            self.__radar_system_id_incr = len(radar_systems)

        self.__data_dtypes = {
            'time': 'int64',
            'rs_id': 'int64',
            'x': 'float64',
            'x_err': 'float64',
            'y': 'float64',
            'y_err': 'float64',
            'z': 'float64',
            'z_err': 'float64',
            'id': 'int64',
            'load_time': 'int64'
        }
        self.__data = pd.DataFrame(columns=list(self.__data_dtypes.keys())).astype(self.__data_dtypes)
        self.__last_load_time = None
    
    def trigger(self, **kwargs) -> NoReturn:
        """
        Runs upload_data() method
        """
        super().trigger(**kwargs)

        self.upload_data()
        self.identify_air_objects_alg()
        # self.identify_air_objects_rnn()

        # print(set(self.__data['detection_time']))

    def identify_air_objects_alg(self) -> NoReturn:
        last_detections_time = self.__data['time'].max()

    def identify_air_objects_nn(self) -> NoReturn:
        pass

    def upload_data(self) -> NoReturn:
        current_time = self.time.get()

        for k, v in self.__radar_systems.items():
            rs_data = v.get_detections()
            if self.__last_load_time is not None:
                rs_data = rs_data[rs_data['time'] > self.__last_load_time]
            if len(rs_data) != 0:
                rs_data.loc[:, ['load_time']] = int(current_time)
                rs_data.loc[:, ['rs_id']] = int(k)
                self.__concat_data(rs_data)

        self.__last_load_time = current_time

    def __concat_data(self, df: pd.DataFrame) -> NoReturn:
        df = df[list(self.__data_dtypes.keys())].astype(self.__data_dtypes)
        if len(self.__data) == 0:
            self.__data = df
        else:
            df.index += len(self.__data)
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
