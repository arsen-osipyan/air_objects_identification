import time
import pandas as pd
from typing import NoReturn, List

from .model import Model
from .radarsystem import RadarSystem
from .algorithms import identify_air_objects_nn, identify_air_objects_determined


class ControlPoint(Model):

    def __init__(self, radar_systems: List[RadarSystem] = None, uploading_period: int = 10000,
                 uploading_delay: int = 0) -> NoReturn:
        super().__init__()

        self.__uploading_period = uploading_period
        self.__uploading_delay = uploading_delay % uploading_period

        self.__radar_systems = dict()
        self.__radar_system_id_incr = 0

        if radar_systems is not None:
            self.__radar_systems = {i: radar_systems[i] for i in range(len(radar_systems))}
            self.__radar_system_id_incr = len(radar_systems)

        self.__data_dtypes = {
            'rs_id': 'int64',
            'id': 'int64',
            'time': 'int64',
            'x': 'float64',
            'y': 'float64',
            'z': 'float64',
            'x_err': 'float64',
            'y_err': 'float64',
            'z_err': 'float64',
            'air_object_id': 'int64',
            'load_time': 'int64'
        }
        self.__data = pd.DataFrame(columns=list(self.__data_dtypes.keys())).astype(self.__data_dtypes)
        self.__last_load_time = None
    
    def trigger(self) -> NoReturn:
        if self.time.get() % self.__uploading_period == self.__uploading_delay:
            self.upload_data()

    def identify_air_objects_nn(self):
        start_time = time.time()
        identify_air_objects_nn(self.__data)
        end_time = time.time()
        print('Elapsed time: ', end_time - start_time)

    def identify_air_objects_determined(self):
        identify_air_objects_determined(self.__data)

    def upload_data(self) -> NoReturn:
        current_time = self.time.get()

        for k, v in self.__radar_systems.items():
            rs_data = v.get_data()
            if self.__last_load_time is not None:
                rs_data = rs_data[rs_data['time'] > self.__last_load_time]
            if len(rs_data) != 0:
                rs_data.loc[:, ['load_time']] = int(current_time)
                rs_data.loc[:, ['rs_id']] = int(k)
                rs_data.loc[:, ['air_object_id']] = -1
                self.__concat_data(rs_data)

        self.__last_load_time = current_time

        self.__extend_tracks_ids()

    def __extend_tracks_ids(self):
        for rs_id in self.__data['rs_id'].unique():
            for id in self.__data[self.__data['rs_id'] == rs_id]['id'].unique():
                ao_ids = self.__data[(self.__data['rs_id'] == rs_id) &
                                     (self.__data['id'] == id)]['air_object_id'].unique()

                if len(ao_ids) >= 2 and -1 not in ao_ids:
                    raise RuntimeError(f'identification error, track ({rs_id}, {id}) has multiple ids - {ao_ids}')

                if len(ao_ids) == 2:
                    self.__data.loc[(self.__data['rs_id'] == rs_id) & (self.__data['id'] == id) &
                                    (self.__data['air_object_id'] == -1), 'air_object_id'] = max(ao_ids)

    def __concat_data(self, df: pd.DataFrame) -> NoReturn:
        df = df[list(self.__data_dtypes.keys())].astype(self.__data_dtypes)
        if len(self.__data) == 0:
            self.__data = df
        else:
            self.__data = pd.concat([self.__data, df])
            self.__data.reset_index(inplace=True, drop=True)

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
        return self.__data.copy()

    def clear_data(self) -> NoReturn:
        self.__data = self.__data.iloc[0:0]
        self.__last_load_time = None

    def __repr__(self) -> str:
        return '<ControlPoint: radar_systems={}>'.format(
            self.__radar_systems
        )

    def __str__(self) -> str:
        return self.__repr__()
