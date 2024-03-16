import pandas as pd
from tqdm import tqdm
from typing import List, NoReturn

from .time import Time
from .collections import AirObject, AirEnv, RadarSystem, ControlPoint


class Supervisor:

    def __init__(self, air_env: AirEnv = None, radar_systems: List[RadarSystem] = None,
                 control_point: ControlPoint = None, control_point_triggering: bool = False) -> NoReturn:
        self.__air_env = air_env
        self.__radar_systems = radar_systems
        for rs in self.__radar_systems:
            rs.set_air_environment(self.__air_env)

        if control_point is None:
            self.__control_point = ControlPoint(radar_systems=self.__radar_systems)
        else:
            self.__control_point = control_point

        self.__control_point_triggering = control_point_triggering

    def run(self, t_min: int, t_max: int, dt: int = 1) -> NoReturn:
        t = Time()
        t.set(t_min)

        progressbar = tqdm(range(t_min, t_max + 1, dt), ncols=100)
        progressbar.set_description('Running system')
        for _ in progressbar:
            for radar_system in self.__radar_systems:
                radar_system.trigger()
            if self.__control_point_triggering:
                self.__control_point.trigger()

            t.step(dt)

        self.__control_point.upload_data()

    def get_data(self) -> pd.DataFrame:
        return self.__control_point.get_data()

    def clear_data(self) -> NoReturn:
        self.__control_point.clear_data()
        for rs in self.__radar_systems:
            rs.clear_data()
