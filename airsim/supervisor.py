import pandas as pd
from tqdm import tqdm
from typing import List, NoReturn

from .time import Time
from .collections import AirObject, AirEnv, RadarSystem, ControlPoint


class Supervisor:

    def __init__(self, air_objects: List[AirObject] = None, radar_systems: List[RadarSystem] = None) -> NoReturn:
        self.__air_objects = air_objects
        self.__air_env = AirEnv(air_objects=self.__air_objects)
        self.__radar_systems = radar_systems
        for rs in self.__radar_systems:
            rs.set_air_environment(self.__air_env)
        self.__control_point = ControlPoint(radar_systems=self.__radar_systems)

    def run(self, t_min: int, t_max: int, dt: int) -> NoReturn:
        t = Time()
        t.set(t_min)

        progressbar = tqdm(range(t_min, t_max + 1, dt), ncols=100)
        progressbar.set_description('Running system')
        for _ in progressbar:
            for radar_system in self.__radar_systems:
                radar_system.trigger()

            t.step(dt)

        for radar_system in self.__radar_systems:
            radar_system.estimate_velocity()
            radar_system.estimate_acceleration()
        self.__control_point.upload_data()

    def get_data(self) -> pd.DataFrame:
        return self.__control_point.get_data()

    def clear_data(self) -> NoReturn:
        self.__control_point.clear_data()
        for rs in self.__radar_systems:
            rs.clear_data()
