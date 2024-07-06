import pandas as pd
from tqdm import tqdm
from typing import List, NoReturn

from .time import Time
from .collections import AirObject, AirEnv, RadarSystem, ControlPoint


class SimulationManager:

    def __init__(self, air_env: AirEnv = None, radar_systems: List[RadarSystem] = None,
                 control_point: ControlPoint = None, control_point_triggering: bool = False,
                 identification_method: str = 'determined') -> NoReturn:
        self.__air_env = air_env
        self.__radar_systems = radar_systems
        for rs in self.__radar_systems:
            rs.set_air_environment(self.__air_env)

        if control_point is None:
            self.__control_point = ControlPoint(radar_systems=self.__radar_systems,
                                                identification_method=identification_method)
        else:
            self.__control_point = control_point
            self.__control_point.set_identification_method(identification_method)

        self.__control_point_triggering = control_point_triggering

    def run(self, t_min: int, t_max: int, dt: int = 1, progress: bool = False) -> NoReturn:
        '''
        Запуск моделирования в заданном интервале времени с заданным шагом
        :param t_min: начальный момент времени (в мс)
        :param t_max: конечный момент времени (в мс)
        :param dt: шаг по времени (в мс)
        :param progress: показывать прогресс процесса моделирования (если True) или нет (если False)
        '''
        t = Time()
        t.set(t_min)

        if progress:
            progressbar = tqdm(range(t_min, t_max + 1, dt), ncols=100)
            progressbar.set_description('Running system')
        else:
            progressbar = range(t_min, t_max + 1, dt)
        for _ in progressbar:
            for radar_system in self.__radar_systems:
                radar_system.trigger()
            if self.__control_point_triggering:
                self.__control_point.trigger()

            t.step(dt)

        self.__control_point.upload_data()

    def identify_air_objects(self) -> NoReturn:
        '''
        Запуск алгоритма отождествления на ПУ
        '''
        self.__control_point.identify_air_objects()

    def get_data(self) -> pd.DataFrame:
        '''
        Получение данных ПУ
        '''
        return self.__control_point.get_data()

    def clear_data(self) -> NoReturn:
        '''
        Очистка данных ПУ и всех РЛС в системе
        '''
        self.__control_point.clear_data()

        for rs in self.__radar_systems:
            rs.clear_data()
