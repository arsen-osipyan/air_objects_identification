from typing import NoReturn
import time


def singleton(cls):
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance


@singleton
class Time:
    def __init__(self) -> NoReturn:
        # print(f'Time initialized: 0ms')
        self.__t = 0

    def get(self) -> int:
        return self.__t

    def set(self, t: int) -> NoReturn:
        # print(f'Time set: {self.__t}ms -> {t}ms')
        self.__t = t

    def step(self, dt: int) -> NoReturn:
        # print(f'Time step: {self.__t}ms -> {self.__t + dt}ms (+{dt}ms)')
        self.__t += dt
