from typing import NoReturn

from .time import Time


class Model:

    def __init__(self) -> NoReturn:
        self.time = Time()

    def trigger(self, **kwargs) -> NoReturn:
        raise NotImplementedError()
