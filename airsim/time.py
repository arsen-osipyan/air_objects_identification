def singleton(cls):
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance


@singleton
class Time:
    def __init__(self):
        self.__t = 0.0

    def get(self):
        return self.__t

    def set(self, t):
        self.__t = t

    def step(self, dt):
        self.__t += dt
