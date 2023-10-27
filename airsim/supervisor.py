from .time import Time


class Supervisor:

    def __init__(self, t_min, t_max, dt):
        self.__t_min = t_min
        self.__t_max = t_max
        self.__dt = dt

        self.__air_env = None
        self.__radar_systems = []
        self.__control_point = None

    def attach_air_environment(self, air_env):
        self.__air_env = air_env

    def attach_radar_systems(self, radar_systems):
        self.__radar_systems = radar_systems

    def attach_control_point(self, control_point):
        self.__control_point = control_point

    def trigger_models(self):
        self.__air_env.trigger()
        for i in range(len(self.__radar_systems)):
            self.__radar_systems[i].trigger()
        self.__control_point.trigger()

    def run(self):
        t = Time()
        t.set(self.__t_min)

        while t.get() <= self.__t_max:
            self.trigger_models()

            t.step(self.__dt)