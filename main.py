import numpy as np
import time

from airsim.time import Time
from airsim.collections import AirObject, AirEnv, RadarSystem, ControlPoint, Supervisor


air_objects = [
    AirObject(position=np.array([0, 0, 10000], dtype=float),
              velocity=np.array([400, 0, -20], dtype=float),
              acceleration=np.array([0, 0, -1], dtype=float)),
    AirObject(position=np.array([0, 0, 4000], dtype=float),
              velocity=np.array([500, 0, 50], dtype=float),
              acceleration=np.array([0, 0, 10], dtype=float))
]

radar_systems = [
    RadarSystem(position=np.array([0, 0, 0], dtype=float),
                detection_radius=100000,
                error=10)
]

air_env = AirEnv()
for i in range(len(air_objects)):
    air_env.attach_air_object(air_objects[i])

control_point = ControlPoint()
for i in range(len(radar_systems)):
    radar_systems[i].attach_air_environment(air_env)
    control_point.attach_radar_system(radar_systems[i])

supervisor = Supervisor(0.0, 10.0, 1.0)
supervisor.attach_air_environment(air_env)
supervisor.attach_control_point(control_point)
supervisor.attach_radar_systems(radar_systems)

supervisor.run()

print(control_point.get_data())
