import numpy as np
import pandas as pd

from airsim.collections import AirObject, AirEnv, RadarSystem, ControlPoint, Supervisor
from airsim.time import Time


t_min = 0
t_max = 5000
dt = 50

supervisor = Supervisor(
    air_env=AirEnv(air_objects=[
        AirObject(track=lambda t: np.array([20.0, 0.3 * t, 10000.0])),
        AirObject(track=lambda t: np.array([0.3 * t, 0.0, 10000.0])),
    ]),
    radar_systems=[
        RadarSystem(position=np.array([0, 0, 0]),
                    detection_radius=1e+308,
                    error=1.0,
                    detection_fault_probability=0.01,
                    detection_period=250,
                    detection_delay=np.random.randint(0, 250//dt) * dt),
        RadarSystem(position=np.array([0, 0, 0]),
                    detection_radius=1e+308,
                    error=1.0,
                    detection_fault_probability=0.01,
                    detection_period=250,
                    detection_delay=np.random.randint(0, 250//dt) * dt),
    ]
)

supervisor.run(t_min, t_max, dt)

print(supervisor.get_data())
