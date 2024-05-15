import numpy as np
import pandas as pd

from airsim.collections import AirObject, AirEnv, RadarSystem, ControlPoint, Supervisor
from airsim.time import Time


t_min = 0
t_max = 5 * 125000
dt = 250

supervisor = Supervisor(
    air_env=AirEnv(air_objects=[
        # AirObject(track=lambda t: np.array([-50000 + 0.2 * t + 0.15 * t**2 / 625000, 200, 30000])),
        # AirObject(track=lambda t: np.array([-50000 + 0.2 * t + 0.10 * t**2 / 625000, -200, 30000])),
        AirObject(track=lambda t: np.array([-50000 + 0.2 * t, 10, 30000])),
        AirObject(track=lambda t: np.array([-50000 + 0.2 * t, -10, 30000])),
    ]),
    radar_systems=[
        RadarSystem(position=np.array([0, 0, 0]),
                    detection_radius=50000,
                    error=1.0,
                    detection_fault_probability=0.01,
                    detection_period=1000,
                    detection_delay=np.random.randint(0, 250//dt) * dt),
        RadarSystem(position=np.array([50000, 0, 0]),
                    detection_radius=50000,
                    error=1.0,
                    detection_fault_probability=0.01,
                    detection_period=1000,
                    detection_delay=np.random.randint(0, 250//dt) * dt),
        RadarSystem(position=np.array([100000, 0, 0]),
                    detection_radius=50000,
                    error=1.0,
                    detection_fault_probability=0.01,
                    detection_period=1000,
                    detection_delay=np.random.randint(0, 250//dt) * dt),
        RadarSystem(position=np.array([150000, 0, 0]),
                    detection_radius=50000,
                    error=1.0,
                    detection_fault_probability=0.01,
                    detection_period=1000,
                    detection_delay=np.random.randint(0, 250//dt) * dt),
    ],
    identification_method='determined'
)

supervisor.run(t_min, t_max, dt)

cp_data = supervisor.get_data()

print(cp_data)

for rs_id in cp_data['rs_id'].unique():
    for id in cp_data[cp_data['rs_id'] == rs_id]['id'].unique():
        print(f'radar_system: {rs_id}, air_object: {id}, ids: ', end='')
        print(cp_data[(cp_data['rs_id'] == rs_id) & (cp_data['id'] == id)]['air_object_id'].unique())
