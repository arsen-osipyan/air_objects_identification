import numpy as np
import pandas as pd

from airsim.collections import AirObject, AirEnv, RadarSystem, ControlPoint, Supervisor
from airsim.time import Time


t_min = 0
t_max = 5 * 125000
dt = 250

supervisor = Supervisor(
    air_env=AirEnv(air_objects=[
        AirObject(track=lambda t: np.array([-50000 + 0.4 * t, 2, 30000])),
        AirObject(track=lambda t: np.array([-50000 + 0.4 * t, -3, 30000])),
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
    ]
)

supervisor.run(t_min, t_max, dt)

cp_data = supervisor.get_data()

print(cp_data)


def can_build_one_to_one_mapping(arr1, arr2):
    if len(arr1) != len(arr2):
        return False

    mapping = {}

    for i in range(len(arr1)):
        if arr1[i] in mapping:
            if mapping[arr1[i]] != arr2[i]:
                return False
        else:
            mapping[arr1[i]] = arr2[i]

    return True


print(can_build_one_to_one_mapping(cp_data['id'], cp_data['air_object_id']))
