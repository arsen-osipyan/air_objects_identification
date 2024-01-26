import numpy as np

from airsim.collections import AirObject, AirEnv, RadarSystem, ControlPoint
from airsim.time import Time


ao_list = [
    AirObject(track=lambda x: np.array([100, 0, 10500]) + np.array([223, 0, 0]) * 0.001*x),
    AirObject(track=lambda x: np.array([0, 100, 10500]) + np.array([0, 223, 0]) * 0.001*x),
    AirObject(track=lambda x: np.array([-100, 0, 10500]) + np.array([-223, 0, 0]) * 0.001*x),
    AirObject(track=lambda x: np.array([0, -100, 10500]) + np.array([0, -223, 0]) * 0.001*x),
]
ae = AirEnv(air_objects=ao_list)
rs_list = [
    RadarSystem(position=np.array([0, 0, 0]), detection_radius=100000, error=1, air_env=ae)
]
cp = ControlPoint(radar_systems=rs_list)

models = [ae] + rs_list + [cp]
print(models)

t = Time()
t.set(0)

while t.get() < 1000*60:
    for model in models:
        model.trigger()

    t.step(1000)

print(cp.get_data())
