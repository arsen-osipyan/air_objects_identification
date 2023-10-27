import numpy as np
import time

from airsim.collections import AirObject
from airsim.time import Time


ao = AirObject(position=np.array([0,0,10000], dtype=float),
               velocity=np.array([400,0,0], dtype=float),
               acceleration=np.array([-1,1,0], dtype=float))

t = Time()
t.set(0.0)

while t.get() <= 300:
    print(f'\nTime: {t.get()}')
    print(f'Before trigger: {ao}')

    ao.trigger()

    print(f'After trigger: {ao}')

    t.step(1.0)
    time.sleep(0.01)

