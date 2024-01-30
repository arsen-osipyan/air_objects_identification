import numpy as np
import pandas as pd
import torch

from airsim.collections import AirObject, AirEnv, RadarSystem, ControlPoint
from airsim.time import Time


t_min = 0      # ms
t_max = 60001  # ms
dt    = 200    # ms
air_objects = [
    AirObject(track=lambda t: 1000 * np.array([np.cos(2*np.pi*t/(t_max - t_min)), np.sin(2*np.pi*t/(t_max - t_min)), 1])),
    AirObject(track=lambda t: 1000 * np.array([np.cos(2*np.pi*t/(t_max - t_min)), np.sin(2*np.pi*t/(t_max - t_min)), 2])),
    AirObject(track=lambda t: 1000 * np.array([np.cos(2*np.pi*t/(t_max - t_min)), np.sin(2*np.pi*t/(t_max - t_min)), 3])),
    AirObject(track=lambda t: 1000 * np.array([np.cos(2*np.pi*t/(t_max - t_min)), np.sin(2*np.pi*t/(t_max - t_min)), 4]))
]
ae = AirEnv(air_objects=air_objects)
radar_systems = [
    RadarSystem(position=np.array([0, 0, 0]), detection_radius=50000, error=0.5, air_env=ae),
    RadarSystem(position=np.array([0, 0, 2500]), detection_radius=50000, error=1, air_env=ae),
    RadarSystem(position=np.array([0, 0, 5000]), detection_radius=50000, error=2, air_env=ae),
]
cp = ControlPoint(radar_systems=radar_systems)
models = [ae] + radar_systems + [cp]

t = Time()
t.set(t_min)

while t.get() < t_max:
    for model in models:
        model.trigger()

    t.step(dt)

data = cp.get_data()


def _generate_past_new_detections_cross_join(data, t):
    detection_columns = [
        'time',
        'x', 'y', 'z',
        'x_err', 'y_err', 'z_err',
        'v_x_est', 'v_y_est', 'v_z_est',
        'a_x_est', 'a_y_est', 'a_z_est'
    ]

    dfl = data.loc[data['time'] < t].copy()
    dfl = dfl.sort_values(by=['id', 'time', 'err_ratio'], ascending=[True, False, False])
    dfl = dfl.drop_duplicates(subset=['id', 'time'])
    dfl = dfl.groupby(by=['id']).head(1)
    dfl = dfl.rename(columns={f'{col}': f'{col}_1' for col in detection_columns})
    dfl = dfl[['id'] + [f'{col}_1' for col in detection_columns]]
    dfl = dfl.reset_index(drop=True)

    dfr = data.loc[data['time'] == t].copy()
    dfr = dfr.sort_values(by=['id', 'err_ratio'], ascending=[True, False])
    # dfr = dfr.drop_duplicates(subset=['id', 'time'])
    # dfr = dfr.groupby(by=['id']).head(1)
    dfr = dfr[['id'] + detection_columns]
    dfr = dfr.reset_index(drop=True)

    df = pd.merge(dfl, dfr, how='cross')
    df['is_identical'] = df['id_x'] == df['id_y']
    df = df.astype({'is_identical': 'float64'})
    df = df.drop(columns=['id_x', 'id_y'])

    return df


def generate_past_new_detections_cross_join(data, timestamps=None):
    dfs = []
    data['err_ratio'] = np.sqrt(3.0 / (data['x_err'] ** 2 + data['y_err'] ** 2 + data['z_err'] ** 2))
    if timestamps is None:
        for t in sorted(set(data['time']))[3:]:
            df_t = _generate_past_new_detections_cross_join(data, t)
            dfs.append(df_t)
    else:
        for t in timestamps:
            df_t = _generate_past_new_detections_cross_join(data, t)
            dfs.append(df_t)
    return pd.concat(dfs).reset_index(drop=True)

train_df = generate_past_new_detections_cross_join(data=data)
