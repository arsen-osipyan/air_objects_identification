import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import NoReturn

from airsim.collections import AirObject, RadarSystem, Supervisor


class DataGenerator:

    def __init__(self) -> NoReturn:
        pass

    def generate_1ao_2rs(self, ao_track, t_min, t_max, dt):
        supervisor = Supervisor(
            air_objects=[
                AirObject(track=ao_track)
            ],
            radar_systems=[
                RadarSystem(position=np.array([0, 0, 0]), detection_radius=1e+308, error=0.1),
                RadarSystem(position=np.array([0, 0, 0]), detection_radius=1e+308, error=0.1)
            ]
        )
        supervisor.run(t_min=t_min, t_max=t_max, dt=dt)
        return self.__get_past_new_detections_cross_join(supervisor.get_data())

    def __get_past_new_detections_cross_join_by_time(self, data, t):
        detection_columns = ['time', 'x', 'y', 'z', 'x_err', 'y_err', 'z_err']
        v_est_columns = [f'v_{axis}_est' for axis in ('x', 'y', 'z')]
        a_est_columns = [f'a_{axis}_est' for axis in ('x', 'y', 'z')]

        suffix = '_'
        cols = detection_columns + v_est_columns + a_est_columns
        cols_new = [f'{el}{suffix}' for el in (detection_columns + v_est_columns + a_est_columns)]
        dfl = data.loc[data['time'] < t].copy()
        dfl = dfl.sort_values(by=['id', 'time', 'err_ratio'], ascending=[True, False, False])
        dfl = dfl.drop_duplicates(subset=['id', 'time'])
        dfl = dfl.groupby(by=['id']).head(1)
        dfl = dfl.rename(columns={cols[i]: cols_new[i] for i in range(len(cols))})
        dfl = dfl[['id'] + cols_new]
        dfl = dfl.reset_index(drop=True)

        dfr = data.loc[data['time'] == t].copy()
        dfr = dfr.sort_values(by=['id', 'err_ratio'], ascending=[True, False])
        dfr = dfr[['id'] + detection_columns]
        dfr = dfr.reset_index(drop=True)

        df = pd.merge(dfl, dfr, how='cross')
        df['is_identical'] = df['id_x'] == df['id_y']
        df = df.astype({'is_identical': 'float64'})
        df = df.drop(columns=['id_x', 'id_y'])

        return df

    def __get_past_new_detections_cross_join(self, data):
        dfs = []
        data['err_ratio'] = np.sqrt(3.0 / (data['x_err'] ** 2 + data['y_err'] ** 2 + data['z_err'] ** 2))

        timestamps = sorted(set(data['time']))[1:]
        progressbar = tqdm(timestamps, ncols=100)
        progressbar.set_description('Generating data')
        for t in progressbar:
            df_t = self.__get_past_new_detections_cross_join_by_time(data, t)
            dfs.append(df_t)

        df = pd.concat(dfs).reset_index(drop=True)
        return df


def main():
    dg = DataGenerator()

    def f(t):
        if t <= 15000:
            return np.array([t, 0, 10000])
        elif t <= 30000:
            return np.array([30000 - t, 0, 10000])
        elif t <= 45000:
            return np.array([0.5 * (t - 30000), 0, 10000])
        elif t <= 60000:
            return np.array([15000 - 0.5 * (t - 30000), 0, 10000])
    df = dg.generate_1ao_2rs(ao_track=f, t_min=0, t_max=60000, dt=1)
    df.to_csv('data/1ao_2rs_f.csv', index=False)

    df = dg.generate_1ao_2rs(ao_track=lambda t: np.array([0.7 * t, 0, 10000]), t_min=0, t_max=1000, dt=1)
    df.to_csv('data/1ao_2rs_g.csv', index=False)

    df = dg.generate_1ao_2rs(ao_track=lambda t: np.array([1.5 * t, 0, 10000]), t_min=0, t_max=1000, dt=1)
    df.to_csv('data/1ao_2rs_h.csv', index=False)


if __name__ == "__main__":
    main()
