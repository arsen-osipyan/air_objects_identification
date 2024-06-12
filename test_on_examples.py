import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from airsim.identification.determined import identify_air_objects_determined
from airsim.identification.nn import identify_air_objects_nn


def prepare_data(df):
    df = df.astype(str)

    to_int = ['rs_id', 'id', 'id_in_cp']
    df[to_int] = df[to_int].astype(float).fillna(-1.0)
    df[to_int] = df[to_int].astype(int)

    to_float = ['time', 'x', 'y', 'z', 'x_std2', 'y_std2', 'z_std2']
    df[to_float] = df[to_float].astype(str)
    for col in to_float:
        df[col] = [x.replace(',', '.') for x in df[col]]
    df[to_float] = df[to_float].astype(float)

    df['x_err'] = np.sqrt(df['x_std2'])
    df['y_err'] = np.sqrt(df['y_std2'])
    df['z_err'] = np.sqrt(df['z_std2'])

    df = df[['rs_id', 'id', 'time', 'x', 'y', 'z', 'x_err', 'y_err', 'z_err']]
    # df = df.rename({'y': 'z', 'z': 'y', 'y_err': 'z_err', 'z_err': 'y_err'})

    df.loc[:, 'air_object_id'] = -1

    return df


print('\n' * 20)

for file in os.listdir('examples'):
    f = os.path.join('examples', file)
    if os.path.isfile(f) and f.endswith('.csv'):  # and f == 'examples/tracks_29_44_50_55.csv':
        print('File ', f)

        data = pd.read_csv(f)
        data = prepare_data(data)

        # fig, ax = plt.subplots(figsize=(8, 8))
        #
        # ax.set_title('Траектории ' + ', '.join(file.split('.')[0].split('_')[1:]) + ' в плоскости XoZ', fontsize=20)
        # ax.set_xlabel('z, м', fontsize=16)
        # ax.set_ylabel('x, м', fontsize=16)
        #
        # for rs_id in data['rs_id'].unique():
        #     for id in data[data['rs_id'] == rs_id]['id'].unique():
        #         track = (data[(data['rs_id'] == rs_id) & (data['id'] == id)]
        #                  .sort_values(by=['time']).reset_index(drop=True))
        #         ax.scatter(track['z'].head(1), track['x'].head(1))
        #         ax.plot(track['z'], track['x'], label=f'Траектория {id} (РЛС {rs_id + 1})', linewidth=2.5)
        #
        # ax.grid()
        # ax.legend(loc='best', fontsize=16)
        # fig.savefig(f'examples/{file[:-4]}.png')
        # plt.close()

        fig, axis = plt.subplots(1, 2, figsize=(12, 6))

        for rs_id in data['rs_id'].unique():
            for id in data[data['rs_id'] == rs_id]['id'].unique():
                track = (data[(data['rs_id'] == rs_id) & (data['id'] == id)]
                         .sort_values(by=['time']).reset_index(drop=True))

                axis[0].set_title('Траектории ' + ', '.join(file.split('.')[0].split('_')[1:]) + ' в плоскости XoZ')
                axis[0].set_xlabel('z, м')
                axis[0].set_ylabel('x, м')

                axis[1].set_title('Y(t) траекторий ' + ', '.join(file.split('.')[0].split('_')[1:]))
                axis[1].set_xlabel('t, с')
                axis[1].set_ylabel('y, м')

                axis[0].scatter(track['z'].head(1), track['x'].head(1))
                axis[0].plot(track['z'], track['x'], label=f'Траектория {id} (РЛС {rs_id + 1})')

                axis[1].plot(track['time'], track['y'], label=f'Y(t) траектории {id} (РЛС {rs_id + 1})')

        axis[0].grid()
        axis[0].legend(loc='best')
        axis[1].grid()
        axis[1].legend(loc='best')
        fig.savefig(f'examples/{file[:-4]}.png')

        data_1, data_2 = data.copy(), data.copy()

        print('- Determined')
        identify_air_objects_determined(data_1)

        for rs_id in data_1['rs_id'].unique():
            for id in data_1[data_1['rs_id'] == rs_id]['id'].unique():
                print(f'radar_system: {rs_id}, air_object: {id}, ids: ', end='')
                print(data_1[(data_1['rs_id'] == rs_id) & (data_1['id'] == id)]['air_object_id'].unique())

        print('- NN')
        identify_air_objects_nn(data_2)

        for rs_id in data_2['rs_id'].unique():
            for id in data_2[data_2['rs_id'] == rs_id]['id'].unique():
                track = (data_2[(data_2['rs_id'] == rs_id) & (data_2['id'] == id)]
                         .sort_values(by=['time']).reset_index(drop=True))
                print(f'rs: {rs_id}, id: {id}, identified ids: ', end='')
                print(data_2[(data_2['rs_id'] == rs_id) & (data_2['id'] == id)]['air_object_id'].unique())

        print()