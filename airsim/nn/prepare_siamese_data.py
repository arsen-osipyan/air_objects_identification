import os
import datetime
import pandas as pd
import torch

from hparams import config
from utils import extend_track_with_linear_interpolation
from airsim.utils import get_tracks_timeranges_intersection, get_track


def generate_siamese_data_from_cp_tracks(track_1, track_2, track_length, drop_last=False):
    if track_1.shape[1] != track_2.shape[1]:
        raise RuntimeError(
            f'Tracks must have the same number of columns, got {track_1.shape[1]} and {track_2.shape[1]} instead.')

    x = torch.empty((0, 2, track_1.shape[1], track_length))

    t_min, t_max = get_tracks_timeranges_intersection(track_1, track_2)

    current_t_max = t_max

    break_flag = False
    while current_t_max > t_min:
        # crop tracks to current max time
        track_1 = track_1[track_1['time'] < current_t_max]
        track_2 = track_2[track_2['time'] < current_t_max]

        # get last parts of tracks
        track_1_part = track_1.tail(track_length)
        track_2_part = track_2.tail(track_length)

        # remove last parts of tracks from original tracks
        track_1 = track_1.drop(track_1.tail(track_length).index)
        track_2 = track_2.drop(track_2.tail(track_length).index)

        # transform to tensors
        track_1_part_tensor = torch.tensor(track_1_part.values, dtype=torch.float32)
        track_2_part_tensor = torch.tensor(track_2_part.values, dtype=torch.float32)

        if track_1_part_tensor.size(0) < track_length:
            if drop_last:
                break
            # track_1_part_tensor = extend_track_with_zeros(track_1_part_tensor, track_length)
            track_1_part_tensor = extend_track_with_linear_interpolation(track_1_part_tensor, track_length)
            if track_1_part_tensor is None:
                break
            break_flag = True

        if track_2_part_tensor.size(0) < track_length:
            if drop_last:
                break
            # track_2_part_tensor = extend_track_with_zeros(track_2_part_tensor, track_length)
            track_2_part_tensor = extend_track_with_linear_interpolation(track_2_part_tensor, track_length)
            if track_2_part_tensor is None:
                break
            break_flag = True

        track_1_part_tensor = track_1_part_tensor.transpose(0, 1).unsqueeze(0)
        track_2_part_tensor = track_2_part_tensor.transpose(0, 1).unsqueeze(0)

        x = torch.cat((x, torch.cat((track_1_part_tensor, track_2_part_tensor), 0).unsqueeze(0)))

        if break_flag:
            break

        current_t_max = min(track_1['time'].max(), track_2['time'].max())

    return x


def generate_siamese_data_from_cp_data_file(file, track_length):
    row_length = 4
    x, y = torch.empty((0, 2, row_length, track_length)), torch.empty((0, 1))

    cp_data = pd.read_csv(file)
    cp_data = cp_data.sort_values(by=['rs_id', 'id', 'time']) \
        .reset_index(drop=True)[['rs_id', 'id', 'time', 'x', 'y', 'z']].dropna()

    track_ids = [(rs_id, id) for rs_id in cp_data['rs_id'].unique() for id in
                 cp_data[cp_data['rs_id'] == rs_id]['id'].unique()]
    n = len(track_ids)

    for i in range(n):
        for j in range(i + 1, n):
            rs_id_1, id_1 = track_ids[i]
            rs_id_2, id_2 = track_ids[j]

            if rs_id_1 == rs_id_2:
                continue

            track_1 = get_track(cp_data, rs_id_1, id_1)
            track_2 = get_track(cp_data, rs_id_2, id_2)

            x_cur = generate_siamese_data_from_cp_tracks(track_1, track_2, track_length, drop_last=False)
            y_cur = torch.zeros(x_cur.size(0), 1) if id_1 == id_2 else torch.ones(x_cur.size(0), 1)

            x = torch.cat((x, x_cur), 0)
            y = torch.cat((y, y_cur), 0)

    return x, y


def generate_siamese_data_from_cp_data_dir(dir_name, track_length):
    row_length = 4
    x, y = torch.empty((0, 2, row_length, track_length)), torch.empty((0, 1))

    for file in os.listdir(dir_name):
        f = os.path.join(dir_name, file)
        if os.path.isfile(f) and f.endswith('.csv'):
            print(f' - file {f}', end=' ')
            x_cur, y_cur = generate_siamese_data_from_cp_data_file(f, track_length)
            print(f'({x_cur.size(0)} rows)')

            x = torch.cat((x, x_cur), 0)
            y = torch.cat((y, y_cur), 0)

    return x, y


def main():
    cp_data_dir = 'CP_data'
    siamese_data_dir = 'data'
    data_usage_aims = ('train', 'test')

    timestamp = False

    track_length = config['track_length']

    for data_usage_aim in data_usage_aims:
        print(f'{cp_data_dir}/{data_usage_aim}')
        x, y = generate_siamese_data_from_cp_data_dir(f'{cp_data_dir}/{data_usage_aim}', track_length)

        if x.size(0) != 0:
            print(f'Saving to {siamese_data_dir}/{data_usage_aim} ({x.size(0)} rows).')

            if timestamp:
                ts = datetime.datetime.now().strftime('%d%m%H%M%S')
                torch.save(x, f'{siamese_data_dir}/{data_usage_aim}/x_{ts}.pt')
                torch.save(y, f'{siamese_data_dir}/{data_usage_aim}/y_{ts}.pt')
            else:
                torch.save(x, f'{siamese_data_dir}/{data_usage_aim}/x.pt')
                torch.save(y, f'{siamese_data_dir}/{data_usage_aim}/y.pt')
        else:
            print(f'Nothing to save.')

        print()


if __name__ == '__main__':
    main()
