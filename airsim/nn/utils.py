import os
import torch


def get_csv_files_from_dir(dir):
    files = []
    for file in os.listdir(dir):
        f = os.path.join(dir, file)
        if os.path.isfile(f) and f.endswith('.csv'):
            files.append(f)
    return files


def transform_cp_data(data):
    data = data.sort_values(by=['rs_id', 'id', 'time']).reset_index(drop=True)
    data = data[['rs_id', 'id', 'time', 'x', 'y', 'z']]
    data = data.dropna()
    return data


def get_track(data, rs_id, id, t_min=None, t_max=None, drop_ids=True):
    track = data[(data['rs_id'] == rs_id) & (data['id'] == id)]
    if t_min is not None:
        track = track[track['time'] >= t_min]
    if t_max is not None:
        track = track[track['time'] <= t_max]
    if drop_ids:
        track = track.drop(columns=['rs_id', 'id'])
    return track


def get_tracks_timeranges_intersection(track_1, track_2):
    t_min_1, t_max_1 = track_1['time'].min(), track_1['time'].max()
    t_min_2, t_max_2 = track_2['time'].min(), track_2['time'].max()

    t_min, t_max = max(t_min_1, t_min_2), min(t_max_1, t_max_2)
    return t_min, t_max


def extend_track_with_zeros(track, track_length, zeros_placement='start'):
    if track.size(0) == 0 or track_length == 0:
        return None
    if track.size(0) >= track_length:
        return track[:track_length, :]

    zeros = torch.zeros(track_length - track.size(0), track.size(1))

    if zeros_placement == 'start':
        return torch.cat((zeros, track), 0)
    elif zeros_placement == 'end':
        return torch.cat((track, zeros), 0)

    return None


def extend_track_with_linear_interpolation(track, track_length):
    if track.size(0) < 2 or track_length == 0:
        return None
    if track.size(0) >= track_length:
        return track[:track_length, :]

    while track.size(0) != track_length:
        diffs = track[:, 0].diff(1, 0)

        point_1_idx = torch.argmax(diffs).item()
        point_2_idx = point_1_idx + 1

        new_point = torch.lerp(track[point_1_idx, :], track[point_2_idx, :], 0.5).unsqueeze(0)

        track = torch.cat((track[:point_1_idx+1, :], new_point, track[point_2_idx:, :]), 0)

    return track
