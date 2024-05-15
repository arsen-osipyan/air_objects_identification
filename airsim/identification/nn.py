import torch
import pandas as pd

from airsim.nn.models import SiameseNetwork
from airsim.nn.hparams import config
from airsim.utils import get_tracks_timeranges_intersection
from airsim.nn.utils import extend_track_with_linear_interpolation
from airsim.identification.utils import get_identified_pairs, find_connected_components


def compute_distance_between_tracks(track_1, track_2, model_filename='airsim/nn/model.pt'):
    if track_1.shape[1] != track_2.shape[1]:
        raise RuntimeError(
            f'Tracks must have the same number of columns, got {track_1.shape[1]} and {track_2.shape[1]} instead.')

    # model initialization
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_filename))

    # tracks timeranges intersection
    t_min, t_max = get_tracks_timeranges_intersection(track_1, track_2)
    if t_max < t_min:
        return

    # crop tracks to current max time
    track_1 = track_1[track_1['time'] <= t_max]
    track_2 = track_2[track_2['time'] <= t_max]

    # get last parts of tracks
    track_1 = track_1.tail(config['track_length'])
    track_2 = track_2.tail(config['track_length'])

    # transform to tensors
    track_1_tensor = torch.tensor(track_1.values, dtype=torch.float32)
    track_2_tensor = torch.tensor(track_2.values, dtype=torch.float32)

    if track_1_tensor.size(0) < config['track_length']:
        track_1_tensor = extend_track_with_linear_interpolation(track_1_tensor, config['track_length'])

    if track_2_tensor.size(0) < config['track_length']:
        track_2_tensor = extend_track_with_linear_interpolation(track_2_tensor, config['track_length'])

    track_1_tensor = track_1_tensor.transpose(0, 1).unsqueeze(0)
    track_2_tensor = track_2_tensor.transpose(0, 1).unsqueeze(0)

    out_1, out_2 = model(track_1_tensor, track_2_tensor)

    dist = torch.nn.functional.pairwise_distance(out_1, out_2, keepdim=True)

    return dist.item()


def identify_air_objects_nn(data):
    if data['air_object_id'].min() > -1:
        return

    # identity table
    tracks_distances = pd.DataFrame(columns=['rs_id_1', 'id_1', 'rs_id_2', 'id_2', 'dist', 'is_identical'])

    # compute distances between all pairs of tracks and write it in df_identity
    tracks_ids = [(rs_id, id) for rs_id in data['rs_id'].unique() for id in data[data['rs_id'] == rs_id]['id'].unique()]
    for i in range(len(tracks_ids)):
        for j in range(i + 1, len(tracks_ids)):
            if tracks_ids[i][0] == tracks_ids[j][0]:
                continue

            track_1 = (data[(data['rs_id'] == tracks_ids[i][0]) &
                            (data['id'] == tracks_ids[i][1])][['time', 'x', 'y', 'z']]
                       .dropna().sort_values(by=['time']).reset_index(drop=True))
            track_2 = (data[(data['rs_id'] == tracks_ids[j][0]) &
                            (data['id'] == tracks_ids[j][1])][['time', 'x', 'y', 'z']]
                       .dropna().sort_values(by=['time']).reset_index(drop=True))

            dist = compute_distance_between_tracks(track_1, track_2)
            if dist is None:
                continue

            tracks_distances.loc[len(tracks_distances)] = {
                'rs_id_1': tracks_ids[i][0],
                'id_1': tracks_ids[i][1],
                'rs_id_2': tracks_ids[j][0],
                'id_2': tracks_ids[j][1],
                'dist': dist,
                'is_identical': False
            }

    # decide which tracks are identical
    rs_ids = data['rs_id'].unique()
    for i in range(len(rs_ids) - 1):
        for j in range(i + 1, len(rs_ids)):
            rs_id_1 = rs_ids[i]
            rs_id_2 = rs_ids[j]

            distances = (tracks_distances[(tracks_distances['rs_id_1'] == rs_id_1) &
                                          (tracks_distances['rs_id_2'] == rs_id_2)][['id_1', 'id_2', 'dist']]
                         .sort_values(by=['dist']).reset_index(drop=True))

            distances = distances[distances['dist'] < 1.0]
            identified_pairs, sum_dist = get_identified_pairs(distances)

            for pair in identified_pairs:
                tracks_distances.loc[(tracks_distances['rs_id_1'] == rs_id_1) &
                                     (tracks_distances['id_1'] == pair[0]) &
                                     (tracks_distances['rs_id_2'] == rs_id_2) &
                                     (tracks_distances['id_2'] == pair[1]), 'is_identical'] = True

    tracks_distances = tracks_distances[tracks_distances['is_identical']]

    graph = {t_id: [] for t_id in tracks_ids}

    for index, row in tracks_distances.iterrows():
        rs_id_1, id_1 = row['rs_id_1'], row['id_1']
        rs_id_2, id_2 = row['rs_id_2'], row['id_2']
        graph[(rs_id_1, id_1)].append((rs_id_2, id_2))
        graph[(rs_id_2, id_2)].append((rs_id_1, id_1))

    identical_tracks_components = find_connected_components(graph)

    for comp in identical_tracks_components:
        ao_ids = {tr: data[(data['rs_id'] == tr[0]) & (data['id'] == tr[1])]['air_object_id'].unique()[0]
                  for tr in comp}
        ao_ids_unique = set(ao_ids.values())
        if len(ao_ids_unique) == 1:
            if -1 in ao_ids_unique:
                new_ao_id = data['air_object_id'].max() + 1
                for track in ao_ids.keys():
                    data.loc[(data['rs_id'] == track[0]) & (data['id'] == track[1]), 'air_object_id'] = new_ao_id
        else:
            if -1 in ao_ids_unique:
                ao_ids_unique.remove(-1)
            new_ao_id = min(ao_ids_unique)
            for track in ao_ids.keys():
                data.loc[(data['rs_id'] == track[0]) & (data['id'] == track[1]), 'air_object_id'] = new_ao_id