import numpy as np
import pandas as pd

from airsim.utils import get_tracks_timeranges_intersection, interpolate_track

from airsim.identification.utils import find_connected_components


def compute_distance_between_tracks_determined(track_1, track_2):
    if track_1.shape[1] != track_2.shape[1]:
        raise RuntimeError(
            f'Tracks must have the same number of columns, got {track_1.shape[1]} and {track_2.shape[1]} instead.')

    # tracks timeranges intersection
    t_min, t_max = get_tracks_timeranges_intersection(track_1, track_2)
    if t_max < t_min:
        return

    # crop tracks to current max time
    track_1 = track_1[track_1['time'] <= t_max]
    track_2 = track_2[track_2['time'] <= t_max]

    t_0 = min(track_1['time'].max(), track_2['time'].max())

    p1 = interpolate_track(track_1, t_0)
    p2 = interpolate_track(track_2, t_0)

    # Delta between two items of data
    delta = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])

    # Building cov matrices
    cov_1 = np.diag([p1['x_err']**2, p1['y_err']**2, p1['z_err']**2])
    cov_2 = np.diag([p2['x_err']**2, p2['y_err']**2, p2['z_err']**2])
    cov = cov_1 + cov_2

    # Calculating proximity measure
    y = delta @ np.linalg.inv(cov) @ delta.T

    return y


def identify_air_objects_determined(data):
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
                            (data['id'] == tracks_ids[i][1])][['time', 'x', 'y', 'z', 'x_err', 'y_err', 'z_err']]
                       .dropna().sort_values(by=['time']).reset_index(drop=True))
            track_2 = (data[(data['rs_id'] == tracks_ids[j][0]) &
                            (data['id'] == tracks_ids[j][1])][['time', 'x', 'y', 'z', 'x_err', 'y_err', 'z_err']]
                       .dropna().sort_values(by=['time']).reset_index(drop=True))

            dist = compute_distance_between_tracks_determined(track_1, track_2)
            if dist is None:
                continue

            tracks_distances.loc[len(tracks_distances)] = {
                'rs_id_1': tracks_ids[i][0],
                'id_1': tracks_ids[i][1],
                'rs_id_2': tracks_ids[j][0],
                'id_2': tracks_ids[j][1],
                'dist': dist,
                'is_identical': dist < 11.35
            }

    print(tracks_distances)

    # decide which tracks are identical
    # rs_ids = data['rs_id'].unique()
    # for i in range(len(rs_ids) - 1):
    #     for j in range(i + 1, len(rs_ids)):
    #         rs_id_1 = rs_ids[i]
    #         rs_id_2 = rs_ids[j]
    #
    #         distances = (tracks_distances[(tracks_distances['rs_id_1'] == rs_id_1) &
    #                                       (tracks_distances['rs_id_2'] == rs_id_2)][['id_1', 'id_2', 'dist']]
    #                      .sort_values(by=['dist']).reset_index(drop=True))
    #         distances = distances[distances['dist'] < 1.0]
    #         identified_pairs, sum_dist = get_identified_pairs(distances)
    #         for pair in identified_pairs:
    #             tracks_distances.loc[(tracks_distances['rs_id_1'] == rs_id_1) &
    #                                  (tracks_distances['id_1'] == pair[0]) &
    #                                  (tracks_distances['rs_id_2'] == rs_id_2) &
    #                                  (tracks_distances['id_2'] == pair[1]), 'is_identical'] = True

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

    return data
