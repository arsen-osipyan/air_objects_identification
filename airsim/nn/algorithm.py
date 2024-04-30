import torch

from .models import SiameseNetwork
from .hparams import config
from .utils import (get_tracks_timeranges_intersection,
                    extend_track_with_linear_interpolation,
                    transform_cp_data,
                    get_track)


def compute_tracks_distance(track_1, track_2, model_filename='airsim/nn/model.pt'):
    if track_1.shape[1] != track_2.shape[1]:
        raise RuntimeError(
            f'Tracks must have the same number of columns, got {track_1.shape[1]} and {track_2.shape[1]} instead.')

    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_filename))

    t_min, t_max = get_tracks_timeranges_intersection(track_1, track_2)

    # crop tracks to current max time
    track_1 = track_1[track_1['time'] < t_max]
    track_2 = track_2[track_2['time'] < t_max]

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

    print(f'out_1 = {out_1}')
    print(f'out_2 = {out_2}')

    return dist.item()


def nn_identify_air_objects(data):
    cp_data = transform_cp_data(data)

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

            print(f'Tracks ({rs_id_1}, {id_1}) and ({rs_id_2}, {id_2})')
            dist = compute_tracks_distance(track_1, track_2)
            print(f'dist = {dist}')

