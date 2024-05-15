import torch


def extend_track_with_zeros(track, track_length, zeros_placement='begin'):
    if track.size(0) == 0 or track_length == 0:
        return None
    if track.size(0) >= track_length:
        return track[:track_length, :]

    zeros = torch.zeros(track_length - track.size(0), track.size(1))

    if zeros_placement == 'begin':
        return torch.cat((zeros, track), 0)
    elif zeros_placement == 'end':
        return torch.cat((track, zeros), 0)

    return None


def extend_track_with_linear_interpolation(track, track_length):
    if track.size(0) == 0 or track_length == 0:
        return None
    if track.size(0) == 1:
        return track.repeat(track_length, 1)
    if track.size(0) >= track_length:
        return track[:track_length, :]

    while track.size(0) != track_length:
        diffs = track[:, 0].diff(1, 0)

        point_1_idx = torch.argmax(diffs).item()
        point_2_idx = point_1_idx + 1

        new_point = torch.lerp(track[point_1_idx, :], track[point_2_idx, :], 0.5).unsqueeze(0)

        track = torch.cat((track[:point_1_idx+1, :], new_point, track[point_2_idx:, :]), 0)

    return track
