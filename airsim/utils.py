import os
from scipy.interpolate import interp1d


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


#         f = os.path.join(dir_name, file)
def interpolate_track(track, time, columns=None):
    if 'time' not in track.columns:
        raise RuntimeError('track dataframe must have \'time\' column')

    if columns is None:
        columns = list(track.columns)
        columns.remove('time')

    track_interp = {'time': time}
    track_sorted = track.sort_values(by='time')
    times = track_sorted['time']

    for column in columns:
        col = track_sorted[column]
        col_interp_func = interp1d(times, col, kind='linear')
        track_interp[column] = float(col_interp_func(time))

    return track_interp
