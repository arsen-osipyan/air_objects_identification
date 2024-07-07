import torch
import pandas as pd

from airsim.nn.models import TrackToVector
from airsim.nn.hparams import config
from airsim.utils import get_tracks_timeranges_intersection
from airsim.nn.utils import extend_track_with_linear_interpolation
from airsim.identification.utils import get_identified_pairs, find_connected_components


def compute_distance_between_tracks(track_1, track_2, model_filename='airsim/nn/model.pt'):
    if track_1.shape[1] != track_2.shape[1]:
        raise RuntimeError(
            f'Tracks must have the same number of columns, got {track_1.shape[1]} and {track_2.shape[1]} instead.')

    # model initialization
    model = TrackToVector()
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

    out_1 = model(track_1_tensor)
    out_2 = model(track_2_tensor)

    dist = torch.nn.functional.pairwise_distance(out_1, out_2, keepdim=True)

    return dist.item()


def identify_air_objects_nn(data):
    # Таблица результатов сравнений
    tracks_distances = pd.DataFrame(columns=['rs_id_1', 'id_1','rs_id_2', 'id_2', 'dist', 'is_identical'])

    # Расчет метрики расстояния между парами траекторий и запись в таблицу
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

            dist = compute_distance_between_tracks(track_1, track_2, model_filename='airsim/nn/TrackToVector.pt')
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

    # Предварительное решение об отождествлении
    # Пороговое значение определено по результатам работы модели на специальной выборке
    # Подаются всевозможные пары объектов и рассчитывается ЕР между векторами МП
    # Значения ЕР распределены по 2 пикам (первый - значения на тождественных парах, второй - на нетождественных)
    # Пики моделируются нормальным распределением и примерно можно определить значение между ними,
    # которое позволит минимизировать ошибку
    tracks_distances.loc[tracks_distances['dist'] < 0.1, 'is_identical'] = True

    print(tracks_distances)

    # Исключение всех нетождественных пар
    # Таким образом track_distances - таблица с информацией о ребрах неориентированного графа
    tracks_distances = tracks_distances[tracks_distances['is_identical']]

    graph = {t_id: [] for t_id in tracks_ids}

    # Построение графа
    for index, row in tracks_distances.iterrows():
        rs_id_1, id_1 = row['rs_id_1'], row['id_1']
        rs_id_2, id_2 = row['rs_id_2'], row['id_2']
        graph[(rs_id_1, id_1)].append((rs_id_2, id_2))
        graph[(rs_id_2, id_2)].append((rs_id_1, id_1))

    # Получение компонент связности (каждая компонента - отдельный ВО)
    identical_tracks_components = find_connected_components(graph)

    # Здесь может быть добавлена проверка на полноту всех компонент связности
    # Пример:
    #   В МТИ есть 3 трассы, 1-ая тождественна 2-ой и 3-ей, но 2-ая и 3-ья между собой не тождественны
    #   В графе это будет неполная компонента связности, эти ситуации можно проверять и предупреждать пользователя
    #   В данной реализации проверки нет

    # Разметка air_object_id в соответствие с компонентами
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
