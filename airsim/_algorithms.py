import numpy as np
import pandas as pd


def identify_air_objects(df: pd.DataFrame, c: float = 11.35) -> pd.Series:
    """
    Отождествление и идентификация записей обнаружений ВО
    :param df: pd.DataFrame записями обнаружений
    :param c: постоянная, влияющая на точность отождествления
    :return: pd.Series с индексами исходного датафрейма и идентификаторами ВО
    """
    new_segment_id = 0
    res = df.copy()
    res.reset_index(inplace=True)
    res.loc[:, 'air_object_id'] = None

    for i in range(len(res) - 1):
        row_1 = res.loc[i, :]
        for j in range(i + 1, len(res)):
            row_2 = res.loc[j, :]

            # Delta between two items of data
            delta = np.array([row_1.x - row_2.x, row_1.y - row_2.y, row_1.z - row_2.z])

            # Building cov matrices
            cov_1 = np.diag([row_1.x_err**2, row_1.y_err**2, row_1.z_err**2])
            cov_2 = np.diag([row_2.x_err**2, row_2.y_err**2, row_2.z_err**2])
            cov = cov_1 + cov_2

            # Calculating proximity measure
            y = delta @ np.linalg.inv(cov) @ delta.T
            if y < c:
                i_segment = res.loc[i, 'air_object_id']
                j_segment = res.loc[j, 'air_object_id']
                i_segment_none = i_segment is None
                j_segment_none = j_segment is None
                if i_segment_none and j_segment_none:
                    res.loc[i, 'air_object_id'] = new_segment_id
                    res.loc[j, 'air_object_id'] = new_segment_id
                    new_segment_id += 1
                elif i_segment_none and not j_segment_none:
                    res.loc[i, 'air_object_id'] = res.loc[j, 'air_object_id']
                elif not i_segment_none and j_segment_none:
                    res.loc[j, 'air_object_id'] = res.loc[i, 'air_object_id']
                else:
                    max_segment, min_segment = max(i_segment, j_segment), min(i_segment, j_segment)
                    res.loc[res['air_object_id'] == max_segment, 'air_object_id'] = min_segment

    return res.set_index('index')['air_object_id']


def check_air_object_ids(df: pd.DataFrame):
    air_objects = dict()
    error = 0

    for i in range(len(df)):
        if df.loc[i, 'air_object_id'] in air_objects.keys():
            if df.loc[i, 'air_object'] != air_objects[df.loc[i, 'air_object_id']]:
                error += 1
        else:
            air_objects[df.loc[i, 'air_object_id']] = df.loc[i, 'air_object']

    return error == 0, error / len(df)
