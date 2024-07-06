import datetime
import numpy as np

from airsim.collections import AirObject, AirEnv, RadarSystem, SimulationManager


def save_cp_data(cp_data, cp_data_dir, data_usage_aim):
    timestamp = datetime.datetime.now().strftime('%d%m%H%M%S')
    filename = f'{cp_data_dir}/{data_usage_aim}/data_{timestamp}.csv'
    cp_data.to_csv(filename)


def generate_linear_tracks(n_tracks, axis='xyz'):
    '''
    Генерация заданного кол-ва функций траекторий по заданным осям
    :param n_tracks: кол-во функций траекторий
    :param axis: оси, по которым происходит движение
    :return: список функций
    '''
    tracks = []

    Rx_0 = np.random.uniform(-1.0e+7, 1.0e+7, size=n_tracks)
    Ry_0 = np.random.uniform(-1.0e+7, 1.0e+7, size=n_tracks)
    Rz_0 = np.random.uniform(5.0e+3, 15.0e+3, size=n_tracks)

    Vx = np.random.uniform(-0.5, 0.5, size=n_tracks) if 'x' in axis else [0.0] * n_tracks
    Vy = np.random.uniform(-0.5, 0.5, size=n_tracks) if 'y' in axis else [0.0] * n_tracks
    Vz = 0.005 * np.random.randn(n_tracks) if 'z' in axis else [0.0] * n_tracks

    for i in range(n_tracks):
        Rx_0_i = Rx_0[i]
        Ry_0_i = Ry_0[i]
        Rz_0_i = Rz_0[i]
        Vx_i = Vx[i]
        Vy_i = Vy[i]
        Vz_i = Vz[i]

        tracks.append(
            lambda t, Rx_0_i=Rx_0_i, Ry_0_i=Ry_0_i, Rz_0_i=Rz_0_i, Vx_i=Vx_i, Vy_i=Vy_i, Vz_i=Vz_i: np.array(
                [Rx_0_i + Vx_i * t, Ry_0_i + Vy_i * t, Rz_0_i + Vz_i * t])
        )

    return tracks


def generate_cp_data(t_max_seconds, axis='xyz'):
    '''
    Генерация данных ПУ (МТИ) системы из 2 РЛС и 2 ВО за указанное время в указанных осях
    :param t_max_seconds: время в секундах конца моделирования
    :param axis: оси, по которых происходит движение ВО
    :return: МТИ
    '''
    t_min = 0
    t_max = t_max_seconds * 1000
    dt = 50

    sm = SimulationManager(
        air_env=AirEnv(air_objects=[
            AirObject(track=ao_track) for ao_track in generate_linear_tracks(2, axis=axis)
        ]),
        radar_systems=[
            RadarSystem(position=np.array([0, 0, 0]),
                        detection_radius=1e+308,
                        error=10.0,
                        detection_fault_probability=0.01,
                        detection_period=250,
                        detection_delay=np.random.randint(0, 250//dt) * dt),
            RadarSystem(position=np.array([0, 0, 0]),
                        detection_radius=1e+308,
                        error=10.0,
                        detection_fault_probability=0.01,
                        detection_period=250,
                        detection_delay=np.random.randint(0, 250//dt) * dt),
        ]
    )

    sm.run(t_min, t_max, dt)

    return sm.get_data()


def generate_train_cp_data(cp_data_dir):
    '''
        Генерация обучающих данных на системах с движением по одной, двум или трем осям
    '''
    data_usage_aim = 'train'

    print(f'x: ', end='')
    for i in range(30):
        cp_data = generate_cp_data(t_max_seconds=1800, axis='x')
        save_cp_data(cp_data, cp_data_dir, data_usage_aim)
        print('*', end='')
    print()

    print(f'y: ', end='')
    for i in range(30):
        cp_data = generate_cp_data(t_max_seconds=1800, axis='y')
        save_cp_data(cp_data, cp_data_dir, data_usage_aim)
        print('*', end='')
    print()

    print(f'xy: ', end='')
    for i in range(30):
        cp_data = generate_cp_data(t_max_seconds=1800, axis='xy')
        save_cp_data(cp_data, cp_data_dir, data_usage_aim)
        print('*', end='')
    print()

    print(f'xyz: ', end='')
    for i in range(120):
        cp_data = generate_cp_data(t_max_seconds=1800, axis='xyz')
        save_cp_data(cp_data, cp_data_dir, data_usage_aim)
        print('*', end='')
    print()


def generate_test_cp_data(cp_data_dir):
    '''
    Генерация тестовых данных на системах с движением по одной, двум или трем осям
    '''
    data_usage_aim = 'valid'

    print(f'x: ', end='')
    for i in range(10):
        cp_data = generate_cp_data(t_max_seconds=100, axis='x')
        save_cp_data(cp_data, cp_data_dir, data_usage_aim)
        print('*', end='')
    print()

    print(f'y: ', end='')
    for i in range(10):
        cp_data = generate_cp_data(t_max_seconds=100, axis='y')
        save_cp_data(cp_data, cp_data_dir, data_usage_aim)
        print('*', end='')
    print()

    print(f'xy: ', end='')
    for i in range(10):
        cp_data = generate_cp_data(t_max_seconds=100, axis='xy')
        save_cp_data(cp_data, cp_data_dir, data_usage_aim)
        print('*', end='')
    print()

    print(f'xyz: ', end='')
    for i in range(40):
        cp_data = generate_cp_data(t_max_seconds=100, axis='xyz')
        save_cp_data(cp_data, cp_data_dir, data_usage_aim)
        print('*', end='')
    print()


if __name__ == '__main__':
    CP_DATA_DIR = 'CP_data'
    generate_train_cp_data(CP_DATA_DIR)
    generate_test_cp_data(CP_DATA_DIR)
