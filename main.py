import numpy as np

from airsim.collections import AirObject, AirEnv, RadarSystem, SimulationManager


# Пример работы с программным модулем


# Создание системы и загрузка в диспетчер моделирования
# достаточно задать лишь воздушную обстановку и РЛС, ПУ создается автоматически SimulationManager
sm = SimulationManager(
    air_env=AirEnv(air_objects=[
        AirObject(track=lambda t: np.array([0.222 * t / 1000.0, 0.0, 10000.0])),
    ]),
    radar_systems=[
        RadarSystem(position=np.array([0, 0, 0]),
                    detection_radius=100000,
                    error=25.0,
                    detection_fault_probability=0.01,
                    detection_period=1000,
                    detection_delay=0)
    ],
    identification_method='nn'
)

# Запуск моделирования
# время в мс
sm.run(0, 1800000, 1000)

# Получение данных ПУ (МТИ)
cp_data = sm.get_data()

# Печать таблицы МТИ
print(cp_data)

# Проверка уникальности присвоения air_object_id для каждой трассы
for rs_id in cp_data['rs_id'].unique():
    for id in cp_data[cp_data['rs_id'] == rs_id]['id'].unique():
        print(f'radar_system: {rs_id}, air_object: {id}, ids: ', end='')
        print(cp_data[(cp_data['rs_id'] == rs_id) & (cp_data['id'] == id)]['air_object_id'].unique())
