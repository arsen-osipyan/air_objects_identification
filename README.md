# Программный модуль отождествления траекторий воздушных объектов

## Модуль <code>airsim</code>

- <code>time.py</code> - класс-синглтон (может иметь единственный экземпляр) модельного времени
- <code>model.py</code> - абстрактный класс модели системы с астрактным методом <code>trigger()</code>
- <code>airobject.py</code> - класс воздушного объекта
- <code>airenv.py</code> - класс воздушной обстановки
- <code>radarsystem.py</code> - класс РЛС
- <code>controlpoint.py</code> - класс пункта управления
- <code>supervisor.py</code> - класс диспетчера для упрощенного моделирования систем
- <code>collections.py</code> - файл для импорта всех классов в модуле

## Подмодуль <code>airsim.nn</code>

- <code>models.py</code> - модели сиамской нейронной сети <code>SiameseNetwork</code>, контрастной функции потерь <code>ContrastiveLoss</code> и датасета для СНС <code>SiameseDataset</code>
- <code>hparams.py</code> - гиперпараметры модели, функции потерь и алгоритма обучения
- <code>prepare_cp_data.py</code> - подготовка файлов с пункта управления, моделирование различных воздушых систем, сохранение в папку <code>CP_data</code>
- <code>prepare_siamese_data.py</code> - работа с файлами из папки <code>CP_data</code>, преобразование в нужный формат для подачи на вход СНС и сохранение в папку <code>data</code>
- <code>train.py</code> - обучение модели
- <code>compute_metrics.py</code> - расчет метрик на тестовой выборке
- <code>algorithm.py</code> - нейросетевой алгоритм
- <code>utils.py</code> - утилиты по преобразованию данных
- <code>model.pt</code> - параметры обученной модели, сохраняются после обучения для дальнейшего использования
- <code>plots.png</code> - графики функции потерь и параметра <code>learning_rate</code>, сохраняются после завершения процесса обучения
- <code>CP_data/</code> - папка с исходными данными ПУ после моделирования систем 
- <code>data/</code> - папка с данными для подачи на вход СНС
