import torch

from airsim.nn.models import TrackToVector


# Окончательно разработанные нейросетевые модели преобразуются в TorchScript объект для
# дальнейшего использования в продакте
# Для преобразования используется подмодуль jit библиотеки pytorch

# Инициализация модели TrackToVector и загрузка параметров
model = TrackToVector()
model.load_state_dict(torch.load('TrackToVector.pt'))

# Пример входных данных
# 1 - кол-во объектов, 4 - кол-во каналов (время time и координаты x, y, z), 32 - кол-во точек участка траектории
example = torch.rand(1, 4, 32)

# Преобразование в TorchScript объект
traced_model = torch.jit.trace(model, example)
traced_model.save('tracks_identifier/TrackToVector_traced.pt')
