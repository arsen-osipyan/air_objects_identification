#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>


int main (int argc, char* argv[])
{
    // Проверка передачи пути до файла с параметрами модели
    if (argc != 2)
    {
        std::cerr << "usage: tracks_identifier <path-to-traced-model>" << std::endl;
        return -1;
    }

    // Загрузка нейросетевой модели из переданного файла
    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e)
    {
        std::cerr << "error loading the model" << std::endl;
        return -1;
    }

    // Создание torch::Tensor для использования в качестве входного объекта
    // 1 - кол-во объектов, 4 - кол-во каналов (время time и координаты x, y, z), 32 - кол-во точек участка траектории
    torch::Tensor example = torch::rand({1, 4, 32});

    // Вычисление ответа модели (вектора из 32 элементов)
    auto output = module.forward({example});

    // Вывод ответа в консоль
    std::cout << output << std::endl;

    return 0;
}
