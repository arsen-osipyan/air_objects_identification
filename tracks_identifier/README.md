# Образец использования нейросетевой модели в C++

- <code>source</code> - директория с программными файлами (содержит файл <code>main.cpp</code>, пояснения к коду в комментариях)
- <code>libtorch.zip</code> - архивированная библиотека Torch для MacOS (актуальные версии можно скачать с <a href='https://pytorch.org/get-started/locally/'>сайта PyTorch</a>, выбрав нужную ОС и Language: C++ / Java) 
- <code>CMakeLists.txt</code> - CMake-файл для сборки и подключения библиотеки
- <code>save_torchscript.py</code> - сохранение параметров модели в нужном формате для использования в C++ (подробнее в комментариях к коду) 
- <code>TrackToVector.pt</code>, <code>TrackToVector_traced.pt</code> - файлы с параметрами нейросетевой модели, исходный и преобразованный соответственно
- <code>start.sh</code> - команды для сборки и запуска утилиты

### Подробнее про <code>start.sh</code>

1. Создание директории <code>build</code> и переход в нее:

```
mkdir build
cd buid
```

2. Сборка проекта утилитой CMake:

```
cmake -DCMAKE_PREFIX_PATH=/Users/arsen-osipyan/Desktop/Diploma/air_objects_identification/tracks_identifier/libtorch ..
cmake --build . --config Release
```

3. Запуск исполняемого файла с передачей относительного пути до параметров модели:

```
./tracks_identifier ../TrackToVector_traced.pt
```

4. Выход в исходную директорию:

```
cd ../
```