mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/Users/arsen-osipyan/Desktop/Diploma/air_objects_identification/tracks_identifier/libtorch ..
cmake --build . --config Release
./tracks_identifier ../TrackToVector_traced.pt
cd ../