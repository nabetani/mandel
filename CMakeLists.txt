set( CMAKE_CXX_COMPILER /usr/bin/clang++ )

set(CMAKE_CXX_STANDARD 17) # C++17

set(CMAKE_CXX_FLAGS "-O2 -Wall -Xpreprocessor -fopenmp /usr/local/opt/libomp/lib/libomp.dylib")

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(opencv_project)

find_package(OpenCV 3.4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable (bin/mandel src/m0.cpp)

target_link_libraries (bin/mandel ${OpenCV_LIBS})