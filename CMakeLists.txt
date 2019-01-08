project(mandel)

set( CMAKE_CXX_COMPILER /usr/bin/clang++ )

set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     ON)

find_package(Boost 1.68.0 COMPONENTS system program_options)

set(CMAKE_CXX_STANDARD 17) # C++17

set(CMAKE_CXX_FLAGS "-O2 -Wall -Xpreprocessor -fopenmp -lomp")

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


find_package(OpenCV 3.4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable (bin/mandel src/m0.cpp)

target_link_libraries (bin/mandel ${OpenCV_LIBS} ${Boost_LIBRARIES})