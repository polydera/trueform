cmake_minimum_required(VERSION 3.16)
project(test_trueform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(trueform CONFIG REQUIRED)

add_executable(test_trueform test_trueform.cpp)
target_link_libraries(test_trueform PRIVATE tf::trueform)
