cmake_minimum_required(VERSION 3.22)
project(pvae)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 23)

find_package(LibTorch REQUIRED)

file(GLOB_RECURSE MAIN_SRCS src/*.cpp src/*.c src/*.hpp src/*.h)
add_executable(main ${MAIN_SRCS})
target_link_libraries(main LibTorch)

