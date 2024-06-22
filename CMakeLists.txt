cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Triathlon LANGUAGES CXX CUDA)

# Cuda versiyonu ayarlama
find_package(CUDA 12.0 REQUIRED)

# header dosyalarını include etme
include_directories(include)

# Source files
set(SOURCES
    src/main.cu
    src/athlete.cu
    src/team.cu
    src/triathlon.cu
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --rdc=true --arch=sm_60 --allow-unsupported-compiler")
