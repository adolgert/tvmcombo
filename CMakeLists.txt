cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CUDA_COMPILER clang++)

project(tvmcombo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

find_package(CUDAToolkit REQUIRED)

add_executable(legacy_app
    src/legacy_app.cu
)

set_target_properties(legacy_app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(legacy_app PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(legacy_app
    CUDA::cudart
    CUDA::cublas
)