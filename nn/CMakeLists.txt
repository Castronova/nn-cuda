CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
PROJECT(nnbathy)



configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${PROJECT_SOURCE_DIR}/config.h)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)


FIND_PACKAGE(CUDA REQUIRED)
#FIND_PACKAGE(MPI REQUIRED)

INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(/usr/local/cuda/include )#${MPI_INCLUDE_PATH})
include_directories(/Developer/NVIDIA/CUDA-7.0/samples/common/inc)

FILE(GLOB SOURCES "*.cu" "*.cpp" "*.c" "*.h" "*.cuh")
CUDA_ADD_EXECUTABLE(nnbathy ${SOURCES})

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

TARGET_LINK_LIBRARIES(nnbathy /usr/local/cuda/lib/libcudart.dylib) # ${MPI_LIBRARIES})


