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



#cmake_minimum_required(VERSION 3.2)
#project(nnbathy)
#
### Pass options to NVCC
#find_package(CUDA QUIET REQUIRED)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}) #;-O3 -gencode arch=compute_22,code=sm_22)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
##set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#
#set(SOURCE_FILES
#    nnbathy.c
#    nnbathy.cu
#    triangle.c
#    delaunay.c
#    istack.c
#    minell.c
#    preader.c
#    nncommon.c
#    lpi.c
#    nnpi.c
#    hash.c
#    nncommon_vulnerable.c
#    nnai.c)
#
#include_directories(/Developer/NVIDIA/CUDA-7.0/samples/common/inc)
#
#cuda_add_executable(nnbathy ${SOURCE_FILES})

#add_executable(nnbathy
#                nnbathy.cu
#                triangle.c
#                delaunay.c
#                istack.c
#                nnbathy.c
#                minell.c
#                preader.c
#                nncommon.c
#                lpi.c
#                nnpi.c
#                hash.c
#                nncommon_vulnerable.c
#                nnai.c
#                )
