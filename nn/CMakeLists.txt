CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
PROJECT(nnbathy)



configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${PROJECT_SOURCE_DIR}/config.h)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

FIND_PACKAGE(CUDA REQUIRED)
#FIND_PACKAGE(MPI REQUIRED)

#INCLUDE(FindCUDA)

if(APPLE)
    INCLUDE_DIRECTORIES(/usr/local/cuda/include )#${MPI_INCLUDE_PATH})
    include_directories(/Developer/NVIDIA/CUDA-7.0/samples/common/inc)
elseif(UNIX)
#    set(CUDA_COMPILER_BINDIR "/usr/bin" CACHE PATH "Path for g++ binary for nvcc")
    include_directories("/usr/local/cuda/include/")
    include_directories("/usr/local/cuda/samples/common/inc/")
else()
    message("Windows is not currently supported for this software")
    return()
endif()



FILE(GLOB SOURCES "*.cu" "*.cpp" "*.c" "*.h" "*.cuh")
CUDA_ADD_EXECUTABLE(nnbathy ${SOURCES})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}) #;-O3 -gencode arch=compute_22,code=sm_22)

#LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")
LIST(APPEND CMAKE_CXX_FLAGS "-std=c99 -O3 -ffast-math -Wall")

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

if(APPLE)
    TARGET_LINK_LIBRARIES(nnbathy /usr/local/cuda/lib/libcudart.dylib) # ${MPI_LIBRARIES})
elseif(UNIX)
    TARGET_LINK_LIBRARIES(nnbathy /usr/local/cuda/lib/libcudart_static.a)
endif()

