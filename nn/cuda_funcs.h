#ifndef _CUDA_FUNCS_H
#define _CUDA_FUNCS_H

#include "nn.h"
#include "delaunay.h"

//namespace cuda_funcs{

#ifdef __cplusplus
extern "C" {

void cuda_nnpi_normalize_weights(nnpi *nn);
int cuda__get_circle(delaunay* d, point* p, int nt);
void cuda_set_delaunay(delaunay* d);
void check_delaunay();
void cuda_set_circles(circle* c, int count);
void cuda_test_get_global_circles(circle* c, int count);
}
#else

void cuda_nnpi_normalize_weights(nnpi *nn);
int cuda__get_circle(delaunay* d, point* p, int nt);
void cuda_set_delaunay(delaunay* d);
void check_delaunay();
void cuda_set_circles(circle* c, int count);
void cuda_test_get_global_circles(circle* c, int count);
#endif



//// The ifdef checks are necessary to prevent name mangling between C and C++ (CUDA)
//#ifdef __cplusplus
//    extern "C" {
//
//        #include "nn.h"
//        __global__ void cuda_nnpi_normalize_weights(nnpi *nn);
//    }
//#else
//    #include "nn.h"
//    void cuda_nnpi_normalize_weights(nnpi *nn);
//    #endif

#endif