#ifndef _CUDA_FUNCS_H
#define _CUDA_FUNCS_H

#include "nn.h"

//namespace cuda_funcs{

#ifdef __cplusplus
extern "C" {

void cuda_nnpi_normalize_weights(nnpi *nn);

}
#else

void cuda_nnpi_normalize_weights(nnpi *nn);

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