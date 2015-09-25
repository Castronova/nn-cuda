#include "cuda_funcs.h"
#include <cuda_runtime.h>
#include "delaunay.h"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <stdbool.h>
#include "istack.h"
#include "nn_internal.h"
#include <iostream>
#include <stdlib.h>

using namespace std;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define HANDLE_ERROR(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define N_SEARCH_TURNON 20
#define N_FLAGS_TURNON 1000
#define N_FLAGS_INC 100


__global__ void _search_cuda_all(double* ptx, double* pty, int npts, double* cx, double* cy, double* cr, int* fidx, int count){
//     pts:    2d array of search point coordinates
//     npts:   size of the pts array (i.e. number of total points)
//     cx:     circle x coordinates
//     cy:     circle y coordinates
//     cz:     circle z coordinates
//     fidx:   array of circle indices in which the pts intersect
//     count:  number of circles

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < npts){

        // grab the next point
        double px = ptx[tid];
        double py = pty[tid];

        // loop over all circles to find match
        int i;
        for (i = 0; i < count; i++) {

            // note: hypot doesn't work in the kernel
            double dist = (cx[i] - px) * (cx[i] - px) + (cy[i] - py) * (cy[i] - py);
            double radi = cr[i] * cr[i];

            // exit loop early if a match is found
            if (dist <= radi) {
                fidx[tid] = i;
                break;
            }
        }

        // increment thread id
        tid += blockDim.x * gridDim.x;
    }
}



int cuda_delaunay_circles_find_all(double pts[][2], int npts, double* cx, double* cy, double* cr, int n_cir){

    cudaGetDevice(0);
    cudaDeviceReset();

    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the

    // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, _search_cuda_all, 0, npts);
    // Round up according to array size
    gridSize = (npts + blockSize - 1) / blockSize;

    int* d_idx;
    double *d_cx, *d_cy, *d_cr, *d_ptx, *d_pty;
//    int idx [n_cir];

    double* ptx = (double*) malloc(npts * sizeof(double));
    double* pty = (double*) malloc(npts * sizeof(double));
    for (int i =0; i< npts; i++){
        ptx[i] = pts[i][0];
        pty[i] = pts[i][1];
    }

    double size = (3 * n_cir * sizeof(double)) +
                  (2 * npts * sizeof(double)) +
                  (npts * sizeof(int));

    cout << "Space allocated on Device: "<< size / 1000000<< " Mb" << endl;

    // move data onto device
    HANDLE_ERROR(cudaMalloc((void **) &d_cx, n_cir * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_cx, cx, n_cir * sizeof(double), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void **) &d_cy, n_cir * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_cy, cy, n_cir * sizeof(double), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void **) &d_cr, n_cir * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_cr, cr, n_cir * sizeof(double), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void **) &d_ptx, npts * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_ptx, ptx, npts * sizeof(double), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void **) &d_pty, npts * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_pty, pty, npts * sizeof(double), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void **) &d_idx, npts * sizeof(int)));

    int *res = (int*)malloc(npts * sizeof(int));
    _search_cuda_all <<< gridSize, blockSize >>> (d_ptx, d_pty, npts, d_cx, d_cy, d_cr, d_idx, n_cir);
    HANDLE_ERROR(cudaMemcpy(res, d_idx, npts * sizeof(int), cudaMemcpyDeviceToHost));

//    cout << "-----------------\n";
//    for (int i = npts-10; i < npts; i++){
//        double valid = (cx[res[i]] - ptx[i] ) * (cx[res[i]] - ptx[i] ) + (cy[res[i]] - pty[i]) * (cy[res[i]] - pty[i]);
//        cout << "Index " << i << ", POINT(" << ptx[i] << ", " << pty[i] << ") IDX: " << res[i] << "\t\t\t" << valid << endl;
//    }

    // free memory
    cudaFree(d_idx);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_cr);
    cudaFree(d_ptx);
    cudaFree(d_pty);

    return 1;
}

















/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return;
    cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
    exit (1);
}