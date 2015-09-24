#include "cuda_funcs.h"
#include <cuda_runtime.h>
#include "delaunay.h"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <stdbool.h>


struct S {
    int x;
    int y;
    int z;
    int* k;
};
typedef struct S S;


__device__  __constant__ S* d_struct;

// tony: this needs to be set, see nnpi.c line 708
__device__ __constant__ delaunay* dev_d;
__device__ __constant__ int** d_circles;
__device__ __constant__ double* d_array;
__device__ __constant__ int s_array[256];
__device__ __constant__ int d_int;
//__device__ __constant__ double d_circles_const[5000][3];

__device__ __constant__ double* cx;
__device__ __constant__ double* cy;
__device__ __constant__ double* cr;
__device__ __constant__ int d_circle_count;
__device__ __constant__ int d_circle_size;


bool ERROR_CHECK(cudaError_t Status) {
    if(Status != cudaSuccess)
    {
        printf(cudaGetErrorString(Status));
        return false;
    }
    return true;
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ void _cuda_nnpi_normalize_weights(nnpi* nn) {

//    int j = blockIdx.x * blockDim.x + threadIdx.x;
//    nnpi* newnn = nn[j];

    int n = nn->nvertices;
    double sum = 0.0;
    int i;

    for (i = 0; i < n; ++i)
        sum += nn->weights[i];

    for (i = 0; i < n; ++i)
        nn->weights[i] /= sum;
}


__global__ void _cuda_find_containing_circle(double px, double py, int* fidx, int count){

    // raw c code
//    for (tid = 0; tid < nt; ++tid) {
//        if (circle_contains(&d->circles[tid], p))
//            break;
//    }



//    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idx = threadIdx.x;
//    printf("%d ", idx);

    if(idx < count){
        if (hypot(cx[idx] - px, cy[idx] - py) <= cr[idx]){
//            printf("FOUND: %d\n", idx);
            *fidx = idx;
        }
    }
//    __syncthreads();
//
//    if (*fidx == -999){
//        *fidx = 10;
//    }
}

__host__ void cuda_nnpi_normalize_weights(nnpi* nn) {

    // tony: adding some cuda stuff

    nnpi *dev_nn;
    cudaMalloc( (void **) &dev_nn, sizeof(nnpi));
    cudaMemcpy(dev_nn, nn, sizeof(nnpi), cudaMemcpyHostToDevice);


    // tony: call cuda_nn_interpolate. todo, split into parallel blocks
    _cuda_nnpi_normalize_weights<<<1,1>>>(dev_nn);

    cudaMemcpy(nn, dev_nn, sizeof(nnpi), cudaMemcpyDeviceToHost);

    // tony: free the allocated memory
    cudaFree(dev_nn);

}

__host__ int cuda__get_circle(delaunay* d, point* p, int nt){

    cudaError_t Status = cudaSuccess;


    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
    // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                                        _cuda_find_containing_circle, 0, nt);


    // Round up according to array size
    gridSize = (nt + blockSize - 1) / blockSize;

//    MyKernel<<< gridSize, blockSize >>>(array, arrayCount);


//    point* dev_p;
    int* dev_idx;
    int idx = -999;  // initial condition.  If -999 is returned then I know that a match was not found


//    double *circlex;
//    circlex = (double *) malloc(sizeof(double) * nt);
//
//    for (int i = 0; i < nt; i++) {
//        circlex[i] = p->x;
//    }

    //398
    double *hcx, *hcy, *hcr;
    Status = cudaMemcpyFromSymbol(&hcx, cx, sizeof(double), 0, cudaMemcpyDeviceToHost);
    Status = cudaMemcpyFromSymbol(&hcy, cy, sizeof(double), 0, cudaMemcpyDeviceToHost);
    Status = cudaMemcpyFromSymbol(&hcr, cr, sizeof(double), 0, cudaMemcpyDeviceToHost);

    double val = hypot(hcx[398] - p->x, hcy[398] - p->y);
    printf("val %3.1f\n", val);

    // hypot(c->x - p->x, c->y - p->y) <= c->r)

    // tony: I do not know what p->x and p->y are switched
    // (lldb) p p->x       (double) $5 = 593424.65000000002
    // (lldb) p p->y       (double) $6 = 5676316.9800000004
    // (lldb) p hcx[398]   (double) $10 = 5677316.4925473807
    // (lldb) p hcy[398]   (double) $11 = 593633.92188352742
    // (lldb) p hcr[398]   (double) $12 = 1733.2493017905658

    // (lldb) p p->x       (double) $400 = 593424.65000000002
    // (lldb) p p->y       (double) $402 = 5676316.9800000004
    // (lldb) p c->x       (double) $399 = 593633.92188352742
    // (lldb) p c->y       (double) $401 = 5677316.4925473807
    // (lldb) p c->r       (double) $403 = 1733.2493017905658

//    double** r_array2;
//    Status = cudaMemcpyFromSymbol(&r_array2, d_circles, size, 0, cudaMemcpyDeviceToHost);
//    ERROR_CHECK(Status);


//    cudaMalloc((void **) &dev_p, sizeof(point));
//    cudaMemcpy(p, dev_p, sizeof(point), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &dev_idx, sizeof(int));
//    cudaMemcpy(&idx, dev_idx, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_idx, &idx, sizeof(int), cudaMemcpyHostToDevice);

//    int blocksize = 32; // value usually chosen by tuning and hardware constraints
//    int nblocks = nt / blocksize + 1; // value determine by block size and total work
//    _cuda_find_containing_circle <<<nblocks, blocksize>>> (p->x, p->y, dev_idx, nt);
//    _cuda_find_containing_circle <<<nt, 1>>> (p->x, p->y, dev_idx, nt);
//    _cuda_find_containing_circle <<<gridSize, blockSize>>> (p->x, p->y, dev_idx, nt);
    _cuda_find_containing_circle <<<1, nt>>> (p->x, p->y, dev_idx, nt);

    int res;

    Status = cudaMemcpy(&res, dev_idx, sizeof(int), cudaMemcpyDeviceToHost);

//    cudaFree(dev_p);
    cudaFree(dev_idx);

    return idx;
}

__host__ void check_delaunay(){


    // testing: make sure delaunay is set properly on the device
    delaunay* test_d;
    cudaMemcpy(test_d, dev_d, sizeof(delaunay), cudaMemcpyDeviceToHost);
    printf("testing");
}

__host__ void cuda_set_delaunay(delaunay* d){

    cudaError_t Status = cudaSuccess;

//    cudaMalloc((void **)&dev_d, sizeof(delaunay));
//    S h_struct = {1,2,3};
//    S r_struct;

    Status = cudaMemcpyToSymbol(dev_d, d, sizeof(struct delaunay));
//    Status = cudaMemcpyToSymbol(d_struct, &h_struct, sizeof(S));
//    Status = cudaMemcpyToSymbol(dev_d, d, sizeof(delaunay));
    ERROR_CHECK(Status);

    // testing: make sure delaunay is set properly on the device
    delaunay test_d;
    Status = cudaMemcpyFromSymbol(&test_d, dev_d, sizeof(struct delaunay), 0, cudaMemcpyDeviceToHost);
    ERROR_CHECK(Status);
    printf("testing");


    // STRUCT POINTER
    int k[5]  = {1, 2, 3, 4, 5};
    S h_struct2 ={11, 22, 33, k};
    S* h_structp = &h_struct2;
    S r_struct2;
    Status = cudaMemcpyToSymbol(d_struct, h_structp, sizeof(h_structp)*2);
    ERROR_CHECK(Status);

    Status = cudaMemcpyFromSymbol(&r_struct2, d_struct, sizeof(d_struct)*2, 0, cudaMemcpyDeviceToHost);
    ERROR_CHECK(Status);
    printf("%d, %d, %d\n",r_struct2.x, r_struct2.y, r_struct2.z);
}

__host__ void cuda_set_circles(circle* c, int count) {

    cudaError_t Status = cudaSuccess;

    // tony: I am splitting the circle array into x, y, and r arrays.  This is b/c cuda was flattening my 2d array
    // tony: causing the find circles function to fail.  This may not be elegant, but it ensures that the data exists
    // tony: correctly on the device.  Performance can be improved by flattening this into a 1d array.
    double* circlex;
    double* circley;
    double* circler;

    circlex =(double*)malloc(sizeof(double)*count);
    circley =(double*)malloc(sizeof(double)*count);
    circler =(double*)malloc(sizeof(double)*count);

    for (int i=0; i<count; i++){
        circlex[i] = c[i].x;
        circley[i] = c[i].y;
        circler[i] = c[i].r;
    }

    // todo: cx, cy, cr must be freed when the app completes.
    // move the circle arrays onto the device.  These will be accessed in the find circles function
    Status = cudaMemcpyToSymbol(cx, &circlex, sizeof(circlex), 0, cudaMemcpyHostToDevice);
    ERROR_CHECK(Status);

    Status = cudaMemcpyToSymbol(cy, &circley, sizeof(circley), 0, cudaMemcpyHostToDevice);
    ERROR_CHECK(Status);

    Status = cudaMemcpyToSymbol(cr, &circler, sizeof(circler), 0, cudaMemcpyHostToDevice);
    ERROR_CHECK(Status);

    Status = cudaMemcpyToSymbol(d_circle_count, &count, sizeof(int), 0, cudaMemcpyHostToDevice);
    ERROR_CHECK(Status);

    int h_circle_size = sizeof(circlex);
    Status = cudaMemcpyToSymbol(d_circle_size, &h_circle_size, sizeof(int), 0, cudaMemcpyHostToDevice);
    ERROR_CHECK(Status);

    int h_circle_count;
    Status = cudaMemcpyFromSymbol(&h_circle_count, d_circle_count, sizeof(int), 0, cudaMemcpyDeviceToHost);
    ERROR_CHECK(Status);



    // free memory
    free(circlex);
    free(circley);
    free(circler);


    //    Checking to see if the arrays were loaded onto the device correctly.
    double *rcx;
    Status = cudaMemcpyFromSymbol(&rcx, cx, sizeof(circlex), 0, cudaMemcpyDeviceToHost);
    ERROR_CHECK(Status);

    double *rcy;
    Status = cudaMemcpyFromSymbol(&rcy, cy, sizeof(circley), 0, cudaMemcpyDeviceToHost);
    ERROR_CHECK(Status);

    double *rcr;
    Status = cudaMemcpyFromSymbol(&rcr, cr, sizeof(circler), 0, cudaMemcpyDeviceToHost);
    ERROR_CHECK(Status);

    printf("%3.1f %3.1f %3.1f\n", rcx[398], rcy[398], rcr[398]);

//    free(&rcx);
//    free(rcy);
//    free(rcr);


    // idx = 398
    // (x = 593633.92188352742, y = 5677316.4925473807, r = 1733.2493017905658)



    // read the circle structure into a simple array
//    double circle_array[count][3];
//    for (int i=0; i<count; i++){
//        circle_array[i][0] = c[i].x;
//        circle_array[i][1] = c[i].y;
//        circle_array[i][2] = c[i].r;
//    }





//    Status = cudaMemcpyToSymbol(d_circles_const, &circle_array, sizeof(circle_array), 0, cudaMemcpyHostToDevice);
//    ERROR_CHECK(Status);

//    int** r_array2;
//    Status = cudaMemcpyFromSymbol(&r_array2, d_circles, sizeof(circle_array), 0, cudaMemcpyDeviceToHost);
//    ERROR_CHECK(Status);


////    double** circle_array = (double**) malloc(count * sizeof(double*));
//    double** circle_array;
//    circle_array = (double**) malloc(count * sizeof(double*));
//    for (int i=0; i<count; i++){
//        circle_array[i] = (double*) malloc(3 * sizeof(double));
//        circle_array[i][0] = i; //c[i].x;
//        circle_array[i][1] = i; //c[i].y;
//        circle_array[i][2] = i; //c[i].r;
//    }

//    cudaMalloc((void**)&d_circles, count * sizeof(double*));
//    for (int i=0; i < count; i++){
//        cudaMalloc((void**)&d_circles[i], 3* sizeof(double));
//    }

//    Status = cudaMemcpyToSymbol(d_circles, &circle_array, sizeof(circle_array), 0, cudaMemcpyHostToDevice);
//    ERROR_CHECK(Status);




//    int** r_array2;
//    Status = cudaMemcpyFromSymbol(&r_array2, d_circles, sizeof(circle_array), 0, cudaMemcpyDeviceToHost);
//    ERROR_CHECK(Status);

//    // free the circle array
//    for(int i=0; i < count; i++){
//        free(circle_array[i]);
//    }
//    free(circle_array);

//    // Dynamic array size
//    int* h_array2;
//    h_array2 = (int*) malloc(10);
//    for (int i=0; i<10; i++){
//        h_array2[i] = i;
//    }
//    Status = cudaMemcpyToSymbol(d_array, &h_array2, sizeof(h_array2), 0, cudaMemcpyHostToDevice);
//    ERROR_CHECK(Status);
//
//    int* r_array2;
//    Status = cudaMemcpyFromSymbol(&r_array2, d_array, sizeof(h_array2), 0, cudaMemcpyDeviceToHost);
//    ERROR_CHECK(Status);




//    free(r_array);
//    free(h_array);

    // allocate space for the d_circles array
//    Status = cudaMalloc(&d_circles, sizeof(circle)*count);
//    ERROR_CHECK(Status);

//    Status = cudaMemcpyToSymbol(d_circles, c, sizeof(circle)*count);
//    ERROR_CHECK(Status);

//    Status = cudaMemcpyToSymbol(d_circle_count, &count, sizeof(int));
//    ERROR_CHECK(Status);

//    int h_circle_count;
//    Status = cudaMemcpyFromSymbol(&h_circle_count, d_circle_count, sizeof(int), 0, cudaMemcpyDeviceToHost);
//    ERROR_CHECK(Status);

//    circle h_circles[h_circle_count];
//    Status = cudaMemcpyFromSymbol(h_circles, d_circles, sizeof(circle)*h_circle_count, 0, cudaMemcpyDeviceToHost);
//    ERROR_CHECK(Status);




}


__host__ void cuda_test_get_global_circles(circle* c, int count) {

    cudaError_t Status = cudaSuccess;

    double *circlex;
    double *circley;
    double *circler;

    circlex = (double *) malloc(sizeof(double) * count);
    circley = (double *) malloc(sizeof(double) * count);
    circler = (double *) malloc(sizeof(double) * count);

    for (int i = 0; i < count; i++) {
        circlex[i] = c[i].x;
        circley[i] = c[i].y;
        circler[i] = c[i].r;
    }

    //    Checking to see if the arrays were loaded onto the device correctly.
    double *rcx;
    Status = cudaMemcpyFromSymbol(&rcx, cx, sizeof(double), 0, cudaMemcpyDeviceToHost);
    ERROR_CHECK(Status);

    double *rcy;
    Status = cudaMemcpyFromSymbol(&rcy, cy, sizeof(circley), 0, cudaMemcpyDeviceToHost);
    ERROR_CHECK(Status);

    double *rcr;
    Status = cudaMemcpyFromSymbol(&rcr, cr, sizeof(circler), 0, cudaMemcpyDeviceToHost);
    ERROR_CHECK(Status);


}