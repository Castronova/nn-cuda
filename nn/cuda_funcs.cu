
#include <iostream>
#include "cuda_funcs.h"
#include <cuda_runtime.h>
#include "delaunay.h"
#include <stdlib.h>
//#include <cuda.h>
#include<stdio.h>


//using namespace std;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define HANDLE_ERROR(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define N_SEARCH_TURNON 20
#define N_FLAGS_TURNON 1000
#define N_FLAGS_INC 100
#define STACK_MAX 300

// --------------------
// Cuda Stack Functions
// --------------------
typedef struct cuda_stack {
    int n;
    int nallocated;
    int* v;
} cuda_stack;



struct cuda_stack* cuda_stack_create(void){

    struct cuda_stack* s = (cuda_stack*) malloc(sizeof(struct cuda_stack));

    s->n = 0;
    s->nallocated = STACK_MAX;
    s->v = (int*) malloc(STACK_MAX * sizeof(int));
    return s;
}
void cuda_stack_reset(cuda_stack* s) {
    s->n = 0;
}
int cuda_stack_contains(cuda_stack* s, int v) {
    int i;

    for (i = 0; i < s->n; ++i)
        if (s->v[i] == v)
            return 1;
    return 0;
}
void cuda_stack_push(cuda_stack* s, int v) {
    if (s->n == s->nallocated) {
        printf("Reached maximum stack size...exiting \n");
//        s->nallocated *= 2;
//        s->v = (int*)realloc(s->v, s->nallocated * sizeof(int));
    }

    s->v[s->n] = v;
    s->n++;
}
int cuda_stack_pop(cuda_stack* s) {
    s->n--;
    return s->v[s->n];
}
int cuda_stack_getnentries(cuda_stack* s) {
    return s->n;
}
int* cuda_stack_getentries(cuda_stack* s) {
    return s->v;
}



void contains(double px, double py, double cx, double cy, double cr, int* found){

    // note: hypot doesn't work in the kernel
    double dist = (cx - px) * (cx - px) + (cy - py) * (cy - py);
    double radi = cr * cr;
    if (dist <= radi){
        *found = 1;
    }
}



void cuda_find_neighboring_delaunay(double ntris, int* circle_ids, int* n_point_triangles, int** point_triangles,
                                    triangle* triangles, double* ptx, double* pty, int npts, double* cx, double* cy, double* cr,
                                    int* n_out, int** v_out){

    // ---------------------------------------------------------
    // this function should mimick lines 575 - 602 in delaunay.c
    // ---------------------------------------------------------

//    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_id = 0;
    if(thread_id < ntris) {

        // get the triangle id
        int tid = circle_ids[thread_id];

        printf("POINT(%3.5f,%3.5f)   ", ptx[tid], pty[tid]);

        // initialize flag object to ensure that initial values will not match triangle ids
        int flags [100];
        for (int f = 0; f < 100; f++){
            flags[f ] = -999;
        }

        int flag_count = 0;

        // define a container to hold triangle ids
        triangle* t;

        // build stack to store triangle info
        cuda_stack* t_in = cuda_stack_create();
        cuda_stack* t_out = cuda_stack_create();
        cuda_stack_push(t_in, tid);

        while (t_in->n > 0) {

            // get triangle, triangle should be a array [ntris][3], containing the vertex ids for each triangle
            tid = cuda_stack_pop(t_in);

            // get the triangles associates with the tid
            t = &triangles[tid];

            int does_contain = 0;
            contains(ptx[thread_id], pty[thread_id], cx[tid], cy[tid], cr[tid], &does_contain);
            if (does_contain == 1) {

                // add this triangle id to the t_out stack
                cuda_stack_push(t_out, tid);

                // loop through these triangle vertices
                for (int i = 0; i < 3; i++) {

                    // get the vertex id
                    int vid = t->vids[i];

                    // get number of point triangles associates with this vid
                    int nt = n_point_triangles[vid];

                    // loop through point triangles associated with this vid
                    for (int j = 0; j < nt; j++) {

                        // get the id of the current point triangle
                        int ntid = point_triangles[vid][j];

                        int flagged = 0;
                        // check if this triangle has already be flagged (i.e. checked)
                        for (int k = 0; k < flag_count; k++) {
                            if (flags[k] == ntid) {
                                flagged = 1;
                            }
                        }
                        // if triangle has not been added yet (i.e. flagged)
                        if (flagged == 0) {
                            // add triangle to the t_in stack
                            cuda_stack_push(t_in, ntid);

                            // add a flag so this value is not checked again
                            flags[flag_count] = ntid;
                            flag_count++;
                        }

                        // set flags so that this id will not be added more than once

                    }
                }

            }

        }
        // set return data
        printf("Found %d triangles: ", t_out->n);
        for (int t = 0; t < t_out->n; t++) {
            printf("%d ", t_out->v[t]);
        }
        printf("\n");


        // save output data
        n_out[thread_id] = t_out->n;
        v_out[thread_id] = t_out->v;


        // increment thread id
//        thread_id += blockDim.x * gridDim.x;
        thread_id ++;
    }

}





// This function performs the equivilent of the linear search in delaunay.c, delaunay_circles_find() (line 463)
__global__ void _cuda_search_all_tricircles(double* ptx, double* pty, int npts, double* cx, double* cy, double* cr, int* fidx, int count, int *outCount){
//     pts:    2d array of search point coordinates
//     npts:   size of the pts array (i.e. number of total points)
//     cx:     circle x coordinates
//     cy:     circle y coordinates
//     cz:     circle z coordinates
//     fidx:   array of circle indices in which the pts intersect
//     count:  number of circles
//     outCount: number of matching circles for each point index

    int idx = 0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stidx = tid * 30;
    if(tid < npts){

        // set the initial values for tid
        for (int t = 0; t < 30; t ++)
            fidx[stidx + t] = -999;

        // grab the next point
        double px = ptx[tid];
        double py = pty[tid];

        // loop over all circles to find match
        int i;
        for (i = 0; i < count; i++) {

            // note: hypot doesn't work in the kernel
            double dist = (cx[i] - px) * (cx[i] - px) + (cy[i] - py) * (cy[i] - py);
            double radi = cr[i] * cr[i];

            // check for tricircle match
            if (dist <= radi) {
                fidx[stidx+ idx] = i;
                idx ++;
            }
        }

        // save the idx count
        outCount[tid] = idx;

        // increment thread id
        tid += blockDim.x * gridDim.x;
    }
}



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

        // set the initial valiue for tid
        fidx[tid] = -999;

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



//int cuda_delaunay_circles_find_all(double pts[][2], int npts, double* cx, double* cy, double* cr, int n_cir){
int* cuda_delaunay_circles_find_all(delaunay* d, point* p, int npts){

    int n_cir = d->ntriangles;
    // reconstruct the delauney objects into simple arrays
    double cx[d->ntriangles], cy[d->ntriangles], cr[d->ntriangles];
    for (int i=0; i<d->ntriangles; i++){
        cx[i] = d->circles[i].x;
        cy[i] = d->circles[i].y;
        cr[i] = d->circles[i].r;

    }

    // reconstruct the point object into simple arrays
    double* ptx = (double*) malloc(npts * sizeof(double));
    double* pty = (double*) malloc(npts * sizeof(double));
    for (int i =0; i< npts; i++){
        ptx[i] = p[i].x;
        pty[i] = p[i].y;
    }

//    cudaGetDevice(0);
//    cudaDeviceReset();

    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the

    // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, _search_cuda_all, 0, npts);
    // Round up according to array size
    gridSize = (npts + blockSize - 1) / blockSize;

    int* d_idx;
    double *d_cx, *d_cy, *d_cr, *d_ptx, *d_pty;

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



    // 593424.65000, 5676316.98000, 0.00000: 	 398
    // 593449.67715, 5676316.98000, 0.00000: 	 398
    // 593474.70430, 5676316.98000, 0.00000: 	 398
    // 593499.73145, 5676316.98000, 0.00000: 	 398
    // 593524.75860, 5676316.98000, 0.00000: 	 398
    // 593549.78575, 5676316.98000, 0.00000: 	 398
    // 593574.81290, 5676316.98000, 0.00000: 	 398
    // 593599.84005, 5676316.98000, 0.00000: 	 398


//    cout << "-----------------\n";
//    for (int i = 0; i < 10; i++){
//        cout << "res " << i << ": " << res[i] << endl;
//        if (res[i] >= 0) {
//            double valid = (cx[res[i]] - ptx[i]) * (cx[res[i]] - ptx[i]) + (cy[res[i]] - pty[i]) * (cy[res[i]] - pty[i]);
//            cout << "Index " << i << ", POINT(" << ptx[i] << ", " << pty[i] << ") IDX: " << res[i] << "\t\t\t" <<
//            valid << endl;
//        }
//    }

    // free memory
    cudaFree(d_idx);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_cr);
    cudaFree(d_ptx);
    cudaFree(d_pty);

    // return the search idx array
    return res;
}



void cuda_delaunay_circles_find_all_tricircles(delaunay* d, point* p, int npts, int **matches, int **nmatches){

    // reconstruct the point object into simple arrays
    double* ptx = (double*) malloc(npts * sizeof(double));
    double* pty = (double*) malloc(npts * sizeof(double));
    for (int i =0; i< npts; i++){
        ptx[i] = p[i].x;
        pty[i] = p[i].y;
    }

    int n_cir = d->ntriangles;
    // reconstruct the delauney objects into simple arrays
    double cx[d->ntriangles], cy[d->ntriangles], cr[d->ntriangles];
    for (int i=0; i<d->ntriangles; i++){
        cx[i] = d->circles[i].x;
        cy[i] = d->circles[i].y;
        cr[i] = d->circles[i].r;

    }

    cudaGetDevice(0);
    cudaDeviceReset();

    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the

    // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, _search_cuda_all, 0, npts);
    // Round up according to array size
    gridSize = (npts + blockSize - 1) / blockSize;

    int* d_idx, *d_count;
    double *d_cx, *d_cy, *d_cr, *d_ptx, *d_pty;
    triangle *d_triangles;

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

    HANDLE_ERROR(cudaMalloc((void **) &d_triangles, d->ntriangles * sizeof(triangle)));
    HANDLE_ERROR(cudaMemcpy(d_triangles, d->triangles, d->ntriangles * sizeof(triangle), cudaMemcpyHostToDevice));

    // allocate 30 spaces for each point (flattened array)
    HANDLE_ERROR(cudaMalloc((void **) &d_idx, 30 * npts * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_count, npts * sizeof(int)));

    *matches = (int*)malloc(30 * npts * sizeof(int));
    *nmatches = (int*)malloc(npts * sizeof(int));
    _cuda_search_all_tricircles <<< gridSize, blockSize >>> (d_ptx, d_pty, npts, d_cx, d_cy, d_cr, d_idx, n_cir, d_count);
    HANDLE_ERROR(cudaMemcpy(*matches, d_idx, 30 * npts * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(*nmatches, d_count, npts * sizeof(int), cudaMemcpyDeviceToHost));


    // free memory
    cudaFree(d_idx);
    cudaFree(d_count);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_cr);
    cudaFree(d_ptx);
    cudaFree(d_pty);
    cudaFree(d_triangles);

}


/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return;
    std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
    exit (1);
}


