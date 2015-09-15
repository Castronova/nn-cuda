#include "cuda_funcs.h"
#include "cuda_runtime.h"

__global__ void _cuda_nnpi_normalize_weights(nnpi* nn)
{

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

__host__ void cuda_nnpi_normalize_weights(nnpi* nn)
{

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