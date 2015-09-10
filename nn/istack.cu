/******************************************************************************
 *
 * File:           istack.c
 *
 * Created:        06/06/2001
 *
 * Author:         Pavel Sakov
 *                 CSIRO Marine Research
 *
 * Purpose:        Handling stack of integers
 *
 * Description:    None
 *
 * Revisions:      None
 *
 *****************************************************************************/

#define STACK_NSTART 50
#define STACK_NINC 50

#include <stdlib.h>
#include <string.h>
#include "istack.h"
#include "cuda_runtime.h"

__device__ istack* istack_create(void)
{
    // tony: added type cast for nvcc compiler
    istack* s = (istack*) malloc(sizeof(istack));

    s->n = 0;
    s->nallocated = STACK_NSTART;
    // tony: added type cast for nvcc compiler
    s->v = (int*) malloc(STACK_NSTART * sizeof(int));
    return s;
}

__device__ void istack_destroy(istack* s)
{
    if (s != NULL) {
        free(s->v);
        free(s);
    }
}

__device__ void istack_reset(istack* s)
{
    s->n = 0;
}

__device__ int istack_contains(istack* s, int v)
{
    int i;

    for (i = 0; i < s->n; ++i)
        if (s->v[i] == v)
            return 1;
    return 0;
}

__device__ void istack_push(istack* s, int v)
{
    if (s->n == s->nallocated) {
        s->nallocated *= 2;
        // tony: added type cast for nvcc compiler
        s->v = (int*) realloc(s->v, s->nallocated * sizeof(int));
    }

    s->v[s->n] = v;
    s->n++;
}

__device__ int istack_pop(istack* s)
{
    s->n--;
    return s->v[s->n];
}

__device__ int istack_getnentries(istack* s)
{
    return s->n;
}

__device__ int* istack_getentries(istack* s)
{
    return s->v;
}
