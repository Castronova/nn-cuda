/******************************************************************************
 *
 * File:           istack.h
 *
 * Created:        06/06/2001
 *
 * Author:         Pavel Sakov
 *                 CSIRO Marine Research
 *
 * Purpose:        Header for handling stack of integers.
 *
 * Description:    None
 *
 * Revisions:      None
 *
 *****************************************************************************/

#if !defined(_ISTACK_H)
#define _ISTACK_H

#if !defined(_ISTACK_STRUCT)
#define _ISTACK_STRUCT
struct istack;
typedef struct istack istack;
#endif

// The ifdef checks are necessary to prevent name mangling between C and C++ (CUDA)
#ifdef __cplusplus
    extern "C"{
        struct istack {
            int n;
            int nallocated;
            int* v;
        };

        __device__ istack* istack_create(void);
        __device__ void istack_destroy(istack* s);
        __device__ void istack_push(istack* s, int v);
        __device__ int istack_pop(istack* s);
        __device__ int istack_contains(istack* s, int v);
        __device__ void istack_reset(istack* s);
    };
#else

struct istack {
    int n;
    int nallocated;
    int* v;
};

istack* istack_create(void);
void istack_destroy(istack* s);
void istack_push(istack* s, int v);
int istack_pop(istack* s);
int istack_contains(istack* s, int v);
void istack_reset(istack* s);
#endif

#endif
