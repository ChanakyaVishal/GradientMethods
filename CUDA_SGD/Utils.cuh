#ifndef UTILS
#define UTILS

/********************/
/* CUDA ERROR CHECK */
/********************/
void gpuAssert(cudaError_t code, const char *file, int line, bool abort);

/*******************/
/* iDivUp FUNCTION */
/*******************/
__host__ __device__ int iDivUp(int a, int b);


void gpuErrchk(cudaError_t ans);

/*******************/
/* RANDOM FUNCTION */
/*******************/
//__global__ void random(int* result, const int n);

#endif
