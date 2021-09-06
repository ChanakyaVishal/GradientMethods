#include "Utils.cuh"
#include<stdio.h>
#include <curand.h>
#include <curand_kernel.h>



/********************/
/* CUDA ERROR CHECK */
/********************/
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { exit(code); }
	}
}


/*******************/
/* iDivUp FUNCTION */
/*******************/
//extern "C" int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }
__host__ __device__ int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }


void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }


