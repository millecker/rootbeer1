#include <stdio.h>
#ifndef NAN
#include <math_constants.h>
#define NAN CUDART_NAN
#endif

#ifndef INFINITY
#include <math_constants.h>
#define INFINITY CUDART_INF
#endif

#include <stdio.h>

__shared__ size_t m_Local[3];
<<<<<<< HEAD
__shared__ char m_shared[40*1024];
=======
__shared__ char m_shared[%%shared_mem_size%%];
>>>>>>> 56f1a04b81d80e4356d3decc3e22ef176f2fd6c7

__device__
int getThreadId(){
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__
int getThreadIdxx(){
  return threadIdx.x;
}

__device__
int getBlockIdxx(){
  return blockIdx.x;
}

__device__
int getBlockDimx(){
  return blockDim.x;
}

__device__
<<<<<<< HEAD
=======
int getGridDimx(){
  return blockDim.x;
}

__device__
>>>>>>> 56f1a04b81d80e4356d3decc3e22ef176f2fd6c7
void edu_syr_pcpratts_syncthreads(){
  __syncthreads();
}

__device__
void edu_syr_pcpratts_threadfence(){
  __threadfence();
}

__device__
void edu_syr_pcpratts_threadfence_block(){
  __threadfence_block();
}

__device__ clock_t global_now;