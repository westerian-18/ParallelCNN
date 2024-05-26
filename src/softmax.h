__global__ void reduceSum(float * in, float * out,int n)
{
  int i =  blockIdx.x * blockDim.x * 2 + threadIdx.x;

  if (i < n){
    for (int stride = blockDim.x; stride > 0 ; stride /= 2){
        if (i + stride < n && threadIdx.x < stride)
          in[i] += in[i + stride];
      __syncthreads(); // Synchronize within each block
    }
    if (threadIdx.x == 0)
      //out[blockIdx.x] = in[numElemsBeforeBlk];
      atomicAdd(out, in[i]);
  }
}

__global__ void softmax(float* in, int n, float *out, float sum){
  int i =  blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n){
    out[i] /= sum;
  }
}