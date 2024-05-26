#include "./mnist.h"
#include <iostream>
#include <random>
#include <bits/stdc++.h> 
//#include "./const.h"

__constant__ float c_weightConv1[6 * 25];
__constant__ float c_biasConv1[6];
__constant__ float c_weightConv2[16 * 25];
__constant__ float c_biasConv2[16];

__constant__ int a[6*16] = {1, 1, 1, 0, 0, 0,
                    0, 1, 1, 1, 0, 0,
                    0, 0, 1, 1, 1, 0,
                    0, 0, 0, 1, 1, 1,
                    1, 0, 0, 0, 1, 1,
                    1, 1, 0, 0, 0, 1,
                    1, 1, 1, 1, 0, 0,
                    0, 1, 1, 1, 1, 0,
                    0, 0, 1, 1, 1, 1,
                    1, 0, 0, 1, 1, 1,
                    1, 1, 0, 0, 1, 1,
                    1, 1, 1, 0, 0, 1,
                    1, 1, 0, 1, 1, 0,
                    0, 1, 1, 0, 1, 1,
                    1, 0, 1, 1, 0, 1, 
                    1, 1, 1, 1, 1, 1};  

class Convolution{
    public:
        //inform of Kernel (numKernel, width, height)
        int numKernel = 1;
        int widthKernel = 0;
        int padding = 0;

        //inform of input (numInput, width, height)
        int widthIn;
        int heightIn;
        int numInput;

        //inform of output (numOut, width, height)
        //numOut == numKernel so don't need assign
        int widthOut;
        int heightOut;
        
        //weight and bias in host
        float* weight = NULL;
        float* bias = NULL;

        //weight and bias in device;
        float* d_weight = NULL;
        float* d_bias = NULL;
        float* d_out = NULL;
        float* out = NULL;


        // init value of convolution;
        ~Convolution();
        Convolution(int num, int width, int inHeight, int inWidth);
        float** forward1(float* data);

        //init kernel by random way
        void initKernel();

        void copyWeightandBias();

};

__device__ float activation(float x);

//convolution, input, size
__global__ void convForward(float* in, int widthIn, int heightIn, 
                            float *output, int widthOut, int heightOut, 
                            float* weight, float* bias, 
                            int numKernel, int widthKernel, int pading);

__global__ void convForward2(float* in, int widthIn, int heightIn, int numInput,
                            float *output, int widthOut, int heightOut, 
                            float* weight, float* bias, 
                            int numKernel, int widthKernel, int pading);

__global__ void convForwardVer1(float* in, int widthIn, int heightIn, 
                            float *output, int widthOut, int heightOut, 
                            int numKernel, int widthKernel, int pading);

__global__ void convForward2Ver1(float* in, int widthIn, int heightIn, int numInput,
                            float *output, int widthOut, int heightOut, 
                            int numKernel, int widthKernel, int pading);                         


Convolution::Convolution(int num, int width, int inHeight, int inWidth){
    
    this->numKernel = num;
    this->widthKernel = width;
    this->padding = (width - 1)/2;
    this->widthIn = inHeight;
    this->heightIn = inWidth;
    this->widthOut = this->widthIn -  this->padding * 2;
    this->heightOut =  this->heightIn -  this->padding * 2;
}

Convolution::~Convolution(){
    cudaFree(this->d_weight);
    cudaFree(this->d_bias);
    cudaFree(this->d_out);
    delete[] out;
}

void Convolution::initKernel(){

  srand(time(0)); 
    //init for weight
    this->weight = new float[this->numKernel* this->widthKernel * this->widthKernel];
    this->bias = new float[this ->numKernel];

    size_t weightSize = this->numKernel * this->widthKernel * this->widthKernel * sizeof(float);
    //float* temp = new float[weightSize];
    cudaMalloc(&this->d_weight, weightSize);

    for (int i = 0 ; i < this->numKernel; i++){
        //this->weight[i] = new float[widthKernel * widthKernel];
        for (int j = 0; j < this->widthKernel* this->widthKernel; j++){
            this->weight[i * this->widthKernel * this->widthKernel + j ] = rand();
            //temp[i * this->widthKernel * this->widthKernel + j ] = this->weight[i][j];
        }
        this->bias[i] = rand();
    }    

  cudaMemcpy(this->d_weight, this->weight, weightSize, cudaMemcpyHostToDevice);

  size_t biasSize= this->numKernel * sizeof(float);
  cudaMalloc(&this->d_bias, biasSize);
	cudaMemcpy(this->d_bias, this->bias, biasSize, cudaMemcpyHostToDevice);

  size_t outSize= this->numKernel*this->widthOut * this->heightOut * sizeof(float);
  cudaMalloc(&this->d_out, outSize);
  cudaMemset(this->d_out, 0x00, outSize);
}

__device__ float activation(float x){
  return x > 0.0f ?  x: 0.00001;
}

__global__ void convForward(float* in, int widthIn, int heightIn, 
                            float *output, int widthOut, int heightOut, 
                            float* weight, float* bias, 
                            int numKernel, int widthKernel, int pading){
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int i = r * widthOut + c;

  if ( c < widthOut && r < heightOut){
    for (int layer = 0; layer < numKernel; layer++){
      int pointX = c + pading;
      int pointY = r + pading; 
      float temp = 0.0;
      for (int fr = 0; fr < widthKernel; fr++){
          for (int fc = 0; fc < widthKernel; fc++){
              temp += in[(pointY - pading + fr) * widthIn + (pointX - pading + fc)] * weight[layer * (widthKernel * widthKernel)+fr * widthKernel + fc];
          }
      }
      output[layer * (widthOut * heightOut) + i] = activation(temp + bias[layer]);
    }
  }

}

__global__ void convForwardVer1(float* in, int widthIn, int heightIn, 
                          float *output, int widthOut, int heightOut, 
                          int numKernel, int widthKernel, int pading){
  int col = blockIdx.x * (blockDim.x - widthKernel + 1) + threadIdx.x;
	int row = blockIdx.y *  (blockDim.x - widthKernel + 1)  + threadIdx.y;
	int row_i = row - widthKernel + 1;
	int col_i = col - widthKernel + 1;

	extern __shared__ float s_data[];

	if (row_i < heightIn && row_i >= 0 && col_i < widthIn && col_i >= 0)
	{
		s_data[threadIdx.y* blockDim.x + threadIdx.x] = in[col_i * widthIn + row_i];
	}
	else
	{
		s_data[threadIdx.y* blockDim.x + threadIdx.x] = 0;
	}

	__syncthreads();

	if (threadIdx.y < (blockDim.y - widthKernel + 1) && threadIdx.x < (blockDim.x - widthKernel + 1) && row < (heightIn - widthKernel + 1) && col < (widthIn - widthKernel + 1))
	{
    for (int layer = 0; layer < numKernel; layer++){
      float tmp = 0;
      for (int i = 0; i< widthKernel;i++){
        for (int j = 0;j < widthKernel;j++)
          tmp += s_data[(threadIdx.y + i)* blockDim.x + threadIdx.x + j] * c_weightConv1[layer * (widthKernel * widthKernel) + j*widthKernel + i];
      }
      output[layer * (widthOut * heightOut) + col*widthOut + row] = activation(tmp+c_biasConv1[layer]);
    }
  }
}

__global__ void convForward2(float* in, int widthIn, int heightIn, int numInput,
                            float *output, int widthOut, int heightOut, 
                            float* weight, float* bias, 
                            int numKernel, int widthKernel, int pading){
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int i = r * widthOut + c;

  if ( c < widthOut && r < heightOut){
    for (int layer = 0; layer < numKernel; layer++){
      if (a[layer*6 + numInput] == 1){
        int pointX = c + pading;
        int pointY = r + pading; 
        float temp = 0.0;
        for (int fr = 0; fr < widthKernel; fr++){
            for (int fc = 0; fc < widthKernel; fc++){
                temp += in[(pointY - pading + fr) * widthIn + (pointX - pading + fc)] * weight[layer*(widthKernel * widthKernel)+fr * widthKernel + fc];
            }
        }
        output[layer * (widthOut * heightOut) + i] += activation(temp + bias[layer]);
      }
    }
  }
}


__global__ void convForward2Ver1(float* in, int widthIn, int heightIn, int numInput,
                            float *output, int widthOut, int heightOut, 
                            int numKernel, int widthKernel, int pading){

  int col = blockIdx.x * (blockDim.x - widthKernel + 1) + threadIdx.x;
	int row = blockIdx.y *  (blockDim.x - widthKernel + 1)  + threadIdx.y;
	int row_i = row - widthKernel + 1;
	int col_i = col - widthKernel + 1;

	extern __shared__ float s_data[];

	if (row_i < heightIn && row_i >= 0 && col_i < widthIn && col_i >= 0)
	{
		s_data[threadIdx.y* blockDim.x + threadIdx.x] = in[col_i * widthIn + row_i];
	}
	else
	{
		s_data[threadIdx.y* blockDim.x + threadIdx.x] = 0;
	}

	__syncthreads();

	if (threadIdx.y < (blockDim.y - widthKernel + 1) && threadIdx.x < (blockDim.x - widthKernel + 1) && row < (heightIn - widthKernel + 1) && col < (widthIn - widthKernel + 1))
	{
    for (int layer = 0; layer < numKernel; layer++){
      if (a[layer*6 + numInput] == 1){
        float tmp = 0;
        for (int i = 0; i< widthKernel;i++){
          for (int j = 0;j < widthKernel;j++)
            tmp += s_data[(threadIdx.y + i)* blockDim.x + threadIdx.x + j] * c_weightConv2[layer * (widthKernel * widthKernel) + j*widthKernel + i];
        }
        output[layer * (widthOut * heightOut) + col*widthOut + row] += activation(tmp+c_biasConv2[layer]);
      }
    }
  }
}

