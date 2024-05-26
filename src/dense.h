#include "./mnist.h"
#include <iostream>
#include <random>
#include <bits/stdc++.h> 
#include <cmath>
//#include "./const.h"

class Dense{
    public:

        //inform of input (numInput, width, height)
        int numIn;

        //inform of output (numOut, width, height)
        //numOut == numKernel so don't need assign
        int numOut;
        
        //weight and bias in host
        float* weight = NULL;
        float* bias = NULL;

        //weight and bias in device;
        float* d_weight = NULL;
        float* d_bias = NULL;
        float* d_out = NULL;
        float* out = NULL;


        // init value of convolution;
        Dense(int numin, int numOuth);
        //float** forward1(float* data);

        //init kernel by random way
        void initKernel();
        ~Dense();

        //void copyWeightandBias();

};

//__device__ float activation(float x);
__global__ void DenseForward(float* in, float* weight, float *bias, float* output, int numIn, int numOut);
__global__ void DenseForwardSofmax(float* in, float* weight, float *bias, float* output, int numIn, int numOut);

//convolution, input, size
Dense::Dense(int numin, int numOut){
    //input
    /*
    this->widthIn = inWidth;
    this->heightIn = inHeight;
    this->numInput = inNum;*/

    //kernel
    this->numIn = numin;
    this->numOut = numOut;
    this->weight = new float[numin * numOut];
    this->bias = new float[numOut];

    size_t weightSize = numIn * numOut * sizeof(float);
    size_t biasSize = numOut * sizeof(float);
    
    cudaMalloc(&this->d_weight, weightSize);
    cudaMalloc(&this->d_bias, biasSize);
    cudaMalloc(&this->d_out, this->numOut * sizeof(float));

    for (int i = 0 ; i < numOut; i++){
        for (int j = 0; j < numin; j++){
            this->weight[i * numin + j] = rand();
            //temp[i * this->widthKernel * this->widthKernel + j ] = this->weight[i][j];
        }
        this->bias[i] = rand();
    }
    cudaMemcpy(this->d_weight, this->weight, weightSize, cudaMemcpyHostToDevice); 
    cudaMemcpy(this->d_bias, this->bias, biasSize, cudaMemcpyHostToDevice);    
}

Dense::~Dense(){
  cudaFree(this->d_bias);
  cudaFree(this->d_weight);
  cudaFree(this->d_out);
}


__global__ void DenseForward(float* in, float* weight, float *bias, float* output, int numIn, int numOut){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //int i = r * blockDim.x + c;

  if (i < numOut) {
    float value = 0;
    for (int j = 0; j < numIn; j++) {
      value += in[j] * weight[i * numIn + j];
    }
    output[i] = activation(value + bias[i]);
  }
}

__global__ void DenseForwardSofmax(float* in, float* weight, float *bias, float* output, int numIn, int numOut){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //int i = r * blockDim.x + c;

  if (i < numOut) {
    float value = 0;
    for (int j = 0; j < numIn; j++) {
      value += in[j] * weight[i * numIn + j];
    }
    output[i] = exp(value + bias[i]);
  }
}
