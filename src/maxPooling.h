#include "./mnist.h"
#include <iostream>

class maxPooling{
    public:
        int widthKernel = 0;

        //inform of input (numInput, width, height)
        int widthIn;
        int heightIn;
        int numInput;

        //inform of output (numOut, width, height)
        //numOut == numKernel so don't need assign
        int widthOut;
        int heightOut;

        float* d_out = NULL;

        void initKernel();
        

        maxPooling(int width, int widthIn, int heightIn, int numInput);

        float** forward(float** data, int chanel, int width, int height);
        ~maxPooling();
};

__global__ void maxPoolingForward(float *data, float * out,
                                   int numInput, int widthIn, int heightIn, int widthKernel);

maxPooling::maxPooling(int width, int widthIn, int heightIn, int numInput){

  this->widthKernel = width;
  this->widthIn = widthIn;
  this->heightIn = heightIn;
  this->widthOut = widthIn / width; 
  this->heightOut = heightIn / width; 
  this->numInput = numInput;
}

void maxPooling::initKernel()
{
  size_t outSize = this->numInput * this->widthOut * this->heightOut * sizeof(float);
  cudaMalloc(&this->d_out, outSize);
}

maxPooling::~maxPooling()
{
  cudaFree(&this->d_out);
}


float** maxPooling::forward(float** data, int chanel, int width, int height){
    float** output;
    int widthOut = width / this->widthKernel; 
    int heightOut = height / this->widthKernel; 
    output = new float*[chanel];
    for (int n = 0; n < chanel; n++){
        output[n] = new float[widthOut*heightOut];
        for (int i = 0; i < heightOut; i++){
            for (int j = 0; j < widthOut; j++){
                int cf = i * this->widthKernel;
                int rf = j * this->widthKernel;
                float max = data[n][rf * width + cf];
                for (int fr = 0; fr < this->widthKernel; fr++){
                    for (int fc = 0; fc < this->widthKernel; fc++){
                        if (max < data[n][(rf + fr)*width + (cf + fc)])
                            max = data[n][(rf + fr)*width + (cf + fc)];
                    }
                }
                output[n][i * widthOut + j] = max;
            }
        }
    }
    this->widthOut = widthOut;
    this->heightOut = heightOut;
    return output;
}


__global__ void maxPoolingForward(float *data, float* out,
                                   int numInput, int widthIn, int heightIn, int widthKernel)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  int widthOut = widthIn / widthKernel; 
  int heightOut = heightIn / widthKernel; 
  int sizeInput = widthIn * heightIn;
  int sizeOutput = widthOut * heightOut;
  int i = r * widthOut + c;

  if ( c < widthOut && r < heightOut){
    for (int layer = 0; layer < numInput; layer++){
      int rf = r * widthKernel;
      int cf = c * widthKernel;
      float max = data[layer * sizeInput + (rf * widthIn) + cf];
      for (int fr = 0; fr < widthKernel; fr++){
          for (int fc = 0; fc < widthKernel; fc++){
              if (max < data[layer * sizeInput + (rf + fr) * widthIn + (cf + fc)])
                  max = data[layer * sizeInput + (rf + fr) * widthIn + (cf + fc)];
          }
      }
      out[(layer * sizeOutput)+ i] = max;
    }
  }

}