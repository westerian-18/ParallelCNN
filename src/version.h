#include "./conv.h"
#include "./dense.h"
#include "./maxPooling.h"
#include "./softmax.h"
//#include "./const.h"

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

void softmax (float *input, int n, float *out){
  float *d_out;
  float sum = 0.001;
  CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  reduceSum<<<32,64>>>(input, d_out, n);
  CHECK(cudaMemcpy(&sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  softmax<<<32,64>>>(input, n, out, sum);
  CHECK(cudaFree(d_out));
}

void versionBasic(float** input);
void versionOptimize1(float** input);
void verFull(float **input);
void versionOptimize2(float** input);

void versionBasic(float** input){

    Convolution conv1(6,5, 32, 32);
    conv1.initKernel();

    maxPooling pooling1(2, conv1.widthOut, conv1.heightOut, 6);
    pooling1.initKernel();

    Convolution conv2(16, 5, pooling1.widthOut, pooling1.heightOut);
    conv2.initKernel();

    maxPooling pooling2(2, conv2.widthOut, conv2.heightOut, 16);
    pooling2.initKernel();

    float* d_ele, *out;
    GpuTimer timer;
    timer.Start();
    for (int i = 0; i < 60000; i++){
      

      CHECK(cudaMalloc(&d_ele, 32 * 32 * sizeof(float)));
	    CHECK(cudaMemcpy(d_ele, &input[i], sizeof(input[i]), cudaMemcpyHostToDevice));
      out = new float[pooling1.widthOut * pooling1.heightOut * sizeof(float)];

      dim3 blockSize(32, 32);
      dim3 gridSize(2);

      convForward<<<gridSize,blockSize>>>(d_ele, conv1.widthIn, conv1.heightIn,
                                          conv1.d_out, conv1.widthOut, conv1.heightOut, 
                                          conv1.d_weight, conv1.d_bias, 
                                          conv1.numKernel, conv1.widthKernel, conv1.padding);
      
      maxPoolingForward<<<gridSize,blockSize>>>(conv1.d_out, pooling1.d_out, 
                                                conv1.numKernel, conv1.widthOut, conv1.heightOut , pooling1.widthKernel);

      for (int i = 0; i < conv1.numKernel; i++){
        convForward2<<<gridSize,blockSize>>>(d_ele, pooling1.widthOut, pooling1.heightOut, i,
                                      conv2.d_out, conv2.widthOut, conv2.heightOut, 
                                      conv2.d_weight, conv2.d_bias, 
                                      conv2.numKernel, conv2.widthKernel, conv2.padding);
      }

      maxPoolingForward<<<gridSize,blockSize>>>(conv2.d_out, pooling2.d_out, 
                                                conv2.numKernel, conv2.widthOut, conv2.heightOut , pooling2.widthKernel);

      

      CHECK(cudaMemcpy(out, pooling2.d_out,  pooling2.widthOut * pooling2.heightOut * sizeof(float), cudaMemcpyDeviceToHost));
      CHECK(cudaGetLastError());
      /*
      cout<<"device"<<endl;
      for (int k = 0; k < conv2.numKernel; k++){
        cout<<"layer: "<<k+ 1<<endl;
        for (int j = 0; j < 7; j++){
            cout<<out[k*(pooling1.widthOut * pooling1.heightOut)+ j]<<"\t";
        }
        cout<<"\n\n";
      }
      
      cout<<endl;*/
      delete[] out;
      cudaFree(d_ele);

    }   
    timer.Stop();
    float time = timer.Elapsed();
		printf("Basic time: %f ms\n", time); 
}

void verFull(float** input, int blksize){

    Convolution conv1(6,5, 32, 32);
    conv1.initKernel();


    maxPooling pooling1(2, conv1.widthOut, conv1.heightOut, 6);
    pooling1.initKernel();

    Convolution conv2(16, 5, pooling1.widthOut, pooling1.heightOut);
    conv2.initKernel();

    maxPooling pooling2(2, conv2.widthOut, conv2.heightOut, 16);
    pooling2.initKernel();

    Dense dense1(pooling2.widthOut * pooling2.heightOut * conv2.numKernel, 120);
    Dense dense2(dense1.numOut, 84);
    Dense dense3(dense2.numOut, 10);

    float* d_ele, *out;
    GpuTimer timer;
    timer.Start();
    for (int i = 0; i < 60000; i++){
      

      CHECK(cudaMalloc(&d_ele, 32 * 32 * sizeof(float)));
	    CHECK(cudaMemcpy(d_ele, &input[i], sizeof(input[i]), cudaMemcpyHostToDevice));
      out = new float[dense3.numOut];
      int n = 32 * 32;
      dim3 blockSize(blksize, blksize);
      dim3 gridSize(n / (blksize * blksize) + 1);

      convForward<<<gridSize,blockSize>>>(d_ele, conv1.widthIn, conv1.heightIn,
                                          conv1.d_out, conv1.widthOut, conv1.heightOut, 
                                          conv1.d_weight, conv1.d_bias, 
                                          conv1.numKernel, conv1.widthKernel, conv1.padding);
      
      maxPoolingForward<<<gridSize,blockSize>>>(conv1.d_out, pooling1.d_out, 
                                                conv1.numKernel, conv1.widthOut, conv1.heightOut , pooling1.widthKernel);

      for (int i = 0; i < conv1.numKernel; i++){
        convForward2<<<gridSize,blockSize>>>(pooling1.d_out, pooling1.widthOut, pooling1.heightOut, i,
                                      conv2.d_out, conv2.widthOut, conv2.heightOut, 
                                      conv2.d_weight, conv2.d_bias, 
                                      conv2.numKernel, conv2.widthKernel, conv2.padding);
      }

      maxPoolingForward<<<gridSize,blockSize>>>(conv2.d_out, pooling2.d_out, 
                                                conv2.numKernel, conv2.widthOut, conv2.heightOut , pooling2.widthKernel);

      
      DenseForward<<<gridSize,blockSize>>>(pooling2.d_out, dense1.d_weight, dense1.d_bias, dense1.d_out, dense1.numIn, dense1.numOut);
      DenseForward<<<gridSize,blockSize>>>(dense1.d_out, dense2.d_weight, dense2.d_bias, dense2.d_out, dense2.numIn, dense2.numOut);
      DenseForwardSofmax<<<gridSize,blockSize>>>(dense2.d_out, dense3.d_weight, dense3.d_bias, dense3.d_out, dense3.numIn, dense3.numOut);
      softmax(dense3.d_out, dense3.numOut, dense3.d_out);
      CHECK(cudaMemcpy(out, dense3.d_out,  dense3.numOut * sizeof(float), cudaMemcpyDeviceToHost));
      CHECK(cudaGetLastError());
      /*
      cout<<"device"<<endl;
      for (int k = 0; k < conv2.numKernel; k++){
        cout<<"layer: "<<k+ 1<<endl;
        for (int j = 0; j < 7; j++){
            cout<<out[k*(pooling1.widthOut * pooling1.heightOut)+ j]<<"\t";
        }
        cout<<"\n\n";
      }
      
      cout<<endl;*/
      delete[] out;
      cudaFree(d_ele);

    }   
    timer.Stop();
    float time = timer.Elapsed();
		printf("Basic time: %f ms\n", time); 
}



void versionOptimize1(float** input,  int blksize){

    Convolution conv1(6,5, 32, 32);
    conv1.initKernel();
    size_t filter1 = conv1.numKernel * conv1.widthKernel * conv1.widthKernel * sizeof(float);
    size_t Bias1 = conv1.numKernel * sizeof(float);
    CHECK(cudaMemcpyToSymbol(c_weightConv1, conv1.weight, filter1));
    CHECK(cudaMemcpyToSymbol(c_biasConv1, conv1.bias, Bias1))

    maxPooling pooling1(2, conv1.widthOut, conv1.heightOut, 6);
    pooling1.initKernel();

    Convolution conv2(16, 5, pooling1.widthOut, pooling1.heightOut);
    conv2.initKernel();
    size_t filter2 = conv2.numKernel * conv2.widthKernel * conv2.widthKernel * sizeof(float);
    size_t Bias2 = conv2.numKernel * sizeof(float);
    CHECK(cudaMemcpyToSymbol(c_weightConv2, conv2.weight, filter2));
    CHECK(cudaMemcpyToSymbol(c_biasConv2, conv2.bias, Bias2))
    

    maxPooling pooling2(2, conv2.widthOut, conv2.heightOut, 16);
    pooling2.initKernel();

    Dense dense1(pooling2.widthOut * pooling2.heightOut * conv2.numKernel, 120);
    Dense dense2(dense1.numOut, 84);
    Dense dense3(dense2.numOut, 10);

    float* d_ele, *out;
    GpuTimer timer;
    timer.Start();
    for (int i = 0; i < 60000; i++){
      CHECK(cudaMalloc(&d_ele, 32 * 32 * sizeof(float)));
	    CHECK(cudaMemcpy(d_ele, &input[i], sizeof(input[i]), cudaMemcpyHostToDevice));
      out = new float[dense3.numOut];

      dim3 blockSize(32, 32);
      dim3 gridSize(2);
      size_t share = 32 * 32 * sizeof(float);

      convForwardVer1<<<gridSize,blockSize, share>>>(d_ele, conv1.widthIn, conv1.heightIn,
                                          conv1.d_out, conv1.widthOut, conv1.heightOut,                               
                                          conv1.numKernel, conv1.widthKernel, conv1.padding);
      
      maxPoolingForward<<<gridSize,blockSize>>>(conv1.d_out, pooling1.d_out, 
                                                conv1.numKernel, conv1.widthOut, conv1.heightOut , pooling1.widthKernel);

      for (int i = 0; i < conv1.numKernel; i++){
        convForward2Ver1<<<gridSize,blockSize, share>>>(pooling1.d_out, pooling1.widthOut, pooling1.heightOut, i,
                                      conv2.d_out, conv2.widthOut, conv2.heightOut, 
                                      conv2.numKernel, conv2.widthKernel, conv2.padding);
      }
      CHECK(cudaGetLastError());
      maxPoolingForward<<<gridSize,blockSize>>>(conv2.d_out, pooling2.d_out, 
                                                conv2.numKernel, conv2.widthOut, conv2.heightOut , pooling2.widthKernel);

      
      DenseForward<<<32,64>>>(pooling2.d_out, dense1.d_weight, dense1.d_bias, dense1.d_out, dense1.numIn, dense1.numOut);
      DenseForward<<<32,64>>>(dense1.d_out, dense2.d_weight, dense2.d_bias, dense2.d_out, dense2.numIn, dense2.numOut);
      DenseForwardSofmax<<<32,64>>>(dense2.d_out, dense3.d_weight, dense3.d_bias, dense3.d_out, dense3.numIn, dense3.numOut);
      softmax(dense3.d_out, dense3.numOut, dense3.d_out);
      CHECK(cudaGetLastError());
      CHECK(cudaMemcpy(out, dense3.d_out,  dense3.numOut * sizeof(float), cudaMemcpyDeviceToHost));
      CHECK(cudaGetLastError());
      /*
      cout<<"device"<<endl;
      for (int k = 0; k < conv2.numKernel; k++){
        cout<<"layer: "<<k+ 1<<endl;
        for (int j = 0; j < 7; j++){
            cout<<out[k*(pooling1.widthOut * pooling1.heightOut)+ j]<<"\t";
        }
        cout<<"\n\n";
      }
      
      cout<<endl;*/
      delete[] out;
      cudaFree(d_ele);

    }   
    timer.Stop();
    float time = timer.Elapsed();
		printf("Ver1 Time : %f ms\n", time); 
}

void versionOptimize2(float** input,  int blksize){

    Convolution conv1(6,5, 32, 32);
    conv1.initKernel();
    size_t filter1 = conv1.numKernel * conv1.widthKernel * conv1.widthKernel * sizeof(float);
    size_t Bias1 = conv1.numKernel * sizeof(float);
    CHECK(cudaMemcpyToSymbol(c_weightConv1, conv1.weight, filter1));
    CHECK(cudaMemcpyToSymbol(c_biasConv1, conv1.bias, Bias1))

    maxPooling pooling1(2, conv1.widthOut, conv1.heightOut, 6);
    pooling1.initKernel();

    Convolution conv2(16, 5, pooling1.widthOut, pooling1.heightOut);
    conv2.initKernel();
    size_t filter2 = conv2.numKernel * conv2.widthKernel * conv2.widthKernel * sizeof(float);
    size_t Bias2 = conv2.numKernel * sizeof(float);
    CHECK(cudaMemcpyToSymbol(c_weightConv2, conv2.weight, filter2));
    CHECK(cudaMemcpyToSymbol(c_biasConv2, conv2.bias, Bias2))
    

    maxPooling pooling2(2, conv2.widthOut, conv2.heightOut, 16);
    pooling2.initKernel();

    Dense dense1(pooling2.widthOut * pooling2.heightOut * conv2.numKernel, 120);
    Dense dense2(dense1.numOut, 84);
    Dense dense3(dense2.numOut, 10);

    float* d_ele, *out;
    GpuTimer timer;
    timer.Start();
    for (int i = 0; i < 60000; i++){
      CHECK(cudaMalloc(&d_ele, 32 * 32 * sizeof(float)));
	    CHECK(cudaMemcpy(d_ele, &input[i], sizeof(input[i]), cudaMemcpyHostToDevice));
      out = new float[dense3.numOut];
      int n = 32*32;
      dim3 blockSize(blksize, blksize);
      dim3 gridSize(n/(blksize*blksize)  + 1);
      size_t share = blksize * blksize * sizeof(float);

      convForwardVer1<<<gridSize,blockSize, share>>>(d_ele, conv1.widthIn, conv1.heightIn,
                                          conv1.d_out, conv1.widthOut, conv1.heightOut,                               
                                          conv1.numKernel, conv1.widthKernel, conv1.padding);
      
      maxPoolingForward<<<gridSize,blockSize>>>(conv1.d_out, pooling1.d_out, 
                                                conv1.numKernel, conv1.widthOut, conv1.heightOut , pooling1.widthKernel);

      for (int i = 0; i < conv1.numKernel; i++){
        convForward2Ver1<<<gridSize,blockSize, share>>>(pooling1.d_out, pooling1.widthOut, pooling1.heightOut, i,
                                      conv2.d_out, conv2.widthOut, conv2.heightOut, 
                                      conv2.numKernel, conv2.widthKernel, conv2.padding);
      }
      CHECK(cudaGetLastError());
      maxPoolingForward<<<gridSize,blockSize>>>(conv2.d_out, pooling2.d_out, 
                                                conv2.numKernel, conv2.widthOut, conv2.heightOut , pooling2.widthKernel);

      
      DenseForward<<<gridSize,blockSize>>>(pooling2.d_out, dense1.d_weight, dense1.d_bias, dense1.d_out, dense1.numIn, dense1.numOut);
      DenseForward<<<gridSize,blockSize>>>(dense1.d_out, dense2.d_weight, dense2.d_bias, dense2.d_out, dense2.numIn, dense2.numOut);
      DenseForwardSofmax<<<gridSize,blockSize>>>(dense2.d_out, dense3.d_weight, dense3.d_bias, dense3.d_out, dense3.numIn, dense3.numOut);
      softmax(dense3.d_out, dense3.numOut, dense3.d_out);
      CHECK(cudaGetLastError());
      CHECK(cudaMemcpy(out, dense3.d_out,  dense3.numOut * sizeof(float), cudaMemcpyDeviceToHost));
      CHECK(cudaGetLastError());
      /*
      cout<<"device"<<endl;
      for (int k = 0; k < conv2.numKernel; k++){
        cout<<"layer: "<<k+ 1<<endl;
        for (int j = 0; j < 7; j++){
            cout<<out[k*(pooling1.widthOut * pooling1.heightOut)+ j]<<"\t";
        }
        cout<<"\n\n";
      }
      
      cout<<endl;*/
      delete[] out;
      cudaFree(d_ele);

    }   
    timer.Stop();
    float time = timer.Elapsed();
    printf("Block Size : %d \n", blksize); 
		printf("Ver2 Time : %f ms\n", time); 
}