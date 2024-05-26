#include "./src/mnist.h"
//#include "./src/conv.h"
//#include "./src/dense.h"
//#include "./src/maxPooling.h"
#include <iostream>
#include "./src/time.h"
#include "./src/version.h"

using namespace std;

float** addPading(vector<vector<double>> data, int width, int height, int pading = 2){
    int numOfPic = data.size();
    float** newdata = (float**) malloc(numOfPic * sizeof(float*));
    for (int n = 0; n < numOfPic; n++){
        newdata[n] = (float*) malloc((width + (pading* 2)) * (height +(pading*2)) * sizeof(float));
        for (int i = 0; i < height; i++){
            for (int j = 0; j < width; j++){
                newdata[n][(i + pading)*(width+pading*2) +(j + pading)] = data[n][i * width + j]; 
            }
        }
        for (int i = 0; i < pading; i++){
            for(int k = 0; k < width + pading * 2; k++){
                newdata[n][i * ( width + pading * 2) + k] = newdata[n][pading * ( width + pading * 2) + k ];
                newdata[n][(height * (width + pading * 2) + i)  + k] = newdata[n][(height + pading - 1) * ( width + pading * 2) + k];
            }
        }
         for (int i = 0; i < height + pading * 2; i++){
            for(int k = 0; k < pading; k++){
                newdata[n][i * ( width + pading * 2) + k] = newdata[n][i * ( width + pading * 2) + pading];
                newdata[n][i * ( width + pading * 2) + k + width + pading] = newdata[n][i * ( width + pading * 2) + width + pading - 1];
            }
        }
    }
    return newdata;
}



int main(int argc, const  char **argv)
{
	MNIST dataset("/content/drive/MyDrive/Project/data/");
    dataset.read();
    int n_train = dataset.train_data.size();
    int n_test = dataset.test_data.size();
    std::cout << "mnist train number: " << n_train << std::endl;
    std::cout<< "size pic: "<< dataset.width<<"x"<<dataset.height<< "\n";
    std::cout << "mnist test number: " << n_test << std::endl;
    float** input = addPading(dataset.train_data, dataset.width, dataset.height, 0);

    int blockSize = 32;
    int ver = 0;
    if (argc == 3)
    {
      blockSize = atoi(argv[2]);    
      ver = atoi(argv[1]);  
    }  
    if (argc == 2){
      ver = atoi(argv[1]);
    }
    
    if (ver == 1)
      versionOptimize1(input, blockSize);
    else if (ver == 2)
      versionOptimize2(input,blockSize);
    else verFull(input, blockSize);

    
    /* 
    float in[5] = {2, 3, 5, 6, 8};
    float w[4*5]= {1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8};
    float b[4] = {3, 6, 8, 6};
    float *d_w, *d_b, *d_i, *d_o, *out;
    cudaMalloc(&d_w, 20 * sizeof(float));
    cudaMalloc(&d_b, 4* sizeof(float));
    cudaMalloc(&d_i, 5 * sizeof(float));
    cudaMalloc(&d_o, 4 * sizeof(float));

    cudaMemcpy(d_w, w, 20 * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_b, b, 4* sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_i, in, 5* sizeof(float), cudaMemcpyHostToDevice); 
    DenseForward<<<128,128>>>(d_i, d_w, d_b, d_o, 5, 4);
    out = new float[4];
    cudaMemcpy(out, d_o, 4* sizeof(float), cudaMemcpyDeviceToHost); 
    for (int i = 0; i < 4; i++){
      cout<<out[i]<<"\t";
    }
    delete[] out;
    //versionOptimize1(input);

    /*Convolution conv1(6,5, 32, 32);
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
      
      cout<<endl;
      delete[] out;
      cudaFree(d_ele);

    }   
    timer.Stop();
    float time = timer.Elapsed();
		printf("Basic time: %f ms\n", time); */


    for (int i = 0; i < sizeof(input)/sizeof(input[0]); i++)
      delete[] input[i];
    delete[] input;
    return 0;
}



