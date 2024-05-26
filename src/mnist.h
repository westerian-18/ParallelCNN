#ifndef _MNIST_H_
#define _MNIST_H_

#include <vector>
#include <string>
#include <fstream>

using namespace std;

using std::vector;


class MNIST {
    private:
        std::string data_dir;

    public:
        int width, height;
        vector<vector<double>> train_data;
        vector<double> train_labels;
        vector<vector<double>> test_data;
        vector<double> test_labels;


        explicit MNIST(std::string data_dir) : data_dir(data_dir) {}
        void read_mnist_data(std::string filename, vector<vector<double>> &data);
        void read_mnist_label(std::string filename, vector<double> &labels);

        void read();
};


int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void MNIST::read_mnist_data(std::string filename, vector<vector<double>> &data)
{
    ifstream file (filename,ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);

        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);

        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);

        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        this->width = n_cols; 
        this->height = n_rows;
        data.resize(number_of_images, vector<double>(n_cols * n_rows));
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    data[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
}


void MNIST::read_mnist_label(std::string filename, vector<double>& labels) {
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));

    magic_number = ReverseInt(magic_number);
    number_of_images = ReverseInt(number_of_images);

    labels.resize(number_of_images);
    for (int i = 0; i < number_of_images; i++) {
      unsigned char label = 0;
      file.read((char*)&label, sizeof(label));
      labels[i] = (double)label;
    }
  }
}

void MNIST::read() {
  read_mnist_data(data_dir + "train-images.idx3-ubyte", train_data);
  read_mnist_data(data_dir + "t10k-images.idx3-ubyte", test_data);
  read_mnist_label(data_dir + "train-labels.idx1-ubyte", train_labels);
  read_mnist_label(data_dir + "t10k-labels.idx1-ubyte", test_labels);
}



#endif