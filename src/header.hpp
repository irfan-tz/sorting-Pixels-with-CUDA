#include <iostream>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

__host__ std::tuple<std::string, std::string> parseCommandLineArguments(int argc, char *argv[]);
__host__ cv::Mat readImageFromFile(std::string inputFile);

__device__ int getPixelValueGPU(const uchar *image, int step, int x, int y);
__global__ void bitonicSortKernel(uchar *d_image, int step, int rows, int cols);

int getPixelValueCPU(const uchar *image, int step, int x, int y);
void merge(uchar *image, int step, int col, int left, int middle, int right, uchar *temp);
void mergeSortCPU(uchar *image, int step, int col, int left, int right, uchar *temp);