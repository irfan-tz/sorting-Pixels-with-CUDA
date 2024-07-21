#include "header.hpp"

__device__ int getPixelValueGPU(const uchar* image, int step, int x, int y) {
    int idx = (y * step) + (x * 3);
    return image[idx] + image[idx + 1] + image[idx + 2];
}

__device__ void swapPixels(uchar* image, int step, int x, int y1, int y2) {
    for (int c = 0; c < 3; c++) {
        uchar temp = image[y1 * step + x * 3 + c];
        image[y1 * step + x * 3 + c] = image[y2 * step + x * 3 + c];
        image[y2 * step + x * 3 + c] = temp;
    }
}

__global__ void bitonicSortKernel(uchar* d_image, int step, int rows, int cols) {
    int tid = threadIdx.x;
    int col = blockIdx.x;

    if (col >= cols) return;

    // Bitonic sort
    for (int k = 2; k <= rows; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = tid; i < rows; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i && ixj < rows) {
                    int value_i = getPixelValueGPU(d_image, step, col, i);
                    int value_ixj = getPixelValueGPU(d_image, step, col, ixj);
                    
                    bool ascending = ((i & k) == 0);
                    if ((ascending && value_i > value_ixj) || (!ascending && value_i < value_ixj)) {
                        swapPixels(d_image, step, col, i, ixj);
                    }
                }
            }
            __syncthreads();
        }
    }
}

// ----------------------------------CPU---------------------------------------

int getPixelValueCPU(const uchar* image, int step, int x, int y) {
    int idx = (y * step) + (x * 3);
    return image[idx] + image[idx + 1] + image[idx + 2];
}

// Function to merge two halves
void merge(uchar* image, int step, int col, int left, int middle, int right, uchar* temp) {
    int n1 = middle - left + 1;
    int n2 = right - middle;

    // Copy data to temp arrays
    for (int i = 0; i < n1; i++) {
        for (int c = 0; c < 3; c++) {
            temp[(left + i) * step + col * 3 + c] = image[(left + i) * step + col * 3 + c];
        }
    }
    for (int j = 0; j < n2; j++) {
        for (int c = 0; c < 3; c++) {
            temp[(middle + 1 + j) * step + col * 3 + c] = image[(middle + 1 + j) * step + col * 3 + c];
        }
    }

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        int leftValue = getPixelValueCPU(temp, step, col, left + i);
        int rightValue = getPixelValueCPU(temp, step, col, middle + 1 + j);

        if (leftValue <= rightValue) {
            for (int c = 0; c < 3; c++) {
                image[k * step + col * 3 + c] = temp[(left + i) * step + col * 3 + c];
            }
            i++;
        } else {
            for (int c = 0; c < 3; c++) {
                image[k * step + col * 3 + c] = temp[(middle + 1 + j) * step + col * 3 + c];
            }
            j++;
        }
        k++;
    }

    while (i < n1) {
        for (int c = 0; c < 3; c++) {
            image[k * step + col * 3 + c] = temp[(left + i) * step + col * 3 + c];
        }
        i++;
        k++;
    }

    while (j < n2) {
        for (int c = 0; c < 3; c++) {
            image[k * step + col * 3 + c] = temp[(middle + 1 + j) * step + col * 3 + c];
        }
        j++;
        k++;
    }
}

// Function to perform merge sort
void mergeSortCPU(uchar* image, int step, int col, int left, int right, uchar* temp) {
    if (left < right) {
        int middle = left + (right - left) / 2;

        mergeSortCPU(image, step, col, left, middle, temp);
        mergeSortCPU(image, step, col, middle + 1, right, temp);
        merge(image, step, col, left, middle, right, temp);
    }
}