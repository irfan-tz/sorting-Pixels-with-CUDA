#include "header.hpp"

int main(int argc, char *argv[]) {
    auto [inputImage, outputImage] = parseCommandLineArguments(argc, argv);
    cout << "Parsed.\n";

    try {
        Mat img = readImageFromFile(inputImage);
        if (img.empty()) {
            cerr << "Failed to load image.\n";
            return 1;
        }
        cout << "Image loaded successfully.\n";

        const int rows = img.rows;
        const int columns = img.cols;
        const int total_pixels = rows * columns;
        const int imageSize = total_pixels * 3 * sizeof(uchar);

        uchar* imageData = img.data;
        const int step = img.step;

        std::cout << "rows: " << rows << " cols: " << columns << " total_pixels: " << total_pixels 
                  << " imageSize: " << imageSize << " step: " << step << '\n';

        // Allocate device memory
        uchar* d_imageData;
        uchar* d_temp;
        cudaMalloc(&d_imageData, imageSize);
        cudaMalloc(&d_temp, imageSize);

        // Copy image data to device
        cudaMemcpy(d_imageData, imageData, imageSize, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = columns;
        bitonicSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_imageData, step, rows, columns);
        cudaDeviceSynchronize();

        // Copy sorted image data back to host
        cudaMemcpy(imageData, d_imageData, imageSize, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_imageData);
        cudaFree(d_temp);

        // Create a new sorted image
        vector<int> compression_params = {IMWRITE_PNG_COMPRESSION, 9};

        if (!imwrite(outputImage, img, compression_params)) {
            cerr << "Failed to write gpu image to file.\n";
            return 1;
        }
        cout << "Successfully written gpu image to file.\n";

        // sorting with cpu

        uchar* temp = new uchar[imageSize];
        for (int i = 0; i < columns; i++) {
            mergeSortCPU(imageData, step, i, 0, rows - 1, temp);
        }
        delete[] temp;

        if (!imwrite("cpuImage.png", img, compression_params)) {
            cerr << "Failed to write cpu image to file.\n";
            return 1;
        }
        cout << "Successfully written cpu image to file.\n";

    } catch (cv::Exception &error_) {
        cerr << "Caught OpenCV exception: " << error_.what() << endl;
        return 1;
    }

    return 0;
}