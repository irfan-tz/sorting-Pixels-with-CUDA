#include "header.hpp"

std::tuple<std::string, std::string> parseCommandLineArguments(int argc, char *argv[]) {
    cout << "Parsing CLI arguments\n";
    std::string inputImage = "image.png";
    std::string outputImage = "sortedImage.png";

    for (int i = 1; i < argc; i++) {
        std::string option(argv[i]);
        if (option == "-i") {
            if (++i < argc) {
                inputImage = argv[i];
            }
        } else if (option == "-o") {
            if (++i < argc) {
                outputImage = argv[i];
            }
        }
    }
    cout << "inputImage: " << inputImage << " outputImage: " << outputImage << "\n";
    return {inputImage, outputImage};
}


cv::Mat readImageFromFile(std::string inputFile) {
    cout << "Reading Image From File\n";
    Mat img = imread(inputFile, IMREAD_COLOR);

    if (img.empty()) {
        cerr << "Error: Could not open or find the image\n";
        return Mat();
    }
    const int rows = img.rows;
    const int columns = img.cols;
    cout << "Rows: " << rows << " Columns: " << columns << "\n";

    return img;
}
