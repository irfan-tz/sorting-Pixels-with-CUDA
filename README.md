# Pixel Sorter
This program goes column wise in an image and **sorts the whole column's** pixels based on each **pixel's blue + green + red** value.  
This program was inspired by Youtuber ["Acerola" video of pixel sorting](https://www.youtube.com/watch?v=HMmmBDRy-jE)  

## Project Description
This program uses c++ with nvcc compiler and cuda library to run on the **compatible cuda supported GPU**.  
This program also uses OpenCV library to read the image and then write it in png format.  

The program sorts the columns by gpu using bitonic sort and for cpu it uses merge sort. It sorts with gpu and cpu to check if results are correct.

## Code Organization

```images/```
* The input and sorted output images are stored in this directory.
* Default input image name in the makefile and program is supposed to be "image.png", but you can use different named image by passing argument "i=your_image.png".
* The cpu sorted image is named as "cpuImage.png" and the gpu sorted image is named as "output.png". 

```src/```
* The source code is placed in this directory with the main code block, sort block and reading image block all in seperate files with their header file.  

```bin/```
* This directory contains the binary created by the program.

## Execution
* To build this program, use command >> "make"
* To run this program, use command >> "make run". If not given any input image argument it would expect to read image "./images/image.png". To give a different image as input, use command >> "make run i=./path_to_your_image/your_image.png"

## Proof of execution
* It is supposed to work on one image at a time and the image is supposed to be of PNG format.  
* There is already an input file with its generated output image output.png in ./images dir.   Comparing this to the Acerola's pixel sorting video image where he sorted an image of a duck, mine got the same result.