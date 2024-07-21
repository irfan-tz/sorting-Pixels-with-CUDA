CXX = nvcc

i ?= ./images/image.png
CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)
CXXFLAGS += -diag-suppress 611
all: clean build

build: 
	$(CXX) ./src/main.cu ./src/readImage.cu ./src/sort.cu --std c++17 `pkg-config opencv --cflags --libs` -o ./bin/sortImage.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./bin/sortImage.exe -i $(i) -o output.png 
	mv ./output.png ./images/output.png
	mv ./cpuImage.png ./images/cpuImage.png

clean:
	rm -f ./bin/sortImage.exe ./bin/cpu.exe ./bin/sort.exe ./bin/sort.exp ./bin/sort.lib 
	rm -f ./images/cpuImage.png ./images/output.png