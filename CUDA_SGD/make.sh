nvcc -c -gencode arch=compute_61,code=sm_61 -lineinfo -std=c++11 GradientDescentGPU.cu 
nvcc -c -gencode arch=compute_61,code=sm_61 -lineinfo -std=c++11  kernel.cu 
nvcc -c -gencode arch=compute_61,code=sm_61 -lineinfo -std=c++11 Utils.cu 
nvcc kernel.o Utils.o GradientDescentGPU.o -o app

