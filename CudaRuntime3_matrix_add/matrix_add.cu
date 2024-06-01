#include "cuda_runtime.h"

#include <iostream>
#include <stdio.h>

__global__ void parallelThreadOperation(float* a, float* b, float* debug, int size_rows, int size_cols)
{
	int blockId = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
	int threadId = blockId * blockDim.x * blockDim.y * blockDim.z;
	threadId += threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	
    if (threadId >= size_rows * size_cols)  return;

    debug[threadId] = a[threadId] + b[threadId];
}

cudaError doCudaOperation(float* a, float* b, float* debug, int size_rows, int size_cols)
{
	//vars for cuda mem space
	float* cuda_a;
	float* cuda_b;
	float* cuda_debug;
	cudaError cudaStatus = cudaSuccess;

	//set device
	cudaStatus = cudaSetDevice(0);

    int size = size_rows * size_cols;

	//allocate memory in cuda 
	cudaStatus = cudaMalloc((void**)&cuda_a, size * sizeof(float));
	cudaStatus = cudaMalloc((void**)&cuda_b, size * sizeof(float));
	cudaStatus = cudaMalloc((void**)&cuda_debug, size * sizeof(float));

	cudaStatus = cudaMemcpy(cuda_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(cuda_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

	//grid & block dims
	dim3 n_blocks(1, 1, 1);		// grid size -> grid dim
	dim3 n_threads(size, 1, 1);	// block size -> block dim, max 1024

	parallelThreadOperation << <n_blocks, n_threads >> > (cuda_a, cuda_b, cuda_debug, size_rows, size_cols);
	cudaStatus = cudaGetLastError();
	cudaStatus = cudaDeviceSynchronize();

	//copy result back to host
	cudaStatus = cudaMemcpy(debug, cuda_debug, size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_debug);

	return cudaStatus;
}



int main()
{
	const int size_rows = 3;
    const int size_cols = 2;

	const int size = size_rows * size_cols;

	float a[size_rows][size_cols] = {
		{ 1.0, 2.0 },
		{ 3.0, 4.0 },
		{ 5.0, 6.0 }
	};

	float b[size_rows][size_cols] = {
		{ -1.0, 2.0 },
		{ -3.0, 4.0 },
		{ -5.0, 6.0 }
	};

    
    float debug[size_rows][size_cols];

	cudaError cudaStatus = doCudaOperation((float*)a, (float*)b, (float*)debug, size_rows, size_cols);

	//input
	std::cout << "input : " << std::endl;
	for (int i = 0; i < size_rows; ++i)
	{
		for (int j = 0; j < size_cols; ++j)
		{
			std::cout << a[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	for (int i = 0; i < size_rows; ++i)
	{
		for (int j = 0; j < size_cols; ++j)
		{
			std::cout << b[i][j] << " ";
		}
		std::cout << std::endl;
	}
    std::cout << std::endl;

	//debug
	std::cout << "debug : " << std::endl;
	for (int i = 0; i < size_rows; ++i)
	{
		for (int j = 0; j < size_cols; ++j)
		{
			std::cout << debug[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cin.get();
	return 0;
}

