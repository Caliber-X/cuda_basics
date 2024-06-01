#include "cuda_runtime.h"

#include <iostream>
#include <stdio.h>

__global__ void parallelThreadOperation(float* a, float* b, float* result, 
										int size_a_rows, int size_a_cols,
										int size_b_rows, int size_b_cols,
										int size_result_rows, int size_result_cols
)
{
	int blockId = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
	int threadId = blockId * blockDim.x * blockDim.y * blockDim.z;
	threadId += threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (threadId >= size_result_rows * size_result_cols)  return;

	int idx_row = threadId / size_result_cols;
	int idx_col = threadId % size_result_cols;

	result[threadId] = 0.0f;
	for (int i = 0; i < size_a_cols; i++)
	{
		// result (idx_row, idx_col)
		// a (idx_row, i)
		// b (i, idx_col)
		result[threadId] += a[idx_row * size_a_cols + i] * b[i * size_b_cols + idx_col];
	}
}

cudaError doCudaOperation(
	float* a, float* b, float* result,
	int size_a_rows, int size_a_cols,
	int size_b_rows, int size_b_cols,
	int size_result_rows, int size_result_cols
)
{
	//vars for cuda mem space
	float* cuda_a;
	float* cuda_b;
	float* cuda_result;
	cudaError cudaStatus = cudaSuccess;

	//set device
	cudaStatus = cudaSetDevice(0);

	//allocate memory in cuda 
	cudaStatus = cudaMalloc((void**)&cuda_a, size_a_rows * size_a_cols * sizeof(float));
	cudaStatus = cudaMalloc((void**)&cuda_b, size_b_rows * size_b_cols * sizeof(float));
	cudaStatus = cudaMalloc((void**)&cuda_result, size_result_rows * size_result_cols * sizeof(float));

	cudaStatus = cudaMemcpy(cuda_a, a, size_a_rows * size_a_cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(cuda_b, b, size_b_rows * size_b_cols * sizeof(float), cudaMemcpyHostToDevice);

	//grid & block dims
	dim3 n_blocks(1, 1, 1);		// grid size -> grid dim
	dim3 n_threads(size_result_rows * size_result_cols, 1, 1);	// block size -> block dim, max 1024

	parallelThreadOperation << <n_blocks, n_threads >> > (cuda_a, cuda_b, cuda_result, 
														  size_a_rows, size_a_cols, 
														  size_b_rows, size_b_cols,
														  size_result_rows, size_result_cols
														);
	cudaStatus = cudaGetLastError();
	cudaStatus = cudaDeviceSynchronize();

	//copy result back to host
	cudaStatus = cudaMemcpy(result, cuda_result, size_result_rows * size_result_cols * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_result);

	return cudaStatus;
}



int main()
{
	const int size_a_rows = 2;
	const int size_a_cols = 3;
	float a[size_a_rows][size_a_cols] = {
		{ 1.0, 2.0, 3.0 },
		{ 4.0, 5.0, 6.0 }
	};

	const int size_b_rows = 3;
	const int size_b_cols = 2;
	float b[size_b_rows][size_b_cols] = {
		{ 10.0, 11.0 },
		{ 20.0, 21.0 },
		{ 30.0, 31.0 }
	};

	// check if multiplication is feasible
	if (size_a_cols != size_b_rows)
	{
		std::cout << "multiplication not feasible" << std::endl;
		return 0;
	}

	const int size_result_rows = size_a_rows;
	const int size_result_cols = size_b_cols;
	float result[size_result_rows][size_result_cols];
	// result should be
	// { 140.0, 146.0 }
	// { 320.0, 335.0 }

	cudaError cudaStatus = doCudaOperation((float*)a, (float*)b, (float*)result, 
											size_a_rows, size_a_cols, 
											size_b_rows, size_b_cols, 
											size_result_rows, size_result_cols
											);

	//input
	std::cout << "input : " << std::endl;
	for (int i = 0; i < size_a_rows; ++i)
	{
		for (int j = 0; j < size_a_cols; ++j)
		{
			std::cout << a[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "input : " << std::endl;
	for (int i = 0; i < size_b_rows; ++i)
	{
		for (int j = 0; j < size_b_cols; ++j)
		{
			std::cout << b[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	//result
	std::cout << "result : " << std::endl;
	for (int i = 0; i < size_result_rows; ++i)
	{
		for (int j = 0; j < size_result_cols; ++j)
		{
			std::cout << result[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cin.get();
	return 0;
}

