
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>

__global__ void parallelThreadOperation(float* a, float* b, float* debug, float* sum, float* prod)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	//__shared__ float shared_array[4];

	//printf("gpu  gridDim=%d blockDim=%d warpSize=%d blockIdx.x=%d threadIdx.x=%d i=%d\n", gridDim, blockDim, warpSize, blockIdx.x, threadIdx.x, i);
	//printf("gpu  %d %d %d %d %d %d\n", gridDim, blockDim, warpSize, blockIdx.x, threadIdx.x, i);
	//printf("gpu  %d %d %d\n", blockIdx.x, threadIdx.x, i);
	//printf("%d %d %d \n", gridDim.x, gridDim.y, gridDim.z);
	//printf("gpu %d", i);
	int x = blockIdx.y * blockDim.x + threadIdx.x;
	debug[x] = x;

	//add operation in block 0
	if (blockIdx.y == 0)
		sum[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
	//multiply operation in block 1
	else if (blockIdx.y == 1)
		prod[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

	//__syncthreads;
}

void showCudaProp()
{
	cudaDeviceProp* device_attrs = new cudaDeviceProp();
	cudaError cudaStatus = cudaGetDeviceProperties(device_attrs, 0);
	std::cout << "name        : " << device_attrs->name << std::endl;
	std::cout << "maxGridSize : "
		<< "x = " << device_attrs->maxGridSize[0] << " "
		<< "y = " << device_attrs->maxGridSize[1] << " "
		<< "z = " << device_attrs->maxGridSize[2] << " "
		<< std::endl;
	std::cout << "maxThreadsPerBlock : " << device_attrs->maxThreadsPerBlock << std::endl;
	std::cout << "maxThreadsDim : "
		<< "x = " << device_attrs->maxThreadsDim[0] << " "
		<< "y = " << device_attrs->maxThreadsDim[1] << " "
		<< "z = " << device_attrs->maxThreadsDim[2] << " "
		<< std::endl;
	delete device_attrs;
}

void showCudaPrintSize()
{
	size_t size;
	cudaDeviceGetLimit(&size, cudaLimitPrintfFifoSize);
	printf("Printf size found to be %d\n", (int)size);

}

cudaError doCudaOperation(float *a, float *b, int size, float *debug, float *sum, float* prod)
{
	//vars for cuda mem space
	float* cuda_a;
	float* cuda_b;
	float* cuda_debug;
	float* cuda_sum;
	float* cuda_prod;
	cudaError cudaStatus = cudaSuccess;
		
	//set device
	cudaStatus = cudaSetDevice(0);

	//std::cout << sizeof(a) << std::endl;
	//std::cout << size * sizeof(float) << std::endl;

	//allocate memory in cuda 
	cudaStatus = cudaMalloc((void**)&cuda_a, size*sizeof(float));
	cudaStatus = cudaMalloc((void**)&cuda_b, size*sizeof(float));
	cudaStatus = cudaMalloc((void**)&cuda_debug, 2*size*sizeof(float));
	cudaStatus = cudaMalloc((void**)&cuda_sum, size*sizeof(float));
	cudaStatus = cudaMalloc((void**)&cuda_prod, size*sizeof(float));

	//copy data from host to device
	cudaStatus = cudaMemcpy(cuda_a, a, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(cuda_b, b, size*sizeof(float), cudaMemcpyHostToDevice);

	//grid & block dims
	dim3 n_blocks(1,2,1);		// grid size -> grid dim
	dim3 n_threads(size,1,1);	// block size -> block dim, max 1024
	
	parallelThreadOperation <<<n_blocks, n_threads>>> (cuda_a, cuda_b, cuda_debug, cuda_sum, cuda_prod);
	cudaStatus = cudaGetLastError();
	cudaStatus = cudaDeviceSynchronize();

	//copy result back to host
	cudaStatus = cudaMemcpy(debug, cuda_debug, 2 * size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(sum, cuda_sum, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(prod, cuda_prod, size*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_debug);
	cudaFree(cuda_sum);
	cudaFree(cuda_prod);

	return cudaStatus;
}



int main()
{
	showCudaProp();
	showCudaPrintSize();

	const int size = 5;
	float a[] = { 1, 2, 3, 5.5, 6 };
	float b[] = { 1.1, 2.2, 3.3, 4.4, 5.5 };
	float debug[2*size];
	float sum[size];
	float prod[size];

	cudaError cudaStatus = doCudaOperation(a, b, size, debug, sum, prod);

	//input
	std::cout << "input : " << std::endl;
	for (int i = 0; i < size; ++i)
		std::cout << a[i] << " ";
	std::cout << std::endl;
	for (int i = 0; i < size; ++i)
		std::cout << b[i] << " ";
	std::cout << std::endl;

	//debug
	std::cout << "debug : ";
	for(int i=0; i<2*size; ++i)
		std::cout << debug[i] << " ";
	std::cout << std::endl;

	//sum
	std::cout << "sum : ";
	for (int i = 0; i < size; ++i)
		std::cout << sum[i] << " ";
	std::cout << std::endl;

	//prod
	std::cout << "prod : ";
	for (int i = 0; i < size; ++i)
		std::cout << prod[i] << " ";
	std::cout << std::endl;

	std::cin.get();
	return 0;
}

