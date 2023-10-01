#include<stdio.h>
#include<sys/time.h>
#include "../common/book.h"
#define SIZE (100*1024*1024)

__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo);

int main( void ) {
    unsigned char *buffer = (unsigned char*)big_random_block(SIZE, 256);

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0);

    unsigned char *dev_buffer;
    unsigned int *dev_histo;

    cudaMalloc((void**)&dev_buffer, SIZE);
    cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_histo, 256*sizeof(long));
    cudaMemset(dev_histo, 0, 256 * sizeof(int));

    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0 );
    int blocks = prop.multiProcessorCount;
    histo_kernel<<<blocks*2,256>>>(dev_buffer, SIZE, dev_histo);

    unsigned int histo[256];
    cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    

    timeval cpustop, cpustart;
    gettimeofday(&cpustart, NULL);
    for(int i = 0; i < SIZE; i++) {
        histo[buffer[i]]--;
    }
    gettimeofday(&cpustop, NULL);
    for (int i = 0; i < 256; i++) {
        if (histo[i] > 0) {
            printf("Failure! @ i = %d\n", i);
            break;
        }
    }

    printf("CPU took: %3.1fms\n", (float)(cpustop.tv_sec - cpustart.tv_sec) * 1000.0 + (float)(cpustop.tv_usec - cpustart.tv_usec) / 1000.0);
    printf("GPU took: %3.1fms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_histo);
    cudaFree(dev_buffer);
    free(buffer);

    return 0;
}

__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo) {
    
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    while(i < size) {
        atomicAdd( &temp[buffer[i]], 1);
        i += offset;
    }

    __syncthreads();

    atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x]);
}