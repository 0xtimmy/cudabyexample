#include<stdio.h>
int main (void) {

    cudaDeviceProp prop;

    int count;
    cudaGetDeviceCount( &count );

    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties( &prop, i);
        printf("\n\n");
        printf("--- General Information for Device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock Rate: %d\n", prop.clockRate);
        printf("\n");
        printf("--- Memory Information for Device %d ---\n", i);
        printf("Total Global Memory: %ld\n", prop.totalGlobalMem);
        printf("total Constant Memory: %ld\n", prop.totalConstMem);
        printf("Max mem pitch: %ld\n", prop.memPitch);
        printf("\n");
        printf("--- Microprocessor Information for Device %d ---\n", i);
        printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n\n");
    }
    return 0;
}