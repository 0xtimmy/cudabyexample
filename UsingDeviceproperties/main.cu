#include<stdio.h>

int main( void ) {
    cudaDeviceProp prop;
    int dev;

    cudaGetDevice( &dev );
    printf("Id of current device: %d\n", dev);

    memset ( &prop, 0, sizeof( cudaDeviceProp ));
    prop.major = 8;
    prop.minor = 7;
    cudaChooseDevice(&dev, &prop);
    printf("Id of CUDA device closest to revision %d.%d: %d\n", prop.major, prop.minor, dev);
    cudaSetDevice(dev);
}