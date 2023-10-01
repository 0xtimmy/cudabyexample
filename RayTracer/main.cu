#include "../common/cpu_bitmap.h"

#define rnd(x) (x*rand() / RAND_MAX)
#define SPHERES 50
#define DIM 500
#define INF 2e10f

__global__ void kernel(unsigned char *ptr);

struct Sphere {
    float r,b,g;
    float radius;
    float x,y,z;

    __device__ float hit (float ox, float oy, float *n) {
        float dx = ox - x;
        float dy = oy - y;

        if(dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf(radius*radius - dx*dx - dy*dy);
            *n = dz /sqrtf(radius*radius);
            return dz + z;
        }

        return -INF;
    }
};

__constant__ Sphere s[SPHERES];

int main( void ) {
    cudaEvent_t start, stop;

    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    CPUBitmap bitmap( DIM, DIM );
    unsigned char *dev_bitmap;

    cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd( 1.0f );
        temp_s[i].g = rnd( 1.0f );
        temp_s[i].b = rnd( 1.0f );
        temp_s[i].x = rnd( 1000.0f ) - 500.0;
        temp_s[i].y = rnd( 1000.0f ) - 500.0;
        temp_s[i].z = rnd( 1000.0f ) - 500.0;
        temp_s[i].radius = rnd( 100.0f ) + 20.0;
    }

    cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES);

    free(temp_s);

    dim3 grids(DIM/16,DIM/16);
    dim3 threads(16,16);

    cudaEventRecord( start, 0 );

    kernel<<<grids,threads>>>(dev_bitmap);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop);
    printf("Time to generate: %3.1f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

    bitmap.display_and_exit();

    cudaFree(dev_bitmap);
}

__global__ void kernel(unsigned char *ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x < DIM && y < DIM) {
        int offset = x + y * DIM;

        float ox = (x - DIM/2);
        float oy = (y - DIM/2);

        float r=0, g=0, b=0;
        float maxz = -INF;
        for(int i = 0; i < SPHERES; i++) {
            float n;
            float t = s[i].hit(ox, oy, &n);
            if (t > maxz) {
                float fscale = n;
                r = s[i].r * fscale;
                g = s[i].g * fscale;
                b = s[i].b * fscale;
            }
        }

        //if(ox > 0) r = 1;
        //if(oy > 0) b = 1;

        ptr[offset*4+0] = (int)(r * 255);
        ptr[offset*4+1] = (int)(g * 255);
        ptr[offset*4+2] = (int)(b * 255);
        ptr[offset*4+3] = 255;
    }
}