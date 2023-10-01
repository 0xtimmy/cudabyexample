#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__

#include<stdio.h>

struct CPUBitmap {
    unsigned char *pixels;
    int x, y;
    void *dataBlock;
    void (*bitmapExit)(void*);

    CPUBitmap( int width, int height, void *d = NULL) {
        pixels = new unsigned char[width * height * 4];
        x = width;
        y = height;
        dataBlock = d;
    }

    ~CPUBitmap() {
        delete [] pixels;
    }

    unsigned char* get_ptr( void ) const { return pixels; };
    long image_size( void ) const { return x * y * 4; }; 

    void display_and_exit( void ) {
        CPUBitmap** bitmap = get_bitmap_ptr();
        *bitmap = this;

        FILE* ptr = fopen("img.ppm", "w");
        fprintf(ptr, "P3\n%d %d\n255\n", x, y);
        for (int py = 0; py < y; py++) {
            for (int px = 0; px < x; px++) {
                fprintf(
                    ptr,
                    "%d %d %d\n",
                    pixels[(py*x + px)*4 + 0],
                    pixels[(py*x + px)*4 + 1],
                    pixels[(py*x + px)*4 + 2]
                );
            }
        }
        fclose(ptr);
        return;
    }

    static CPUBitmap** get_bitmap_ptr( void ) {
        static CPUBitmap *gBitmap;
        return &gBitmap;
    }

    /*
    static void Key(unsigned chat key, int x, int y) {
        switch(key) {
            case 27:
            CPUBitmap* bitmap = *(get_bitmap_ptr());
            glClearColor( 0.0, 0.0, 0.0, 1.0 );
            glClear( GL_COLOR_BUFFER_BIT );
            glDrawPixels( bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels );
            glFlush();
        }
    }

    static void Draw( void ) {
        CPUBitmap* bitmap = *(get_bitmap_ptr());
        glClearColor( 0.0, 0.0, 0.0, 1.0 );
        glClear();
        glDrawPixels(bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
    }
    */
};

#endif