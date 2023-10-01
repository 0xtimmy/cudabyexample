#ifndef __BOOK_H__
#define __BOOK_H__

void *big_random_block ( int size, int max ) {
    unsigned char *data = (unsigned char*)malloc( size );
    for (int i = 0; i < size; i++) {
        data[i] = rand() % max;
    }

    return data;
}

#endif