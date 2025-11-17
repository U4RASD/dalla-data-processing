/*********************************************************************
 * Copyright (c) 2011-2015 Jan Pomikalek                             *
 * All rights reserved.                                              *
 *                                                                   *
 * This software is licensed as described in the file COPYING, which *
 * you should have received as part of this distribution.            *
 *********************************************************************/

#ifndef BUZHASH_H
#define BUZHASH_H
#define BUZHASH_MAX 18446744073709551615ul

#include <stdint.h>

typedef uint64_t hash_t;
typedef struct {
    int size;
    hash_t *elems;
    int elem_count;     // current number of elements in the buffer
    int last_index;     // the index of the last element (buffer is circular)
    hash_t hash;        // current hash value
} buzhash_buffer_t;

hash_t hash_string(char* string);
void buzhash_init_buffer(buzhash_buffer_t* buffer, int size);
void buzhash_clear_buffer(buzhash_buffer_t* buffer);
void buzhash_free_buffer(buzhash_buffer_t* buffer);
int buzhash_is_full_buffer(buzhash_buffer_t* buffer);
hash_t buzhash(char* string, buzhash_buffer_t* buffer);
#endif
