/*********************************************************************
 * Copyright (c) 2011-2015 Jan Pomikalek                             *
 * All rights reserved.                                              *
 *                                                                   *
 * This software is licensed as described in the file COPYING, which *
 * you should have received as part of this distribution.            *
 *********************************************************************/

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "buzhash.h"
#include "version.h"

#define OUTPUT_FILE "duphashes"

// options
char *Output_file = OUTPUT_FILE;
int Quiet = 0;
FILE* Input;
long int Input_size;

void print_usage(FILE *stream) {
    fprintf(stream, "\
Usage: hashdup [OPTIONS] FILE [FILE...]\n\
Identify duplicate hashes.\n\
\n\
 -o FILE   output file (default: %s)\n\
 -q        quiet; suppress all output except for errors\n\
\n\
 -V        print version information and exit\n\
 -h        display this help and exit\n\
\n\
Project home page: <http://code.google.com/p/onion/>\n",
        OUTPUT_FILE);
}

// taken from http://cs.wikipedia.org/wiki/Quicksort
void quicksort(hash_t array[], long int left_begin, long int right_begin) {
    hash_t pm = array[(left_begin + right_begin) / 2];
    long int left_index, right_index;
    left_index = left_begin;
    right_index = right_begin;
    do {
        while (array[left_index] < pm)
            left_index++;
        while (array[right_index] > pm)
            right_index--;
        if (left_index <= right_index) {
            hash_t value = array[left_index];
            array[left_index] = array[right_index];
            array[right_index] = value;
            left_index++;
            right_index--;
        }
    } while (left_index < right_index);
    if (right_index > left_begin)
        quicksort(array, left_begin, right_index);
    if (left_index < right_begin)
        quicksort(array, left_index, right_begin);
}

void print_progress(int processed_files, int total_files) {
    time_t now;
    time(&now);
    fprintf(stderr, "[%.24s] hashdup: %i / %i files processed\n", ctime(&now),
            processed_files, total_files);
}

int main(int argc, char **argv) {
    // get options
    int c;
    while ((c = getopt(argc, argv, "o:qVh")) != -1) {
        errno = 0;
        switch (c) {
            case 'o':
                Output_file = optarg;
                break;
            case 'q':
                Quiet = 1;
                break;
            case 'V':
                print_version("hashdup");
                return 0;
            case 'h':
                print_usage(stdout);
                return 0;
            case '?':
                print_usage(stderr);
                return 1;
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "No input.\n");
        print_usage(stderr);
        return 1;
    }

    // output file
    errno = 0;
    FILE* output_fp = fopen(Output_file, "w");
    if (errno != 0) {
        fprintf(stderr, "Unable to open %s for writing.\n", Output_file);
        return 1;
    }

    int input_files_count = argc - optind;

    // for all input files
    int i;
    for (i=optind; i<argc; i++) {
        // open file
        char* filename = argv[i];
        int input_fd = open(filename, O_RDONLY);
        if (input_fd == -1) {
            fprintf(stderr, "Unable to open %s for reading.\n", filename);
            return 1;
        }

        // determine file size
        unsigned long int file_size = lseek(input_fd, 0L, SEEK_END);
        lseek(input_fd, 0L, SEEK_SET);

        // map hashes into memory
        hash_t* hashes = NULL;
        hashes = (hash_t*) mmap(hashes, file_size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE, input_fd, 0);

        // sort hashes
        unsigned long int hash_count = file_size / sizeof(hash_t);
        quicksort(hashes, 0, hash_count-1);

        // send duplicate hashes to the output
        int written = 0;
        hash_t prev_hash = hashes[0];
        hash_t hash;
        unsigned long int j;
        for (j=1; j<hash_count; j++) {
            hash = hashes[j];
            if (hash == prev_hash) {
                if (!written) {
                    fwrite(&hash, sizeof(hash), 1, output_fp);
                    written = 1;
                }
            }
            else {
                written = 0;
            }
            prev_hash = hash;
        }

        munmap(hashes, file_size);
        close(input_fd);

        // print progress information
        if (!Quiet)
            print_progress(i - optind + 1, input_files_count);
    }

    fclose(output_fp);

    return 0;
}
