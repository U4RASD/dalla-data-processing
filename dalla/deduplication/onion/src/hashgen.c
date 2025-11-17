/*********************************************************************
 * Copyright (c) 2011-2015 Jan Pomikalek                             *
 * All rights reserved.                                              *
 *                                                                   *
 * This software is licensed as described in the file COPYING, which *
 * you should have received as part of this distribution.            *
 *********************************************************************/

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "buzhash.h"
#include "version.h"

#define MAX_LINE_LENGTH 10000
#define NGRAM_SIZE 5
#define OUTPUT_PREFIX "hashes."
#define OUTPUT_COUNT 10

// options
int Ngram_size = NGRAM_SIZE;
char *Output_prefix = OUTPUT_PREFIX;
int Output_count = OUTPUT_COUNT;
int Quiet = 0;
FILE* Input;
long int Input_size;

void print_usage(FILE *stream) {
    fprintf(stream, "\
Usage: hashgen [OPTIONS] [FILE]\n\
Generate hashes of n-grams.\n\
\n\
 -n NUM    n-gram length (default: %i)\n\
 -o STR    prefix of output files (default: %s)\n\
 -c NUM    number of output files (default: %i)\n\
 -q        quiet; suppress all output except for errors\n\
\n\
 -V        print version information and exit\n\
 -h        display this help and exit\n\
\n\
With no FILE, or when FILE is -, read standard input.\n\
\n\
Project home page: <http://code.google.com/p/onion/>\n",
        NGRAM_SIZE, OUTPUT_PREFIX, OUTPUT_COUNT);
}

void print_progress(unsigned long int processed_bytes, float percent_done) {
    time_t now;
    time(&now);
    fprintf(stderr, "[%.24s] hashgen: %6li MB processed", ctime(&now),
            processed_bytes / (1024 * 1024));
    if (percent_done >= 0)
        fprintf(stderr, " (%6.2f%%)", percent_done);
    fprintf(stderr, "\n");
}

int main(int argc, char **argv) {
    // get options
    int c;
    char *endptr;
    while ((c = getopt(argc, argv, "n:o:c:qVh")) != -1) {
        errno = 0;
        switch (c) {
            case 'n':
                Ngram_size = strtol(optarg, &endptr, 10);
                if (errno != 0 || *endptr != '\0') {
                    fprintf(stderr, "Integer value expected for -n, got: %s\n", optarg);
                    print_usage(stderr);
                    return 1;
                }
                break;
            case 'o':
                Output_prefix = optarg;
                break;
            case 'c':
                Output_count = strtol(optarg, &endptr, 10);
                if (errno != 0 || *endptr != '\0') {
                    fprintf(stderr, "Integer value expected for -c, got: %s\n", optarg);
                    print_usage(stderr);
                    return 1;
                }
                break;
            case 'q':
                Quiet = 1;
                break;
            case 'V':
                print_version("hashgen");
                return 0;
            case 'h':
                print_usage(stdout);
                return 0;
            case '?':
                print_usage(stderr);
                return 1;
        }
    }

    Input = stdin;
    Input_size = -1;
    if (optind < argc) {
        char* filename = argv[optind];
        if (strcmp(filename, "-") != 0) {
            errno = 0;
            Input = fopen(filename, "r");
            if (errno != 0) {
                fprintf(stderr, "Unable to open %s for reading.\n", filename);
                return 1;
            }
            fseek(Input, 0L, SEEK_END);
            Input_size = ftell(Input);
            fseek(Input, 0L, SEEK_SET);
        }
    }

    // output files
    FILE** output_files = (FILE**) malloc(Output_count * sizeof(FILE*));
    char* filename = (char*) malloc(
            (strlen(Output_prefix) + (Output_count/10+1) + 1) * sizeof(char));
    int i;
    for (i=0; i<Output_count; i++) {
        sprintf(filename, "%s%i", Output_prefix, i);
        errno = 0;
        output_files[i] = fopen(filename, "w");
        if (errno != 0) {
            fprintf(stderr, "Unable to open %s for writing.\n", filename);
            return 1;
        }
    }
    free(filename);

    // hash range boundaries
    hash_t* range_boundaries = (hash_t*) malloc(Output_count * sizeof(hash_t));
    hash_t range_size = BUZHASH_MAX / Output_count;
    for (i=0; i<Output_count-1; i++)
        range_boundaries[i] = (i+1) * range_size;
    range_boundaries[Output_count-1] = BUZHASH_MAX;

    // buzhash
    hash_t hash;
    buzhash_buffer_t bh_buffer;
    buzhash_init_buffer(&bh_buffer, Ngram_size);

    // other variables
    char line[MAX_LINE_LENGTH];
    unsigned long int line_number = 0;
    unsigned long int processed_bytes = 0;

    while (fgets(line, MAX_LINE_LENGTH, Input)) {
        // read line and strip trailing newline
        line_number++;
        int linelen = strlen(line);
        char* newline_pointer = strchr(line, '\n');
        if (newline_pointer == NULL) {
            if (linelen >= MAX_LINE_LENGTH - 1)
                fprintf(stderr, "Warning: line %li too long; "
                        "processing only first %i chars.\n", line_number,
                        linelen);
            else
                fprintf(stderr, "Warning: line %li contains a NUL character; "
                        "processing only the first %i chars.\n", line_number,
                        linelen);
        }
        else {
            *newline_pointer = '\0';
        }
        processed_bytes+= linelen;

        // skip lines starting with <
        if (line[0] == '<')
            continue;

        // compute hash
        hash = buzhash(line, &bh_buffer);
        if (!buzhash_is_full_buffer(&bh_buffer))
            continue;

        // store hash in the correct file
        int range_index = 0;
        while (hash > range_boundaries[range_index])
            range_index++;
        fwrite(&hash, sizeof(hash), 1, output_files[range_index]);

        // print progress information
        if (!Quiet && line_number % 10000000 == 0) {
            float percent_done = -1;
            if (Input_size > 0)
                percent_done = 100.0 * processed_bytes / Input_size;
            print_progress(processed_bytes, percent_done);
        }
    }

    // print progress information
    if (!Quiet)
        print_progress(processed_bytes, 100);

    for (i=0; i<Output_count; i++)
        fclose(output_files[i]);

    if (Input != stdin)
        fclose(Input);

    return 0;
}
