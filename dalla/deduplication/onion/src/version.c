/*********************************************************************
 * Copyright (c) 2011-2016 Jan Pomikalek, Milos Jakubicek            *
 * All rights reserved.                                              *
 *                                                                   *
 * This software is licensed as described in the file COPYING, which *
 * you should have received as part of this distribution.            *
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "version.h"

void print_version(const char* progname) {
    printf("%s: onion v%s\n\n", progname, VERSION);
    printf("Copyright (c) 2011-2020 Lexical Computing Limited and Lexical Computing CZ s.r.o.\n");
}
