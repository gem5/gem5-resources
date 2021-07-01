/*
 * main.cc
 *
 *  Created on: Jan 23, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include <cstdlib>
#include <iostream>

#include "InputFile.hh"
#include "Driver.hh"

using namespace std;


int main(const int argc, const char** argv)
{
    if (argc != 2) {
       cerr << "Usage: pennant <filename>" << endl;
       exit(1);
    }

    const char* filename = argv[1];
    InputFile inp(filename);

    string probname(filename);
    // strip .pnt suffix from filename
    int len = probname.length();
    if (probname.substr(len - 4, 4) == ".pnt")
        probname = probname.substr(0, len - 4);

    Driver drv(&inp, probname);

    drv.run();

    return 0;

}



