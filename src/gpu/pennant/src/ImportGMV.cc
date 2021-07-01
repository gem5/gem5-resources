/*
 * ImportGMV.cc
 *
 *  Created on: Feb 27, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "ImportGMV.hh"

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include "Vec2.hh"
#include "Mesh.hh"

using namespace std;


namespace {  // unnamed
    void errorExit(const string& msg) {
        cerr << "ImportGMV error: " << msg << endl;
        exit(1);
    }
}


ImportGMV::ImportGMV(Mesh* m) : mesh(m) {}


ImportGMV::~ImportGMV() {}


// NOTES ON: FORMAT OF GMV FILE
// This code currently only handles files with nf = 1 for
// all "cells" entries
//----------------------------------------------------------------
//gmvinput ascii
//
//nodes nn
//  x... y... z...
//
//cells nc
//  general nf          ---> general 4
//    nverts(nf)        ---> 3 3 3 3
//    vert(sum_nverts)  ---> 1 3 2  1 3 4  1 4 2  2 3 4
//  ...
//
// <lots of other things that aren't used on input>
//
//endgmv


void ImportGMV::read(
        const string& filename,
        vector<double2>& nodepos,
        vector<int>& cellstart,
        vector<int>& cellsize,
        vector<int>& cellnodes) {

    // open file
    ifstream ifs(filename.c_str());
    if (!ifs.good()) {
        errorExit("Cannot open file " + filename + " for reading");
    }

    // check gmv header
    string line;
    getline(ifs, line);
    if (!ifs.good()) {
        errorExit("Bad format for GMV header");
    }
    if (line != "gmvinput ascii") {
        errorExit("File " + filename + " is not a GMV ascii file");
    }

    // read number of nodes
    string nodelabel;
    int nnodes;
    ifs >> nodelabel >> nnodes;
    if (!ifs.good() || nodelabel != "nodes") {
        errorExit("Wrong format for number of nodes");
    }

    nodepos.resize(nnodes);

    // read x and y values for nodes
    for (int n = 0; n < nnodes; ++n) {
        ifs >> nodepos[n].x;
    }
    for (int n = 0; n < nnodes; ++n) {
        ifs >> nodepos[n].y;
    }
    double dum;
    // skip over z values for nodes (unused in 2D)
    for (int n = 0; n < nnodes; ++n) {
        ifs >> dum;
    }
    // check for format errors in the above
    if (!ifs.good()) {
        errorExit("Wrong format for node coordinates");
    }

    // read number of cells
    string celllabel;
    int ncells;
    ifs >> celllabel >> ncells;
    if (!ifs.good() || celllabel != "cells") {
        errorExit("Wrong format for number of cells");
    }

    cellstart.resize(ncells);
    cellsize.resize(ncells);
    cellnodes.reserve(3 * ncells);
    int maxcsize = 0;

    // read node lists for cells
    for (int c = 0; c < ncells; ++c) {
        string typelabel;
        int ntmp;
        ifs >> typelabel >> ntmp;
        // for now we only know how to read a single "general"
        // cell at a time
        if (!ifs.good() || typelabel != "general" || ntmp != 1) {
            errorExit("Wrong format for list of cells");
        }
        int csize;
        ifs >> csize;
        if (!ifs.good() || csize <= 0) {
            errorExit("Wrong format for list of cells");
        }
        maxcsize = max(maxcsize, csize);
        cellstart[c] = cellnodes.size();
        cellsize[c] = csize;
        for (int n = 0; n < csize; ++n)    {
            int ntmp;
            ifs >> ntmp;
            // convert node number from 1- to 0-based
            cellnodes.push_back(ntmp - 1);
        }
        // check for format errors in the above
        if (!ifs.good()) {
            errorExit("Wrong format for list of cells");
        }
    }

    ifs.close();

}

