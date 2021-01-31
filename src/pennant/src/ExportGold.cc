/*
 * ExportGold.cc
 *
 *  Created on: Mar 1, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "ExportGold.hh"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>

#include "Vec2.hh"
#include "Mesh.hh"

using namespace std;


ExportGold::ExportGold(Mesh* m) : mesh(m) {}

ExportGold::~ExportGold() {}


void ExportGold::write(
        const string& basename,
        const int cycle,
        const double time,
        const double* zr,
        const double* ze,
        const double* zp) {

    writeCaseFile(basename);

    sortZones();
    writeGeoFile(basename, cycle, time);

    writeVarFile(basename, "zr", zr);
    writeVarFile(basename, "ze", ze);
    writeVarFile(basename, "zp", zp);

}


void ExportGold::writeCaseFile(
        const string& basename) {

    // open file
    const string filename = basename + ".case";
    ofstream ofs(filename.c_str());
    if (!ofs.good()) {
        cerr << "Cannot open file " << filename << " for writing"
             << endl;
        exit(1);
    }

    ofs << "#" << endl;
    ofs << "# Created by PENNANT" << endl;
    ofs << "#" << endl;

    ofs << "FORMAT" << endl;
    ofs << "type: ensight gold" << endl;

    ofs << "GEOMETRY" << endl;
    ofs << "model: " << basename << ".geo" << endl;

    ofs << "VARIABLE" << endl;
    ofs << "scalar per element: zr " << basename << ".zr" << endl;
    ofs << "scalar per element: ze " << basename << ".ze" << endl;
    ofs << "scalar per element: zp " << basename << ".zp" << endl;

    ofs.close();

}


void ExportGold::writeGeoFile(
        const string& basename,
        const int cycle,
        const double time) {

    // open file
    const string filename = basename + ".geo";
    ofstream ofs(filename.c_str());
    if (!ofs.good()) {
        cerr << "Cannot open file " << filename << " for writing"
             << endl;
        exit(1);
    }

    // write general header
    ofs << scientific;
    ofs << "cycle = " << setw(8) << cycle << endl;
    ofs << setprecision(8);
    ofs << "t = " << setw(15) << time << endl;
    ofs << "node id assign" << endl;
    ofs << "element id given" << endl;

    // write header for the one "part" (entire mesh)
    ofs << "part" << endl;
    ofs << setw(10) << 1 << endl;
    ofs << "universe" << endl;

    const int nump = mesh->nump;
    const double2* px = mesh->px;

    // write node info
    ofs << "coordinates" << endl;
    ofs << setw(10) << nump << endl;
    ofs << setprecision(5);
    for (int p = 0; p < nump; ++p)
        ofs << setw(12) << px[p].x << endl;
    for (int p = 0; p < nump; ++p)
        ofs << setw(12) << px[p].y << endl;
    // Ensight expects z-coordinates, so write 0 for those
    for (int p = 0; p < nump; ++p)
        ofs << setw(12) << 0. << endl;

    const int* znump = mesh->znump;
    const int* mapsp1 = mesh->mapsp1;

    int ntris = tris.size();
    int nquads = quads.size();
    int nothers = others.size();

    // write triangles
    if (ntris > 0) {
        ofs << "tria3" << endl;
        ofs << setw(10) << ntris << endl;
        for (int i = 0; i < ntris; ++i)
            ofs << setw(10) << tris[i] + 1 << endl;
        for (int i = 0; i < ntris; ++i) {
            int z = tris[i];
            int sbase = mapzs[z];
            for (int s = sbase; s < sbase + 3; ++s)
                ofs << setw(10) << mapsp1[s] + 1;
            ofs << endl;
        }
   } // if ntris > 0

    // write quads
    if (nquads > 0) {
        ofs << "quad4" << endl;
        ofs << setw(10) << nquads << endl;
        for (int i = 0; i < nquads; ++i)
            ofs << setw(10) << quads[i] + 1 << endl;
        for (int i = 0; i < nquads; ++i) {
            int z = quads[i];
            int sbase = mapzs[z];
            for (int s = sbase; s < sbase + 4; ++s)
                ofs << setw(10) << mapsp1[s] + 1;
            ofs << endl;
        }
   } // if nquads > 0

    // write others
    if (nothers > 0) {
        ofs << "nsided" << endl;
        ofs << setw(10) << nothers << endl;
        for (int i = 0; i < nothers; ++i)
            ofs << setw(10) << others[i] + 1 << endl;
        for (int i = 0; i < nothers; ++i) {
            int z = others[i];
            ofs << setw(10) << znump[z] << endl;
        }
        for (int i = 0; i < nothers; ++i) {
            int z = others[i];
            int sbase = mapzs[z];
            for (int s = sbase; s < sbase + znump[z]; ++s)
                ofs << setw(10) << mapsp1[s] + 1;
            ofs << endl;
        }
   } // if nothers > 0

    ofs.close();

}


void ExportGold::writeVarFile(
        const string& basename,
        const string& varname,
        const double* var) {

    // open file
    const string filename = basename + "." + varname;
    ofstream ofs(filename.c_str());
    if (!ofs.good()) {
        cerr << "Cannot open file " << filename << " for writing"
             << endl;
        exit(1);
    }

    ofs << varname << endl;
    ofs << "part" << endl;
    ofs << setw(10) << 1 << endl;

    int ntris = tris.size();
    int nquads = quads.size();
    int nothers = others.size();

    ofs << scientific << setprecision(5);

    // write values on triangles
    if (ntris > 0) {
        ofs << "tria3" << endl;
        for (int i = 0; i < ntris; ++i) {
            int z = tris[i];
            ofs << setw(12) << var[z] << endl;
        }
   } // if ntris > 0

    // write values on quads
    if (nquads > 0) {
        ofs << "quad4" << endl;
        for (int i = 0; i < nquads; ++i) {
            int z = quads[i];
            ofs << setw(12) << var[z] << endl;
        }
   } // if nquads > 0

    // write values on others
    if (nothers > 0) {
        ofs << "nsided" << endl;
        for (int i = 0; i < nothers; ++i) {
            int z = others[i];
            ofs << setw(12) << var[z] << endl;
        }
   } // if nothers > 0

    ofs.close();

}


void ExportGold::sortZones() {

    const int numz = mesh->numz;
    const int* znump = mesh->znump;

    mapzs.resize(numz);

    // sort zones by size, create an inverse map
    int scount = 0;
    for (int z = 0; z < numz; ++z) {
        int zsize = znump[z];
        if (zsize == 3)
            tris.push_back(z);
        else if (zsize == 4)
            quads.push_back(z);
        else // zsize > 4
            others.push_back(z);
        mapzs[z] = scount;
        scount += zsize;
    } // for z

}
