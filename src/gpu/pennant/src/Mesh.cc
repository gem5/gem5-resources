/*
 * Mesh.cc
 *
 *  Created on: Jan 5, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Mesh.hh"

#include <cmath>
#include <iostream>
#include <algorithm>

#include "Vec2.hh"
#include "Memory.hh"
#include "InputFile.hh"
#include "GenMesh.hh"
#include "WriteXY.hh"
#include "ExportGold.hh"

using namespace std;


Mesh::Mesh(const InputFile* inp) :
        gmesh(NULL), egold(NULL), wxy(NULL) {

    chunksize = inp->getInt("chunksize", 0);
    if (chunksize < 0) {
        cerr << "Error: bad chunksize " << chunksize << endl;
    }
    subregion = inp->getDoubleList("subregion", vector<double>());
    if (subregion.size() != 0 && subregion.size() != 4) {
        cerr << "Error:  subregion must have 4 entries" << endl;
        exit(1);
    }

    gmesh = new GenMesh(inp);
    wxy = new WriteXY(this);
    egold = new ExportGold(this);

    init();
}


Mesh::~Mesh() {
    delete gmesh;
    delete wxy;
    delete egold;
}


void Mesh::init() {

    // read mesh from gmv file
    vector<double2> nodepos;
    vector<int> cellstart, cellsize, cellnodes;
    vector<int> masterpes, mastercounts, slavepoints;
    vector<int> slavepes, slavecounts, masterpoints;
    gmesh->generate(nodepos, cellstart, cellsize, cellnodes,
            masterpes, mastercounts, slavepoints,
            slavepes, slavecounts, masterpoints);

    nump = nodepos.size();
    numz = cellstart.size();
    nums = cellnodes.size();

    // copy node positions to mesh, apply scaling factor
    px = Memory::alloc<double2>(nump);
    copy(nodepos.begin(), nodepos.end(), px);

    // copy cell sizes to mesh
    znump = Memory::alloc<int>(numz);
    copy(cellsize.begin(), cellsize.end(), znump);

    // populate maps:
    // use the cell* arrays to populate the side maps
    initSides(cellstart, cellsize, cellnodes);
    // release memory from cell* arrays
    cellstart.resize(0);
    cellsize.resize(0);
    cellnodes.resize(0);
    // now populate other maps using side maps
    initEdges();
    initCorners();

    // populate chunk information
    initChunks();

    // write mesh statistics
    writeStats();

    // allocate remaining arrays
    ex = Memory::alloc<double2>(nume);
    zx = Memory::alloc<double2>(numz);
    sarea = Memory::alloc<double>(nums);
    svol = Memory::alloc<double>(nums);
    carea = Memory::alloc<double>(numc);
    cvol = Memory::alloc<double>(numc);
    zarea = Memory::alloc<double>(numz);
    zvol = Memory::alloc<double>(numz);
    smf = Memory::alloc<double>(nums);

    // do a few initial calculations
    #pragma omp parallel for
    for (int ch = 0; ch < numsch; ++ch) {
        int sfirst = schsfirst[ch];
        int slast = schslast[ch];
        calcCtrs(px, ex, zx, sfirst, slast);
        calcVols(px, zx, sarea, svol, carea, cvol, zarea, zvol,
                sfirst, slast);
        calcSideFracs(sarea, zarea, smf, sfirst, slast);
    }

}


void Mesh::initSides(
        const std::vector<int>& cellstart,
        const std::vector<int>& cellsize,
        const std::vector<int>& cellnodes) {

    mapsp1 = Memory::alloc<int>(nums);
    mapsp2 = Memory::alloc<int>(nums);
    mapsz  = Memory::alloc<int>(nums);
    mapss3 = Memory::alloc<int>(nums);
    mapss4 = Memory::alloc<int>(nums);

    for (int z = 0; z < numz; ++z) {
        int sbase = cellstart[z];
        int size = cellsize[z];
        for (int n = 0; n < size; ++n) {
            int s = sbase + n;
            int snext = sbase + (n + 1 == size ? 0 : n + 1);
            int slast = sbase + (n == 0 ? size : n) - 1;
            mapsz[s] = z;
            mapsp1[s] = cellnodes[s];
            mapsp2[s] = cellnodes[snext];
            mapss3[s] = slast;
            mapss4[s] = snext;
        } // for n
    } // for z

}


void Mesh::initEdges() {

    vector<vector<int> > edgepp(nump), edgepe(nump);

    mapse = Memory::alloc<int>(nums);
    // nums = upper bound for number of edges
    mapep1 = Memory::alloc<int>(nums);
    mapep2 = Memory::alloc<int>(nums);

    int e = 0;
    for (int s = 0; s < nums; ++s) {
        int p1 = min(mapsp1[s], mapsp2[s]);
        int p2 = max(mapsp1[s], mapsp2[s]);

        vector<int>& vpp = edgepp[p1];
        vector<int>& vpe = edgepe[p1];
        int i = find(vpp.begin(), vpp.end(), p2) - vpp.begin();
        if (i == vpp.size()) {
            // (p, p2) isn't in the edge list - add it
            vpp.push_back(p2);
            vpe.push_back(e);
            mapep1[e] = p1;
            mapep2[e] = p2;
            ++e;
        }
        mapse[s] = vpe[i];
    }  // for s

    nume = e;

}


void Mesh::initCorners() {

    numc = nums;

    mapcz = Memory::alloc<int>(numc);
    mapcp = Memory::alloc<int>(numc);
    mapsc1 = Memory::alloc<int>(nums);
    mapsc2 = Memory::alloc<int>(nums);

    for (int s = 0; s < nums; ++s) {
        int c = s;
        int c2 = mapss4[s];
        mapsc1[s] = c;
        mapsc2[s] = c2;
        mapcz[c] = mapsz[s];
        mapcp[c] = mapsp1[s];
    }

}


void Mesh::initChunks() {

    // check for bad chunksize
    if (chunksize <= 0) {
        cerr << "Error: bad chunksize " << chunksize << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

    // compute side chunks
    // use 'chunksize' for maximum chunksize; decrease as needed
    // to ensure that no zone has its sides split across chunk
    // boundaries
    int s1, s2 = 0;
    while (s2 < nums) {
        s1 = s2;
        s2 = min(s2 + chunksize, nums);
        while (s2 < nums && mapsz[s2] == mapsz[s2-1])
            --s2;
        schsfirst.push_back(s1);
        schslast.push_back(s2);
        schzfirst.push_back(mapsz[s1]);
        schzlast.push_back(mapsz[s2-1] + 1);
    }
    numsch = schsfirst.size();

    // compute point chunks
    int p1, p2 = 0;
    while (p2 < nump) {
        p1 = p2;
        p2 = min(p2 + chunksize, nump);
        pchpfirst.push_back(p1);
        pchplast.push_back(p2);
    }
    numpch = pchpfirst.size();

}


void Mesh::writeStats() {
    cout << "--- Mesh Information ---" << endl;
    cout << "Points:  " << nump << endl;
    cout << "Zones:  "  << numz << endl;
    cout << "Sides:  "  << nums << endl;
    cout << "Edges:  "  << nume << endl;
    cout << "Side chunks:  " << numsch << endl;
    cout << "Point chunks:  " << numpch << endl;
    cout << "Chunk size:  " << chunksize << endl;
    cout << "------------------------" << endl;

}


void Mesh::write(
        const string& probname,
        const int cycle,
        const double time,
        const double* zr,
        const double* ze,
        const double* zp) {

    wxy->write(probname, zr, ze, zp);
    egold->write(probname, cycle, time, zr, ze, zp);

}


void Mesh::calcCtrs(
        const double2* px,
        double2* ex,
        double2* zx,
        const int sfirst,
        const int slast) {

    int zfirst = mapsz[sfirst];
    int zlast = (slast < nums ? mapsz[slast] : numz);
    fill(&zx[zfirst], &zx[zlast], double2(0., 0.));

    for (int s = sfirst; s < slast; ++s) {
        int p1 = mapsp1[s];
        int p2 = mapsp2[s];
        int e = mapse[s];
        int z = mapsz[s];
        ex[e] = 0.5 * (px[p1] + px[p2]);
        zx[z] += px[p1] / (double) znump[z];
    }

}


void Mesh::calcVols(
        const double2* px,
        const double2* zx,
        double* sarea,
        double* svol,
        double* carea,
        double* cvol,
        double* zarea,
        double* zvol,
        const int sfirst,
        const int slast) {

    int cfirst = sfirst;
    int clast = slast;
    int zfirst = mapsz[sfirst];
    int zlast = (slast < nums ? mapsz[slast] : numz);
    fill(&cvol[cfirst], &cvol[clast], 0.);
    fill(&carea[cfirst], &carea[clast], 0.);
    fill(&zvol[zfirst], &zvol[zlast], 0.);
    fill(&zarea[zfirst], &zarea[zlast], 0.);

    int nserr = 0;

    for (int s = sfirst; s < slast; ++s) {
        int p1 = mapsp1[s];
        int p2 = mapsp2[s];
        int z = mapsz[s];

        // compute side volumes, sum to zone
        double sa = 0.5 * cross(px[p2] - px[p1], zx[z] - px[p1]);
        double sv = sa * (px[p1].x + px[p2].x + zx[z].x) / 3.;
        sarea[s] = sa;
        svol[s] = sv;
        zarea[z] += sa;
        zvol[z] += sv;

        // check for negative side volumes
        if (sv <= 0.) nserr += 1;

        int c1 = mapsc1[s];
        int c2 = mapsc2[s];

        // sum side volumes to corners
        double hsa = 0.5 * sa;
        double ex = 0.5 * (px[p1].x + px[p2].x);
        double hsv1 = hsa * (px[p1].x + zx[z].x + ex) / 3.;
        double hsv2 = hsa * (px[p2].x + zx[z].x + ex) / 3.;
        carea[c1] += hsa;
        carea[c2] += hsa;
        cvol[c1] += hsv1;
        cvol[c2] += hsv2;

    } // for s

    // if there were negative side volumes, error exit
    if (nserr > 0) {
        cerr << "Error: " << nserr << " negative side volumes" << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

}


void Mesh::calcSideFracs(
        const double* sarea,
        const double* zarea,
        double* smf,
        const int sfirst,
        const int slast) {

    for (int s = sfirst; s < slast; ++s) {
        int z = mapsz[s];
        smf[s] = sarea[s] / zarea[z];
    }
}


