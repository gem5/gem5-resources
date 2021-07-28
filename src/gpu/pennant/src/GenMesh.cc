/*
 * GenMesh.cc
 *
 *  Created on: Jun 4, 2013
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "GenMesh.hh"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "Vec2.hh"
#include "Parallel.hh"
#include "InputFile.hh"

using namespace std;


GenMesh::GenMesh(const InputFile* inp) {
    meshtype = inp->getString("meshtype", "");
    if (meshtype.empty()) {
        cerr << "Error:  must specify meshtype" << endl;
        exit(1);
    }
    if (meshtype != "pie" &&
            meshtype != "rect" &&
            meshtype != "hex") {
        cerr << "Error:  invalid meshtype " << meshtype << endl;
        exit(1);
    }
    vector<double> params =
            inp->getDoubleList("meshparams", vector<double>());
    if (params.empty()) {
        cerr << "Error:  must specify meshparams" << endl;
        exit(1);
    }
    if (params.size() > 4) {
        cerr << "Error:  meshparams must have <= 4 values" << endl;
        exit(1);
    }

    gnzx = params[0];
    gnzy = (params.size() >= 2 ? params[1] : gnzx);
    if (meshtype != "pie")
        lenx = (params.size() >= 3 ? params[2] : 1.0);
    else
        // convention:  x = theta, y = r
        lenx = (params.size() >= 3 ? params[2] : 90.0)
                * M_PI / 180.0;
    leny = (params.size() >= 4 ? params[3] : 1.0);

    if (gnzx <= 0 || gnzy <= 0 || lenx <= 0. || leny <= 0. ) {
        cerr << "Error:  meshparams values must be positive" << endl;
        exit(1);
    }
    if (meshtype == "pie" && lenx >= 2. * M_PI) {
        cerr << "Error:  meshparams theta must be < 360" << endl;
        exit(1);
    }

}


GenMesh::~GenMesh() {}


void GenMesh::generate(
        std::vector<double2>& pointpos,
        std::vector<int>& zonestart,
        std::vector<int>& zonesize,
        std::vector<int>& zonepoints,
        std::vector<int>& masterpes,
        std::vector<int>& mastercounts,
        std::vector<int>& slavepoints,
        std::vector<int>& slavepes,
        std::vector<int>& slavecounts,
        std::vector<int>& masterpoints){

    // do calculations common to all mesh types
    calcNumPE();
    zxoffset = mypex * gnzx / numpex;
    const int zxstop = (mypex + 1) * gnzx / numpex;
    nzx = zxstop - zxoffset;
    zyoffset = mypey * gnzy / numpey;
    const int zystop = (mypey + 1) * gnzy / numpey;
    nzy = zystop - zyoffset;

    // mesh type-specific calculations
    if (meshtype == "pie")
        generatePie(pointpos, zonestart, zonesize, zonepoints,
                masterpes, mastercounts, slavepoints,
                slavepes, slavecounts, masterpoints);
    else if (meshtype == "rect")
        generateRect(pointpos, zonestart, zonesize, zonepoints,
                masterpes, mastercounts, slavepoints,
                slavepes, slavecounts, masterpoints);
    else if (meshtype == "hex")
        generateHex(pointpos, zonestart, zonesize, zonepoints,
                masterpes, mastercounts, slavepoints,
                slavepes, slavecounts, masterpoints);

}


void GenMesh::generateRect(
        std::vector<double2>& pointpos,
        std::vector<int>& zonestart,
        std::vector<int>& zonesize,
        std::vector<int>& zonepoints,
        std::vector<int>& masterpes,
        std::vector<int>& mastercounts,
        std::vector<int>& slavepoints,
        std::vector<int>& slavepes,
        std::vector<int>& slavecounts,
        std::vector<int>& masterpoints) {

    using Parallel::numpe;
    using Parallel::mype;

    const int nz = nzx * nzy;
    const int npx = nzx + 1;
    const int npy = nzy + 1;
    const int np = npx * npy;

    // generate point coordinates
    pointpos.reserve(np);
    for (int j = 0; j < npy; ++j) {
        double y = (leny * (double) (j + zyoffset) / (double) gnzy);
        for (int i = 0; i < npx; ++i) {
            double x = (lenx * (double) (i + zxoffset) / (double) gnzx);
            pointpos.push_back(make_double2(x, y));
        }
    }

    // generate zone adjacency lists
    zonestart.reserve(nz);
    zonesize.reserve(nz);
    zonepoints.reserve(4 * nz);
    for (int j = 0; j < nzy; ++j) {
        for (int i = 0; i < nzx; ++i) {
            zonestart.push_back(zonepoints.size());
            zonesize.push_back(4);
            int p0 = j * npx + i;
            zonepoints.push_back(p0);
            zonepoints.push_back(p0 + 1);
            zonepoints.push_back(p0 + npx + 1);
            zonepoints.push_back(p0 + npx);
       }
    }

    if (numpe == 1) return;

    // estimate sizes of slave/master arrays
    slavepoints.reserve((mypey != 0) * npx + (mypex != 0) * npy);
    masterpoints.reserve((mypey != numpey - 1) * npx +
            (mypex != numpex - 1) * npy + 1);

    // enumerate slave points
    // slave point with master at lower left
    if (mypex != 0 && mypey != 0) {
        int mstrpe = mype - numpex - 1;
        slavepoints.push_back(0);
        masterpes.push_back(mstrpe);
        mastercounts.push_back(1);
    }
    // slave points with master below
    if (mypey != 0) {
        int mstrpe = mype - numpex;
        int oldsize = slavepoints.size();
        int p = 0;
        for (int i = 0; i < npx; ++i) {
            if (i == 0 && mypex != 0) { p++; continue; }
            slavepoints.push_back(p);
            p++;
        }
        masterpes.push_back(mstrpe);
        mastercounts.push_back(slavepoints.size() - oldsize);
    }
    // slave points with master to left
    if (mypex != 0) {
        int mstrpe = mype - 1;
        int oldsize = slavepoints.size();
        int p = 0;
        for (int j = 0; j < npy; ++j) {
            if (j == 0 && mypey != 0) { p += npx; continue; }
            slavepoints.push_back(p);
            p += npx;
        }
        masterpes.push_back(mstrpe);
        mastercounts.push_back(slavepoints.size() - oldsize);
    }

    // enumerate master points
    // master points with slave to right
    if (mypex != numpex - 1) {
        int slvpe = mype + 1;
        int oldsize = masterpoints.size();
        int p = npx - 1;
        for (int j = 0; j < npy; ++j) {
            if (j == 0 && mypey != 0) { p += npx; continue; }
            masterpoints.push_back(p);
            p += npx;
        }
        slavepes.push_back(slvpe);
        slavecounts.push_back(masterpoints.size() - oldsize);
    }
    // master points with slave above
    if (mypey != numpey - 1) {
        int slvpe = mype + numpex;
        int oldsize = masterpoints.size();
        int p = (npy - 1) * npx;
        for (int i = 0; i < npx; ++i) {
            if (i == 0 && mypex != 0) { p++; continue; }
            masterpoints.push_back(p);
            p++;
        }
        slavepes.push_back(slvpe);
        slavecounts.push_back(masterpoints.size() - oldsize);
    }
    // master point with slave at upper right
    if (mypex != numpex - 1 && mypey != numpey - 1) {
        int slvpe = mype + numpex + 1;
        int p = npx * npy - 1;
        masterpoints.push_back(p);
        slavepes.push_back(slvpe);
        slavecounts.push_back(1);
    }

}


void GenMesh::generatePie(
        std::vector<double2>& pointpos,
        std::vector<int>& zonestart,
        std::vector<int>& zonesize,
        std::vector<int>& zonepoints,
        std::vector<int>& masterpes,
        std::vector<int>& mastercounts,
        std::vector<int>& slavepoints,
        std::vector<int>& slavepes,
        std::vector<int>& slavecounts,
        std::vector<int>& masterpoints) {

    using Parallel::numpe;
    using Parallel::mype;

    const int nz = nzx * nzy;
    const int npx = nzx + 1;
    const int npy = nzy + 1;
    const int np = (mypey == 0 ? npx * (npy - 1) + 1 : npx * npy);

    // generate point coordinates
    pointpos.reserve(np);
    for (int j = 0; j < npy; ++j) {
        if (j + zyoffset == 0) {
            pointpos.push_back(make_double2(0., 0.));
            continue;
        }
        double r = (leny * (double) (j + zyoffset) / (double) gnzy);
        for (int i = 0; i < npx; ++i) {
            double th = (lenx * (double) (gnzx - (i + zxoffset)) /
                    (double) gnzx);
            double x = r * cos(th);
            double y = r * sin(th);
            pointpos.push_back(make_double2(x, y));
        }
    }

    // generate zone adjacency lists
    zonestart.reserve(nz);
    zonesize.reserve(nz);
    zonepoints.reserve(4 * nz);
    for (int j = 0; j < nzy; ++j) {
        for (int i = 0; i < nzx; ++i) {
            zonestart.push_back(zonepoints.size());
            int p0 = j * npx + i;
            if (mypey == 0) p0 -= npx - 1;
            if (j + zyoffset == 0) {
                zonesize.push_back(3);
                zonepoints.push_back(0);
            }
            else {
                zonesize.push_back(4);
                zonepoints.push_back(p0);
                zonepoints.push_back(p0 + 1);
            }
            zonepoints.push_back(p0 + npx + 1);
            zonepoints.push_back(p0 + npx);
        }
    }

    if (numpe == 1) return;

    // estimate sizes of slave/master arrays
    slavepoints.reserve((mypey != 0) * npx + (mypex != 0) * npy);
    masterpoints.reserve((mypey != numpey - 1) * npx +
            (mypex != numpex - 1) * npy + 1);

    // enumerate slave points
    // slave point with master at lower left
    if (mypex != 0 && mypey != 0) {
        int mstrpe = mype - numpex - 1;
        slavepoints.push_back(0);
        masterpes.push_back(mstrpe);
        mastercounts.push_back(1);
    }
    // slave points with master below
    if (mypey != 0) {
        int mstrpe = mype - numpex;
        int oldsize = slavepoints.size();
        int p = 0;
        for (int i = 0; i < npx; ++i) {
            if (i == 0 && mypex != 0) { p++; continue; }
            slavepoints.push_back(p);
            p++;
        }
        masterpes.push_back(mstrpe);
        mastercounts.push_back(slavepoints.size() - oldsize);
    }
    // slave points with master to left
    if (mypex != 0) {
        int mstrpe = mype - 1;
        int oldsize = slavepoints.size();
        if (mypey == 0) {
            slavepoints.push_back(0);
            // special case:
            // slave point at origin, master not to immediate left
            if (mypex > 1) {
                masterpes.push_back(0);
                mastercounts.push_back(1);
                oldsize += 1;
            }
        }
        int p = (mypey > 0 ? npx : 1);
        for (int j = 1; j < npy; ++j) {
            slavepoints.push_back(p);
            p += npx;
        }
        masterpes.push_back(mstrpe);
        mastercounts.push_back(slavepoints.size() - oldsize);
    }

    // enumerate master points
    // master points with slave to right
    if (mypex != numpex - 1) {
        int slvpe = mype + 1;
        int oldsize = masterpoints.size();
        // special case:  origin as master for slave on PE 1
        if (mypex == 0 && mypey == 0) {
            masterpoints.push_back(0);
        }
        int p = (mypey > 0 ? 2 * npx - 1 : npx);
        for (int j = 1; j < npy; ++j) {
            masterpoints.push_back(p);
            p += npx;
        }
        slavepes.push_back(slvpe);
        slavecounts.push_back(masterpoints.size() - oldsize);
        // special case:  origin as master for slaves on PEs > 1
        if (mypex == 0 && mypey == 0) {
            for (int slvpe = 2; slvpe < numpex; ++slvpe) {
                masterpoints.push_back(0);
                slavepes.push_back(slvpe);
                slavecounts.push_back(1);
            }
        }
    }
    // master points with slave above
    if (mypey != numpey - 1) {
        int slvpe = mype + numpex;
        int oldsize = masterpoints.size();
        int p = (npy - 1) * npx;
        if (mypey == 0) p -= npx - 1;
        for (int i = 0; i < npx; ++i) {
            if (i == 0 && mypex != 0) { p++; continue; }
            masterpoints.push_back(p);
            p++;
        }
        slavepes.push_back(slvpe);
        slavecounts.push_back(masterpoints.size() - oldsize);
    }
    // master point with slave at upper right
    if (mypex != numpex - 1 && mypey != numpey - 1) {
        int slvpe = mype + numpex + 1;
        int p = npx * npy - 1;
        if (mypey == 0) p -= npx - 1;
        masterpoints.push_back(p);
        slavepes.push_back(slvpe);
        slavecounts.push_back(1);
    }

}


void GenMesh::generateHex(
        std::vector<double2>& pointpos,
        std::vector<int>& zonestart,
        std::vector<int>& zonesize,
        std::vector<int>& zonepoints,
        std::vector<int>& masterpes,
        std::vector<int>& mastercounts,
        std::vector<int>& slavepoints,
        std::vector<int>& slavepes,
        std::vector<int>& slavecounts,
        std::vector<int>& masterpoints) {

    using Parallel::numpe;
    using Parallel::mype;

    const int nz = nzx * nzy;
    const int npx = nzx + 1;
    const int npy = nzy + 1;

    // generate point coordinates
    pointpos.resize(2 * npx * npy);  // overestimate
    double dx = lenx / (gnzx - 1);
    double dy = leny / (gnzy - 1);

    vector<int> pbase(npy);
    int p = 0;
    for (int j = 0; j < npy; ++j) {
        pbase[j] = p;
        int gj = j + zyoffset;
        for (int i = 0; i < npx; ++i) {
            int gi = i + zxoffset;
            double x = dx * ((double) gi - 0.5);
            double y = dy * ((double) gj - 0.5);
            if (gi == 0) x = 0.;
            else if (gi == gnzx) x = lenx;
            if (gj == 0) y = 0.;
            else if (gj == gnzy) y = leny;
            if (gi == 0 || gi == gnzx || gj == 0 || gj == gnzy)
                pointpos[p++] = make_double2(x, y);
            else if (i == nzx && j == 0)
                pointpos[p++] = make_double2(x - dx / 6., y + dy / 6.);
            else if (i == 0 && j == nzy)
                pointpos[p++] = make_double2(x + dx / 6., y - dy / 6.);
            else {
                pointpos[p++] = make_double2(x - dx / 6., y + dy / 6.);
                pointpos[p++] = make_double2(x + dx / 6., y - dy / 6.);
            }
        } // for i
    } // for j
    int np = p;
    pointpos.resize(np);

    // generate zone adjacency lists
    zonestart.resize(nz);
    zonesize.resize(nz);
    for (int j = 0; j < nzy; ++j) {
        int gj = j + zyoffset;
        int pbasel = pbase[j];
        int pbaseh = pbase[j+1];
        if (mypex > 0) {
            if (gj > 0) pbasel += 1;
            if (j < nzy - 1) pbaseh += 1;
        }
        for (int i = 0; i < nzx; ++i) {
            int gi = i + zxoffset;
            int z = j * nzx + i;
            vector<int> v(6);
            v[1] = pbasel + 2 * i;
            v[0] = v[1] - 1;
            v[2] = v[1] + 1;
            v[5] = pbaseh + 2 * i;
            v[4] = v[5] + 1;
            v[3] = v[4] + 1;
            if (gj == 0) {
                v[0] = pbasel + i;
                v[2] = v[0] + 1;
                if (gi == gnzx - 1) v.erase(v.begin()+3);
                v.erase(v.begin()+1);
            } // if j
            else if (gj == gnzy - 1) {
                v[5] = pbaseh + i;
                v[3] = v[5] + 1;
                v.erase(v.begin()+4);
                if (gi == 0) v.erase(v.begin()+0);
            } // else if j
            else if (gi == 0)
                v.erase(v.begin()+0);
            else if (gi == gnzx - 1)
                v.erase(v.begin()+3);
            zonestart[z] = zonepoints.size();
            zonesize[z] = v.size();
            zonepoints.insert(zonepoints.end(), v.begin(), v.end());
        } // for i
    } // for j

    if (numpe == 1) return;

    // estimate sizes of slave/master arrays
    slavepoints.reserve((mypey != 0) * 2 * npx +
            (mypex != 0) * 2 * npy);
    masterpoints.reserve((mypey != numpey - 1) * 2 * npx +
            (mypex != numpex - 1) * 2 * npy + 2);

    // enumerate slave points
    // slave points with master at lower left
    if (mypex != 0 && mypey != 0) {
        int mstrpe = mype - numpex - 1;
        slavepoints.push_back(0);
        slavepoints.push_back(1);
        masterpes.push_back(mstrpe);
        mastercounts.push_back(2);
    }
    // slave points with master below
    if (mypey != 0) {
        int p = 0;
        int mstrpe = mype - numpex;
        int oldsize = slavepoints.size();
        for (int i = 0; i < npx; ++i) {
            if (i == 0 && mypex != 0) {
                p += 2;
                continue;
            }
            if (i == 0 || i == nzx)
                slavepoints.push_back(p++);
            else {
                slavepoints.push_back(p++);
                slavepoints.push_back(p++);
            }
        }  // for i
        masterpes.push_back(mstrpe);
        mastercounts.push_back(slavepoints.size() - oldsize);
    }  // if mypey != 0
    // slave points with master to left
    if (mypex != 0) {
        int mstrpe = mype - 1;
        int oldsize = slavepoints.size();
        for (int j = 0; j < npy; ++j) {
            if (j == 0 && mypey != 0) continue;
            int p = pbase[j];
            if (j == 0 || j == nzy)
                slavepoints.push_back(p++);
            else {
                slavepoints.push_back(p++);
                slavepoints.push_back(p++);
           }
        }  // for j
        masterpes.push_back(mstrpe);
        mastercounts.push_back(slavepoints.size() - oldsize);
    }  // if mypex != 0

    // enumerate master points
    // master points with slave to right
    if (mypex != numpex - 1) {
        int slvpe = mype + 1;
        int oldsize = masterpoints.size();
        for (int j = 0; j < npy; ++j) {
            if (j == 0 && mypey != 0) continue;
            int p = (j == nzy ? np : pbase[j+1]);
            if (j == 0 || j == nzy)
                masterpoints.push_back(p-1);
            else {
                masterpoints.push_back(p-2);
                masterpoints.push_back(p-1);
           }
        }
        slavepes.push_back(slvpe);
        slavecounts.push_back(masterpoints.size() - oldsize);
    }  // if mypex != numpex - 1
    // master points with slave above
    if (mypey != numpey - 1) {
        int p = pbase[nzy];
        int slvpe = mype + numpex;
        int oldsize = masterpoints.size();
        for (int i = 0; i < npx; ++i) {
            if (i == 0 && mypex != 0) {
                p++;
                continue;
            }
            if (i == 0 || i == nzx)
                masterpoints.push_back(p++);
            else {
                masterpoints.push_back(p++);
                masterpoints.push_back(p++);
            }
        }  // for i
        slavepes.push_back(slvpe);
        slavecounts.push_back(masterpoints.size() - oldsize);
    }  // if mypey != numpey - 1
    // master points with slave at upper right
    if (mypex != numpex - 1 && mypey != numpey - 1) {
        int slvpe = mype + numpex + 1;
        masterpoints.push_back(np-2);
        masterpoints.push_back(np-1);
        slavepes.push_back(slvpe);
        slavecounts.push_back(2);
    }

}


void GenMesh::calcNumPE() {

    using Parallel::numpe;
    using Parallel::mype;

    // pick numpex, numpey such that PE blocks are as close to square
    // as possible
    // we would like:  gnzx / numpex == gnzy / numpey,
    // where numpex * numpey = numpe (total number of PEs available)
    // this solves to:  numpex = sqrt(numpe * gnzx / gnzy)
    // we compute this, assuming gnzx <= gnzy (swap if necessary)
    double nx = static_cast<double>(gnzx);
    double ny = static_cast<double>(gnzy);
    bool swapflag = (nx > ny);
    if (swapflag) swap(nx, ny);
    double n = sqrt(numpe * nx / ny);
    // need to constrain n to be an integer with numpe % n == 0
    // try rounding n both up and down
    int n1 = floor(n + 1.e-12);
    n1 = max(n1, 1);
    while (numpe % n1 != 0) --n1;
    int n2 = ceil(n - 1.e-12);
    while (numpe % n2 != 0) ++n2;
    // pick whichever of n1 and n2 gives blocks closest to square,
    // i.e. gives the shortest long side
    double longside1 = max(nx / n1, ny / (numpe/n1));
    double longside2 = max(nx / n2, ny / (numpe/n2));
    numpex = (longside1 <= longside2 ? n1 : n2);
    numpey = numpe / numpex;
    if (swapflag) swap(numpex, numpey);
    mypex = mype % numpex;
    mypey = mype / numpex;

}

