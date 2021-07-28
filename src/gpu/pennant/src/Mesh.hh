/*
 * Mesh.hh
 *
 *  Created on: Jan 5, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef MESH_HH_
#define MESH_HH_

#include <string>
#include <vector>

#include "Vec2.hh"

// forward declarations
class InputFile;
class GenMesh;
class WriteXY;
class ExportGold;


class Mesh {
public:

    // children
    GenMesh* gmesh;
    WriteXY* wxy;
    ExportGold* egold;

    // parameters
    int chunksize;                 // max size for processing chunks
    std::vector<double> subregion; // bounding box for a subregion
                                   // if nonempty, should have 4 entries:
                                   // xmin, xmax, ymin, ymax

    // mesh variables
    // (See documentation for more details on the mesh
    //  data structures...)
    int nump, nume, numz, nums, numc;
                                   // number of points, edges, zones,
                                   // sides, corners, resp.
    int* mapcz;        // map: corner -> zone
    int* mapcp;        // map: corner -> point
    int* mapep1;       // maps: edge -> points 1 and 2
    int* mapep2;
    int* mapsp1;       // maps: side -> points 1 and 2
    int* mapsp2;
    int* mapsc1;       // maps: side -> corners 1 and 2
    int* mapsc2;
    int* mapsz;        // map: side -> zone
    int* mapse;        // map: side -> edge
    int* mapss3;       // map: side -> previous side
    int* mapss4;       // map: side -> next side

    int* znump;        // number of points in zone

    double2* px;       // point coordinates
    double2* ex;       // edge center coordinates
    double2* zx;       // zone center coordinates
    double2* pxp;      // point coords, middle of cycle
    double2* exp;      // edge ctr coords, middle of cycle
    double2* zxp;      // zone ctr coords, middle of cycle
    double2* px0;      // point coords, start of cycle

    double* sarea;     // side area
    double* svol;      // side volume
    double* carea;     // corner area
    double* cvol;      // corner volume
    double* zarea;     // zone area
    double* zvol;      // zone volume
    double* sareap;    // side area, middle of cycle
    double* svolp;     // side volume, middle of cycle
    double* careap;    // corner area, middle of cycle
    double* cvolp;     // corner volume, middle of cycle
    double* zareap;    // zone area, middle of cycle
    double* zvolp;     // zone volume, middle of cycle
    double* zvol0;     // zone volume, start of cycle

    double2* ssurfp;   // side surface vector
    double* elen;      // edge length
    double* smf;       // side mass fraction
    double* zdl;       // zone characteristic length

    int numsch;                    // number of side chunks
    std::vector<int> schsfirst;    // start/stop index for side chunks
    std::vector<int> schslast;
    std::vector<int> schzfirst;    // start/stop index for zone chunks
    std::vector<int> schzlast;
    int numpch;                    // number of point chunks
    std::vector<int> pchpfirst;    // start/stop index for point chunks
    std::vector<int> pchplast;

    Mesh(const InputFile* inp);
    ~Mesh();

    void init();

    // populate mapping arrays
    void initSides(
            const std::vector<int>& cellstart,
            const std::vector<int>& cellsize,
            const std::vector<int>& cellnodes);
    void initEdges();
    void initCorners();

    // populate chunk information
    void initChunks();

    // write mesh statistics
    void writeStats();

    // write mesh
    void write(
            const std::string& probname,
            const int cycle,
            const double time,
            const double* zr,
            const double* ze,
            const double* zp);

    // compute edge, zone centers
    void calcCtrs(
            const double2* px,
            double2* ex,
            double2* zx,
            const int sfirst,
            const int slast);

    // compute side, corner, zone volumes
    void calcVols(
            const double2* px,
            const double2* zx,
            double* sarea,
            double* svol,
            double* carea,
            double* cvol,
            double* zarea,
            double* zvol,
            const int sfirst,
            const int slast);

    // compute side mass fractions
    void calcSideFracs(
            const double* sarea,
            const double* zarea,
            double* smf,
            const int sfirst,
            const int slast);

}; // class Mesh



#endif /* MESH_HH_ */
