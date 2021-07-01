/*
 * Hydro.cc
 *
 *  Created on: Dec 22, 2011
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Hydro.hh"

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include "Memory.hh"
#include "InputFile.hh"
#include "Mesh.hh"
#include "PolyGas.hh"
#include "TTS.hh"
#include "QCS.hh"
#include "HydroBC.hh"
#include "HydroGPU.hh"

using namespace std;


Hydro::Hydro(const InputFile* inp, Mesh* m) : mesh(m) {
    cfl = inp->getDouble("cfl", 0.6);
    cflv = inp->getDouble("cflv", 0.1);
    rinit = inp->getDouble("rinit", 1.);
    einit = inp->getDouble("einit", 0.);
    rinitsub = inp->getDouble("rinitsub", 1.);
    einitsub = inp->getDouble("einitsub", 0.);
    uinitradial = inp->getDouble("uinitradial", 0.);
    bcx = inp->getDoubleList("bcx", vector<double>());
    bcy = inp->getDoubleList("bcy", vector<double>());

    pgas = new PolyGas(inp, this);
    tts = new TTS(inp, this);
    qcs = new QCS(inp, this);

    init();

    hydroInitGPU();
    hydroInit(
            mesh->nump, mesh->numz, mesh->nums, mesh->numc, mesh->nume,
            pgas->gamma, pgas->ssmin,
            tts->alfa, tts->ssmin,
            qcs->qgamma, qcs->q1, qcs->q2,
            cfl, cflv,
            bcx.size(), &bcx[0], bcy.size(), &bcy[0],
            mesh->px,
            pu,
            zm,
            zr,
            mesh->zvol,
            ze, zetot,
            zwrate,
            mesh->smf,
            mesh->mapsp1,
            mesh->mapsp2,
            mesh->mapsz,
            mesh->mapss4,
            mesh->mapse,
            mesh->znump);

}


Hydro::~Hydro() {

    hydroFinalGPU();

    delete tts;
    delete qcs;
    for (int i = 0; i < bcs.size(); ++i) {
        delete bcs[i];
    }
}


void Hydro::init() {

    dtrec = 1.e99;
    msgdtrec = "Hydro default";

    const int nump = mesh->nump;
    const int numz = mesh->numz;

    const double2* zx = mesh->zx;
    const double* zvol = mesh->zvol;

    // allocate arrays
    pu = Memory::alloc<double2>(nump);
    zm = Memory::alloc<double>(numz);
    zr = Memory::alloc<double>(numz);
    ze = Memory::alloc<double>(numz);
    zetot = Memory::alloc<double>(numz);
    zwrate = Memory::alloc<double>(numz);
    zp = Memory::alloc<double>(numz);

    // initialize hydro vars
    fill(&zr[0], &zr[numz], rinit);
    fill(&ze[0], &ze[numz], einit);
    fill(&zwrate[0], &zwrate[numz], 0.);

    const vector<double>& subrgn = mesh->subregion;
    if (!subrgn.empty()) {
        const double eps = 1.e-12;
        for (int z = 0; z < numz; ++z) {
            if (zx[z].x > (subrgn[0] - eps) &&
                zx[z].x < (subrgn[1] + eps) &&
                zx[z].y > (subrgn[2] - eps) &&
                zx[z].y < (subrgn[3] + eps)) {
                zr[z] = rinitsub;
                ze[z] = einitsub;
            }
        }
    }

    for (int z = 0; z < numz; ++z) {
        zm[z] = zr[z] * zvol[z];
        zetot[z] = ze[z] * zm[z];
    }

    if (uinitradial != 0.)
        initRadialVel(uinitradial);
    else
        fill(&pu[0], &pu[nump], double2(0., 0.));

}


void Hydro::getData() {

    hydroGetData(
            mesh->nump, mesh->numz,
            mesh->px,
            zr, ze, zp);

}


void Hydro::initRadialVel(const double vel) {
    const int nump = mesh->nump;
    const double2* px = mesh->px;
    const double eps = 1.e-12;

    for (int p = 0; p < nump; ++p) {
        double pmag = length(px[p]);
        if (pmag > eps)
            pu[p] = vel * px[p] / pmag;
        else
            pu[p] = double2(0., 0.);
    }
}


void Hydro::doCycle(
            const double dt) {

    int idtrec;

    hydroDoCycle(dt, dtrec, idtrec);

    int z = idtrec >> 1;
    bool dtfromvol = idtrec & 1;
    ostringstream oss;
    if (dtfromvol)
        oss << "Hydro dV/V limit for z = " << setw(6) << z;
    else
        oss << "Hydro Courant limit for z = " << setw(6) << z;
    msgdtrec = oss.str();

}


void Hydro::getDtHydro(
        double& dtnew,
        string& msgdtnew) {

    if (dtrec < dtnew) {
        dtnew = dtrec;
        msgdtnew = msgdtrec;
    }

}


