/*
 * Hydro.hh
 *
 *  Created on: Dec 22, 2011
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef HYDRO_HH_
#define HYDRO_HH_

#include <string>
#include <vector>

#include "Vec2.hh"

// forward declarations
class InputFile;
class Mesh;
class PolyGas;
class TTS;
class QCS;
class HydroBC;


class Hydro {
public:

    // associated mesh object
    Mesh* mesh;

    // children of this object
    PolyGas* pgas;
    TTS* tts;
    QCS* qcs;
    std::vector<HydroBC*> bcs;

    double cfl;                 // Courant number, limits timestep
    double cflv;                // volume change limit for timestep
    double rinit;               // initial density for main mesh
    double einit;               // initial energy for main mesh
    double rinitsub;            // initial density in subregion
    double einitsub;            // initial energy in subregion
    double uinitradial;         // initial velocity in radial direction
    std::vector<double> bcx;    // x values of x-plane fixed boundaries
    std::vector<double> bcy;    // y values of y-plane fixed boundaries

    double dtrec;               // maximum timestep for hydro
    std::string msgdtrec;       // message:  reason for dtrec

    double2* pu;       // point velocity
    double2* pu0;      // point velocity, start of cycle
    double2* pap;      // point acceleration
    double2* pf;       // point force
    double* pmaswt;    // point mass, weighted by 1/r
    double* smaswt;    // side contribution to pmaswt

    double* zm;        // zone mass
    double* zr;        // zone density
    double* zrp;       // zone density, middle of cycle
    double* ze;        // zone specific internal energy
                       // (energy per unit mass)
    double* zetot;     // zone total internal energy
    double* zetot0;    // zetot at start of cycle
    double* zw;        // zone work done in cycle
    double* zwrate;    // zone work rate
    double* zp;        // zone pressure
    double* zss;       // zone sound speed
    double* zdu;       // zone velocity difference

    double2* sf;       // side force (from pressure)
    double2* sfq;      // side force from artificial visc.
    double2* sft;      // side force from tts

    Hydro(const InputFile* inp, Mesh* m);
    ~Hydro();

    void init();

    void getData();

    void initRadialVel(const double vel);

    void doCycle(const double dt);

    void getDtHydro(
            double& dtnew,
            std::string& msgdtnew);

}; // class Hydro



#endif /* HYDRO_HH_ */
