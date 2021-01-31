/*
 * HydroGPU.hh
 *
 *  Created on: Aug 2, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef HYDROGPU_HH_
#define HYDROGPU_HH_

#include "Vec2.hh"


void hydroInit(
        const int numpH,
        const int numzH,
        const int numsH,
        const int numcH,
        const int numeH,
        const double pgammaH,
        const double pssminH,
        const double talfaH,
        const double tssminH,
        const double qgammaH,
        const double q1H,
        const double q2H,
        const double hcflH,
        const double hcflvH,
        const int numbcxH,
        const double* bcxH,
        const int numbcyH,
        const double* bcyH,
        const double2* pxH,
        const double2* puH,
        const double* zmH,
        const double* zrH,
        const double* zvolH,
        const double* zeH,
        const double* zetotH,
        const double* zwrateH,
        const double* smfH,
        const int* mapsp1H,
        const int* mapsp2H,
        const int* mapszH,
        const int* mapss4H,
        const int* mapseH,
        const int* znumpH);

void hydroDoCycle(
        const double dtH,
        double& dtnextH,
        int& idtnextH);

void hydroGetData(
        const int numpH,
        const int numzH,
        double2* pxH,
        double* zrH,
        double* zeH,
        double* zpH);

void hydroInitGPU();

void hydroFinalGPU();


#endif /* HYDROGPU_HH_ */
