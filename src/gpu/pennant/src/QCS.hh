/*
 * QCS.hh
 *
 *  Created on: Feb 21, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef QCS_HH_
#define QCS_HH_

#include "Vec2.hh"

// forward declarations
class InputFile;
class Hydro;


class QCS {
public:

    // parent hydro object
    Hydro* hydro;

    double qgamma;                 // gamma coefficient for Q model
    double q1, q2;                 // linear and quadratic coefficients
                                   // for Q model

    double* c0evol;
    double* c0du;
    double* c0div;
    double* c0cos;
    double2* c0qe;
    double2* z0uc;
    double* c0rmu;
    double* c0w;


    QCS(const InputFile* inp, Hydro* h);
    ~QCS();

};  // class QCS



#endif /* QCS_HH_ */
