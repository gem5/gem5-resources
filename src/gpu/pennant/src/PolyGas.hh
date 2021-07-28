/*
 * PolyGas.hh
 *
 *  Created on: Mar 23, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef POLYGAS_HH_
#define POLYGAS_HH_

#include "Vec2.hh"

// forward declarations
class InputFile;
class Hydro;


class PolyGas {
public:

    // parent hydro object
    Hydro* hydro;

    double gamma;                  // coeff. for ideal gas equation
    double ssmin;                  // minimum sound speed for gas

    PolyGas(const InputFile* inp, Hydro* h);
    ~PolyGas();

};  // class PolyGas


#endif /* POLYGAS_HH_ */
