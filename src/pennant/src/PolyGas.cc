/*
 * PolyGas.cc
 *
 *  Created on: Mar 26, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "PolyGas.hh"

#include "InputFile.hh"

using namespace std;


PolyGas::PolyGas(const InputFile* inp, Hydro* h) : hydro(h) {
    gamma = inp->getDouble("gamma", 5. / 3.);
    ssmin = inp->getDouble("ssmin", 0.);

}


