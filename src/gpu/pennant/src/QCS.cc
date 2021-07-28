/*
 * QCS.cc
 *
 *  Created on: Feb 21, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "QCS.hh"

#include "InputFile.hh"

using namespace std;


QCS::QCS(const InputFile* inp, Hydro* h) : hydro(h) {
    qgamma = inp->getDouble("qgamma", 5. / 3.);
    q1 = inp->getDouble("q1", 0.);
    q2 = inp->getDouble("q2", 2.);

}

QCS::~QCS() {}


