/*
 * HydroBC.cc
 *
 *  Created on: Jan 13, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "HydroBC.hh"

#include "Vec2.hh"

using namespace std;


HydroBC::HydroBC(
        Mesh* msh,
        const double2 v,
        const vector<int>& mbp)
    : mesh(msh), numb(mbp.size()), vfix(v) {}


HydroBC::~HydroBC() {}


