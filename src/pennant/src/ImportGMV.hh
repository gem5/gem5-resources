/*
 * ImportGMV.hh
 *
 *  Created on: Feb 27, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef IMPORTGMV_HH_
#define IMPORTGMV_HH_

#include <string>
#include <vector>
#include "Vec2.hh"

// forward declarations
class Mesh;


class ImportGMV {
public:

    // parent object
    Mesh* mesh;

    ImportGMV(Mesh* m);
    ~ImportGMV();

    void read(
            const std::string& filename,
            std::vector<double2>& nodepos,
            std::vector<int>& cellstart,
            std::vector<int>& cellsize,
            std::vector<int>& cellnodes);
}; // class ImportGMV


#endif /* IMPORTGMV_HH_ */
