/*
 * TTS.cc
 *
 *  Created on: Feb 2, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "TTS.hh"

#include "InputFile.hh"

using namespace std;


TTS::TTS(const InputFile* inp, Hydro* h) : hydro(h) {
    alfa = inp->getDouble("alfa", 0.5);
    ssmin = inp->getDouble("ssmin", 0.);

}


TTS::~TTS() {}


