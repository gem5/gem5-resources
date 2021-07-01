/*=========================================================================
                                                                                
Copyright (c) 2007, Los Alamos National Security, LLC

All rights reserved.

Copyright 2007. Los Alamos National Security, LLC. 
This software was produced under U.S. Government contract DE-AC52-06NA25396 
for Los Alamos National Laboratory (LANL), which is operated by 
Los Alamos National Security, LLC for the U.S. Department of Energy. 
The U.S. Government has rights to use, reproduce, and distribute this software. 
NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY,
EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  
If software is modified to produce derivative works, such modified software 
should be clearly marked, so as not to confuse it with the version available 
from LANL.
 
Additionally, redistribution and use in source and binary forms, with or 
without modification, are permitted provided that the following conditions 
are met:
-   Redistributions of source code must retain the above copyright notice, 
    this list of conditions and the following disclaimer. 
-   Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution. 
-   Neither the name of Los Alamos National Security, LLC, Los Alamos National
    Laboratory, LANL, the U.S. Government, nor the names of its contributors
    may be used to endorse or promote products derived from this software 
    without specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
                                                                                
=========================================================================*/

#include <stdlib.h>

#include <fstream>
#include <iostream>
#include "BasicDefinition.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////
//
// First command line parameter is the Cosmo file name
// Second command line parameter is the Gadget-2 file name
//
/////////////////////////////////////////////////////////////////////////////
//
// Gadget-2 format (BLOCK):
//    SKIP_H 4 bytes (size of header)
//    Header (256 bytes)
//    SKIP_H 4 bytes (size of header)
//
//    SKIP_L 4 bytes (size of location block in bytes)
//    Block of location data where each particle's (x,y,z) is stored together
//    SKIP_L 4 bytes (size of location block in bytes)
//
//    SKIP_V 4 bytes (size of velocity block in bytes)
//    Block of velocity data where each particle's (xv,yv,zv) is stored together
//    SKIP_V 4 bytes (size of velocity block in bytes)
//
//    SKIP_T 4 bytes (size of tag block in bytes)
//    Block of tag data
//    SKIP_T 4 bytes (size of tag block in bytes)
//
//    Header file npart[6] array indicates the number of particles of each
//    type stored in the file.  The types are:
//
//       0 Gas
//       1 Halo
//       2 Disk
//       3 Bulge
//       4 Stars
//       5 Boundary
//
//    So npart[1] indicates the number of halo particles
//
/////////////////////////////////////////////////////////////////////////////
//
// Cosmo format (RECORD):
//    X location
//    X velocity
//    Y location
//    Y velocity
//    Z location
//    Z velocity
//    Mass
//    Tag
//
/////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  if (argc != 3) {
    cout << "Usage: CosmoToGadget2 cosmo-file gadget-file" << endl;
    exit(-1);
  }

  string inFile = argv[1];
  string outFile = argv[2];
  int blockSize;

  struct GadgetHeader {
    int      npart[NUM_GADGET_TYPES];	// Number of particles
    double   mass[NUM_GADGET_TYPES];	// Mass of particle
    double   time;
    double   redshift;
    int      flag_sfr;
    int      flag_feedback;
    int      npartTotal[NUM_GADGET_TYPES];
    int      flag_cooling;
    int      num_files; 
    double   BoxSize;
    double   Omega0;
    double   OmegaLambda;
    double   HubbleParam; 
    int      flag_stellarage;
    int      flag_metals;
    int      npartTotalHighWord[NUM_GADGET_TYPES];
    int      flag_entropy_instead_u;
    char     fill[60];			// Fills to 256 bytes
  } gadgetHeader;                          

  // Open input Cosmo file
  ifstream *inStream = new ifstream(inFile.c_str(), ios::in | ios::binary);
  if (inStream->fail()) {
    cout << "File: " << inFile << " cannot be opened" << endl;
    exit(-1);
  }

  // Determine the number of particles in cosmo file by checking file size
  inStream->seekg(0L, ios::end);
  int numberOfParticles = inStream->tellg() / RECORD_SIZE;
  cout << "Number of particles: " << numberOfParticles << endl;
  inStream->seekg(0L, ios::beg);

  // Allocation the gadget2 blocks
  POSVEL_T* location = new POSVEL_T[DIMENSION * numberOfParticles];
  POSVEL_T* velocity = new POSVEL_T[DIMENSION * numberOfParticles];
  ID_T* tag = new ID_T[numberOfParticles];

#ifdef DEBUG
  // Collect statistics
  POSVEL_T minLoc[DIMENSION], maxLoc[DIMENSION];
  POSVEL_T minVel[DIMENSION], maxVel[DIMENSION];
  for (int dim = 0; dim < DIMENSION; dim++) {
    minLoc[dim] = MAX_FLOAT;
    maxLoc[dim] = MIN_FLOAT;
    minVel[dim] = MAX_FLOAT;
    maxVel[dim] = MIN_FLOAT;
  }
#endif

  // Read each cosmo record and transfer data to gadget blocks
  int indx = 0;
  POSVEL_T mass;
  for (int i = 0; i < numberOfParticles; i++) {
    inStream->read(reinterpret_cast<char*>(&location[indx]), sizeof(POSVEL_T));
    inStream->read(reinterpret_cast<char*>(&velocity[indx]), sizeof(POSVEL_T));
    inStream->read(reinterpret_cast<char*>(&location[indx+1]), sizeof(POSVEL_T));
    inStream->read(reinterpret_cast<char*>(&velocity[indx+1]), sizeof(POSVEL_T));
    inStream->read(reinterpret_cast<char*>(&location[indx+2]), sizeof(POSVEL_T));
    inStream->read(reinterpret_cast<char*>(&velocity[indx+2]), sizeof(POSVEL_T));
    inStream->read(reinterpret_cast<char*>(&mass), sizeof(POSVEL_T));
    inStream->read(reinterpret_cast<char*>(&tag[i]), sizeof(ID_T));

#ifdef DEBUG
    // Collect ranges on this file
    for (int dim = 0; dim < DIMENSION; dim++) {
      if (minLoc[dim] > location[indx+dim])
        minLoc[dim] = location[indx+dim];
      if (maxLoc[dim] < location[indx+dim])
        maxLoc[dim] = location[indx+dim];
      if (minVel[dim] > velocity[indx+dim])
        minVel[dim] = velocity[indx+dim];
      if (maxVel[dim] < velocity[indx+dim])
        maxVel[dim] = velocity[indx+dim];
    }
#endif
    indx += DIMENSION;
  }

  // Open the output Gadget-2 file
  ofstream *outStream = new ofstream(outFile.c_str(), ios::out | ios::binary);
  if (outStream->fail()) {
    cout << "File: " << outFile << " cannot be opened" << endl;
    exit(-1);
  }

  // Fill in the Gadget-2 header
  for (int i = 0; i < NUM_GADGET_TYPES; i++) {
    gadgetHeader.npart[i] = 0;
    gadgetHeader.mass[i] = 0.0;
    gadgetHeader.npartTotal[i] = 0;
    gadgetHeader.npartTotalHighWord[i] = 0;
  }
  gadgetHeader.time = 0.0;
  gadgetHeader.redshift = 0.0;
  gadgetHeader.flag_sfr = 0;
  gadgetHeader.flag_feedback = 0;
  gadgetHeader.flag_cooling = 0;
  gadgetHeader.num_files = 0;
  gadgetHeader.BoxSize = 0.0;
  gadgetHeader.Omega0 = 0.0;
  gadgetHeader.OmegaLambda = 0.0;
  gadgetHeader.HubbleParam = 0.0;
  gadgetHeader.flag_stellarage = 0;
  gadgetHeader.flag_metals = 0;
  gadgetHeader.flag_entropy_instead_u = 0;

  gadgetHeader.npart[GADGET_HALO] = numberOfParticles;
  gadgetHeader.mass[GADGET_HALO] = mass;

  // Write the gadget header
  blockSize = sizeof(GadgetHeader);
  outStream->write(reinterpret_cast<char*>(&blockSize), GADGET_SKIP);
  outStream->write(reinterpret_cast<char*>(&gadgetHeader), blockSize);
  outStream->write(reinterpret_cast<char*>(&blockSize), GADGET_SKIP);

  // Write location block
  blockSize = DIMENSION * numberOfParticles * sizeof(POSVEL_T);
  outStream->write(reinterpret_cast<char*>(&blockSize), GADGET_SKIP);
  outStream->write(reinterpret_cast<char*>(location), blockSize);
  outStream->write(reinterpret_cast<char*>(&blockSize), GADGET_SKIP);

  // Write velocity block
  outStream->write(reinterpret_cast<char*>(&blockSize), GADGET_SKIP);
  outStream->write(reinterpret_cast<char*>(velocity), blockSize);
  outStream->write(reinterpret_cast<char*>(&blockSize), GADGET_SKIP);

  // Write tag block
  blockSize = numberOfParticles * sizeof(ID_T);
  outStream->write(reinterpret_cast<char*>(&blockSize), GADGET_SKIP);
  outStream->write(reinterpret_cast<char*>(tag), blockSize);
  outStream->write(reinterpret_cast<char*>(&blockSize), GADGET_SKIP);

  outStream->close();

#ifdef DEBUG
  // Ranges of location and velocity in file
  cout << endl;
  cout << "Number of particles: " << numberOfParticles << endl;
  cout << "Location: ";
  for (int dim = 0; dim < DIMENSION; dim++)
    cout << " [" << minLoc[dim] << ":" << maxLoc[dim] << "] ";
  cout << endl;

  cout << "Velocity: ";
  for (int dim = 0; dim < DIMENSION; dim++)
    cout << " [" << minVel[dim] << ":" << maxVel[dim] << "] ";
  cout << endl << endl;
#endif
}
