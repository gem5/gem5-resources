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

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <set>
#include <math.h>

#include "Partition.h"
#include "ForceTree.h"

#ifndef USE_VTK_COSMO
#include "Timings.h"
#endif

using namespace std;

/////////////////////////////////////////////////////////////////////////
//
// ForceTree calculates particle force using a BHTree
//
/////////////////////////////////////////////////////////////////////////

ForceTree::ForceTree()
{
  // Get the number of processors and rank of this processor
  this->numProc = Partition::getNumProc();
  this->myProc = Partition::getMyProc();

  this->bhTree = 0;
}

ForceTree::~ForceTree()
{
  if (this->bhTree != 0) delete this->bhTree;
}

/////////////////////////////////////////////////////////////////////////
//
// Set the parameters for algorithms
//
/////////////////////////////////////////////////////////////////////////

void ForceTree::setParameters(
                        POSVEL_T* minPos,
			POSVEL_T* maxPos,
			POSVEL_T openAngle,
			POSVEL_T critRadius,
			int minGroup,
			int maxGroup,
			POSVEL_T pmass)
{
  for (int dim = 0; dim < DIMENSION; dim++) {
    this->minLoc[dim] = minPos[dim];
    this->maxLoc[dim] = maxPos[dim];
  }
  this->openingAngle = openAngle;
  this->criticalRadius = critRadius;
  this->minimumGroup = minGroup;
  this->maximumGroup = maxGroup;
  this->particleMass = pmass;
}

/////////////////////////////////////////////////////////////////////////
//
// Set the particle vectors that have already been read and which
// contain only the alive particles for this processor
//
/////////////////////////////////////////////////////////////////////////

void ForceTree::setParticles(
                        vector<POSVEL_T>* xLoc,
                        vector<POSVEL_T>* yLoc,
                        vector<POSVEL_T>* zLoc,
                        vector<POSVEL_T>* xVel,
                        vector<POSVEL_T>* yVel,
                        vector<POSVEL_T>* zVel,
                        vector<POSVEL_T>* pmass,
                        vector<POTENTIAL_T>* potential)
{
  // Extract the contiguous data block from a vector pointer
  this->particleCount = (long) xLoc->size();
  this->xx = &(*xLoc)[0];
  this->yy = &(*yLoc)[0];
  this->zz = &(*zLoc)[0];
  this->vx = &(*xVel)[0];
  this->vy = &(*yVel)[0];
  this->vz = &(*zVel)[0];
  this->mass = &(*pmass)[0];
  this->pot = &(*potential)[0];
}

/////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////

void ForceTree::buildForceTree()
{
  // BHTree is constructed from halo particles
  this->bhTree = new BHForceTree(
                            minLoc, maxLoc,
                            this->particleCount,
                            this->xx, this->yy, this->zz,
                            this->vx, this->vy, this->vz,
                            this->mass, this->particleMass);
}

/////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////

void ForceTree::forceCalculationBarnesAdjust()
{
  this->bhTree->treeForceBarnesAdjust(this->openingAngle,
                                      this->criticalRadius);
}

void ForceTree::forceCalculationBarnesQuick()
{
  this->bhTree->treeForceBarnesQuick(this->openingAngle,
                                     this->criticalRadius);
}

void ForceTree::forceCalculationGadgetTopDown()
{
  for (int p = 0; p < this->particleCount; p++) {
    this->bhTree->treeForceGadgetTopDown(p, this->openingAngle, 
                                         this->criticalRadius);
  }
}

void ForceTree::forceCalculationGadgetBottomUp()
{
  for (int p = 0; p < this->particleCount; p++) {
    this->bhTree->treeForceGadgetBottomUp(p, this->openingAngle, 
                                          this->criticalRadius);
  }
}

void ForceTree::forceCalculationGroup()
{
  this->bhTree->treeForceGroup(this->openingAngle, 
                               this->criticalRadius,
                               this->minimumGroup,
                               this->maximumGroup);
}

void ForceTree::forceCalculationN2()
{
  this->bhTree->treeForceN2(this->criticalRadius);
}
