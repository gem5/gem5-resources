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

/*=========================================================================

Copyright (c) 2011-2012 Argonne National Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

=========================================================================*/

#ifndef RCOForceTree_h
#define RCOForceTree_h

#include "BasicDefinition.h"
#include "ForceLaw.h"
#include "bigchunk.h"

#include <string>
#include <vector>
#include <algorithm>


// The number of points used for the pseudo-particle t-design.
#define QUADRUPOLE_TDPTS 12 // 14
#define MONOPOLE_TDPTS   1

template <int TDPTS>
class RCOForceTree
{
public:
  RCOForceTree(
              POSVEL_T* minLoc,       // Bounding box of halo
              POSVEL_T* maxLoc,       // Bounding box of halo
              POSVEL_T* minForceLoc,  // Bounding box for force updates
              POSVEL_T* maxForceLoc,  // Bounding box for force updates
              ID_T count,             // Number of particles in halo
              POSVEL_T* xLoc,         // Locations of every particle
              POSVEL_T* yLoc,
              POSVEL_T* zLoc,
              POSVEL_T* xVel,         // Velocities of every particle
              POSVEL_T* yVel,
              POSVEL_T* zVel,
              POSVEL_T* mass,         // Mass of each particle
              POSVEL_T* phiLoc,
              ID_T *idLoc,
              MASK_T *maskLoc,
              POSVEL_T avgMass,       // Average mass for estimation
              POSVEL_T fsm,
              POSVEL_T oa,
              ID_T nd = 1,            // The number of particles below which
                                      // to do the direct N^2 calculation
              ID_T ds = 1,            // The "safety" factor to add to the
                                      // estimated maximum depth
              ForceLaw *fl = 0,
              float fcoeff = 0.0,
              POSVEL_T ppc = 0.9);

  ~RCOForceTree();

  void printStats();

protected:
  struct TreeNode
  {
    ID_T count;                       // The number of particles in this node
    ID_T offset;                      // The offset into the particle arrays at
                                      // which data for this tree node starts

    ID_T c[NUM_CHILDREN];             // Left and right children

    POSVEL_T ppm[TDPTS];              // The pseudo-particle masses
    POSVEL_T tdr;                     // The radius of the t-design sphere on
                                      // which the pseudo-particles sit

    POSVEL_T xmin[DIMENSION],
             xmax[DIMENSION],
             xc[DIMENSION];           // The bounding box of this node and its
                                      // center position.

    bool leaf() const {
      for (int i = 0; i < NUM_CHILDREN; ++i) {
        if (c[i] > 0) {
          return false;
        }
      }

      return true;
    }
  };

protected:
  void createRCOForceSubtree(ID_T tl, const ID_T *__restrict tlc);
  void createRCOForceTreeInParallel(ID_T tl = 0);
  void createRCOForceTree();

  void calcInternodeForce(ID_T tl, const std::vector<ID_T> &parents);
  void calcInternodeForces();

protected:
  ID_T   particleCount;         // Total particles

  POSVEL_T fsrrmax;
  POSVEL_T particleMass;        // Average particle mass
  POSVEL_T sinOpeningAngle,     // Criteria for opening node to lower level
           tanOpeningAngle;
  POSVEL_T ppContract;          // The pseudoparticle contraction factor

  POSVEL_T* __restrict xx;      // X location for particles on this processor
  POSVEL_T* __restrict yy;      // Y location for particles on this processor
  POSVEL_T* __restrict zz;      // Z location for particles on this processor
  POSVEL_T* __restrict vx;      // X velocity for particles on this processor
  POSVEL_T* __restrict vy;      // Y velocity for particles on this processor
  POSVEL_T* __restrict vz;      // Z velocity for particles on this processor
  POSVEL_T* __restrict mass;    // Mass for particles on this processor

  POSVEL_T* __restrict phi;
  ID_T*     __restrict id;
  MASK_T*   __restrict mask;

  POSVEL_T minRange[DIMENSION]; // Physical range of data
  POSVEL_T maxRange[DIMENSION]; // Physical range of data
  POSVEL_T minForceRange[DIMENSION]; // Physical range of data for force updates
  POSVEL_T maxForceRange[DIMENSION]; // Physical range of data for force updates

  ID_T nDirect;
  ID_T depthSafety;

  vector<TreeNode, bigchunk_allocator<TreeNode> > tree; // Internal nodes of tree

  bool m_own_fl;
  ForceLaw *m_fl;
  float m_fcoeff;

  // Interaction lists (one per thread)
  vector<vector<POSVEL_T> > inx, iny, inz, inm;
  vector<vector<ID_T> > iq; // The interaction queue
};

typedef RCOForceTree<QUADRUPOLE_TDPTS> RCOQuadrupoleForceTree;
typedef RCOForceTree<MONOPOLE_TDPTS>   RCOMonopoleForceTree;

#endif // RCOForceTree_h

