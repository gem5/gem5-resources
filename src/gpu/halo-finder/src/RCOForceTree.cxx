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

#include "Timings.h"
#include "RCOForceTree.h"

#include <cstring>
#include <cstdio>
using namespace std;

// References:
// Emanuel Gafton and Stephan Rosswog. A fast recursive coordinate bisection tree for
// neighbour search and gravity. Mon. Not. R. Astron. Soc. to appear, 2011.
// http://arxiv.org/abs/1108.0028v1
//
// Atsushi Kawai, Junichiro Makino and Toshikazu Ebisuzaki.
// Performance Analysis of High-Accuracy Tree Code Based on the Pseudoparticle
// Multipole Method. The Astrophysical Journal Supplement Series, 151:13-33, 2004.
// Related: http://arxiv.org/abs/astro-ph/0012041v1
//
// R. H. Hardin and N. J. Sloane
// New Spherical 4-Designs. Discrete Math, 106/107 255-264, 1992.
//
// The library of spherical designs:
// http://www2.research.att.com/~njas/sphdesigns/
namespace {
template <int TDPTS>
struct sphdesign {};

#define DECLARE_SPHDESIGN(TDPTS) \
template <> \
struct sphdesign<TDPTS> \
{ \
  static const POSVEL_T x[TDPTS]; \
  static const POSVEL_T y[TDPTS]; \
  static const POSVEL_T z[TDPTS]; \
}; \
/**/

DECLARE_SPHDESIGN(1)
DECLARE_SPHDESIGN(2)
DECLARE_SPHDESIGN(3)
DECLARE_SPHDESIGN(4)
DECLARE_SPHDESIGN(6)
DECLARE_SPHDESIGN(12)
DECLARE_SPHDESIGN(14)

#undef DECLARE_SPHDESIGN

/* this is not a t-design, but puts the monopole moment
   at the center of mass. */
const POSVEL_T sphdesign<1>::x[] = {
  0
};

const POSVEL_T sphdesign<1>::y[] = {
  0
};

const POSVEL_T sphdesign<1>::z[] = {
  0
};

const POSVEL_T sphdesign<2>::x[] = {
  1.0,
  -1.0
};

const POSVEL_T sphdesign<2>::y[] = {
  0,
  0
};

const POSVEL_T sphdesign<2>::z[] = {
  0,
  0
};

const POSVEL_T sphdesign<3>::x[] = {
  1.0,
  -.5,
  -.5
};

const POSVEL_T sphdesign<3>::y[] = {
  0,
  .86602540378443864675,
  -.86602540378443864675
};

const POSVEL_T sphdesign<3>::z[] = {
  0,
  0,
  0
};

const POSVEL_T sphdesign<4>::x[] = {
  .577350269189625763,
  .577350269189625763,
  -.577350269189625763,
  -.577350269189625763
};

const POSVEL_T sphdesign<4>::y[] = {
  .577350269189625763,
  -.577350269189625763,
  .577350269189625763,
  -.577350269189625763
};

const POSVEL_T sphdesign<4>::z[] = {
  .577350269189625763,
  -.577350269189625763,
  -.577350269189625763,
  .577350269189625763
};

const POSVEL_T sphdesign<6>::x[] = {
  1.0,
  -1.0,
  0,
  0,
  0,
  0
};

const POSVEL_T sphdesign<6>::y[] = {
  0,
  0,
  1.0,
  -1.0,
  0,
  0
};

const POSVEL_T sphdesign<6>::z[] = {
  0,
  0,
  0,
  0,
  1.0,
  -1.0
};

// This is a 3-D 12-point spherical 4-design
// (the verticies of a icosahedron) from Hardin and Sloane.
const POSVEL_T sphdesign<12>::x[] = {
  0,
  0,
  0.525731112119134,
  -0.525731112119134,
  0.85065080835204,
  -0.85065080835204,
  0,
  0,
  -0.525731112119134,
  0.525731112119134,
  -0.85065080835204,
  0.85065080835204
};

const POSVEL_T sphdesign<12>::y[] = {
  0.85065080835204,
  0.85065080835204,
  0,
  0,
  0.525731112119134,
  0.525731112119134,
  -0.85065080835204,
  -0.85065080835204,
  0,
  0,
  -0.525731112119134,
  -0.525731112119134
};

const POSVEL_T sphdesign<12>::z[] = {
  0.525731112119134,
  -0.525731112119134,
  0.85065080835204,
  0.85065080835204,
  0,
  0,
  -0.525731112119134,
  0.525731112119134,
  -0.85065080835204,
  -0.85065080835204,
  0,
  0
};

// This is a 3-D 14-point spherical 4-design by
// R. H. Hardin and N. J. A. Sloane.
const POSVEL_T sphdesign<14>::x[] = {
  1.0e0,
  5.947189772040725e-1,
  5.947189772040725e-1,
  5.947189772040725e-1,
  -5.947189772040725e-1,
  -5.947189772040725e-1,
  -5.947189772040725e-1,
  3.012536847870683e-1,
  3.012536847870683e-1,
  3.012536847870683e-1,
  -3.012536847870683e-1,
  -3.012536847870683e-1,
  -3.012536847870683e-1,
  -1.0e0
};

const POSVEL_T sphdesign<14>::y[] = {
  0.0e0,
  1.776539926025823e-1,
  -7.678419429698292e-1,
  5.90187950367247e-1,
  1.776539926025823e-1,
  5.90187950367247e-1,
  -7.678419429698292e-1,
  8.79474443923065e-1,
  -7.588425179318781e-1,
  -1.206319259911869e-1,
  8.79474443923065e-1,
  -1.206319259911869e-1,
  -7.588425179318781e-1,
  0.0e0
};

const POSVEL_T sphdesign<14>::z[] = {
  0.0e0,
  7.840589244857197e-1,
  -2.381765915652909e-1,
  -5.458823329204288e-1,
  -7.840589244857197e-1,
  5.458823329204288e-1,
  2.381765915652909e-1,
  3.684710570566285e-1,
  5.774116818882528e-1,
  -9.458827389448813e-1,
  -3.684710570566285e-1,
  9.458827389448813e-1,
  -5.774116818882528e-1,
  0.0e0
};
} // anonymous namespace

// Note: In Gafton and Rosswog the far-field force contribution is calculated
// per-cell (at the center of mass), and then a Taylor expansion about the center
// of mass is used to calculate the force on the individual particles. For this to
// work, the functional form of the force must be known (because the Jacobian
// and Hessian are required). Here, however, the functional form is not known,
// and so the pseudo-particle method of Makino is used instead.

template <int TDPTS>
RCOForceTree<TDPTS>::RCOForceTree(
			 POSVEL_T* minLoc,
			 POSVEL_T* maxLoc,
			 POSVEL_T* minForceLoc,
			 POSVEL_T* maxForceLoc,
			 ID_T count,
			 POSVEL_T* xLoc,
			 POSVEL_T* yLoc,
			 POSVEL_T* zLoc,
			 POSVEL_T* xVel,
			 POSVEL_T* yVel,
			 POSVEL_T* zVel,
			 POSVEL_T* ms,
                         POSVEL_T* phiLoc,
                         ID_T *idLoc,
                         MASK_T *maskLoc,
			 POSVEL_T avgMass,
                         POSVEL_T fsm,
                         POSVEL_T oa,
                         ID_T nd,
                         ID_T ds,
			 ForceLaw *fl,
			 float fcoeff,
                         POSVEL_T ppc)
{
  // Extract the contiguous data block from a vector pointer
  particleCount = count;

  xx = xLoc;
  yy = yLoc;
  zz = zLoc;
  vx = xVel;
  vy = yVel;
  vz = zVel;
  mass = ms;
  phi = phiLoc;
  id = idLoc;
  mask = maskLoc;

  particleMass = avgMass;
  fsrrmax = fsm;
  sinOpeningAngle = sinf(oa);
  tanOpeningAngle = tanf(oa);
  nDirect = nd;
  depthSafety = ds;
  ppContract = ppc;

  // Find the grid size of this chaining mesh
  for (int dim = 0; dim < DIMENSION; dim++) {
    minRange[dim] = minLoc[dim];
    maxRange[dim] = maxLoc[dim];
    minForceRange[dim] = minForceLoc[dim];
    maxForceRange[dim] = maxForceLoc[dim];
  }

  if (fl) {
    m_own_fl = false;
    m_fl = fl;
    m_fcoeff = fcoeff;
  } else {
    //maybe change this to Newton's law or something
    m_own_fl = true;
    m_fl = new ForceLawNewton();
    m_fcoeff = 1.0;
  }

  // Because the tree may be built in parallel, and no efficient way of locking
  // the tree seems to be available in OpenMP (no reader/writer locks, etc.),
  // we just estimate the number of tree nodes that will be needed. Hopefully,
  // this will be an over estimate. If we need more than this, then tree nodes
  // that really should be subdivided will not be.
  //
  // If the tree were perfectly balanced, then it would have a depth of
  // log_8(particleCount/nDirect). The tree needs to have (8^depth)+1 entries.
  // To that, a safety factor is added to the depth.
  ID_T nds = (((ID_T)(particleCount/(POSVEL_T)nDirect)) << depthSafety) + 1;
  tree.reserve(nds);

  // Create the recursive RCO tree from the particle locations
  createRCOForceTree();

  printStats();

  // Interaction lists.
  int nthreads = 1;
  inx.resize(nthreads);
  iny.resize(nthreads);
  inz.resize(nthreads);
  inm.resize(nthreads);
  iq.resize(nthreads);

  calcInternodeForces();
}

template <int TDPTS>
RCOForceTree<TDPTS>::~RCOForceTree()
{
  if (m_own_fl) {
    delete m_fl;
  }
}

template <int TDPTS>
void RCOForceTree<TDPTS>::printStats()
{
  size_t zeroLeafNodes = 0;
  size_t nonzeroLeafNodes = 0;
  size_t maxPPN = 0;
  size_t leafParts = 0;

  for (ID_T tl = 1; tl < (ID_T) tree.size(); ++tl) {
    if (tree[tl].leaf()) {
      if (tree[tl].count > 0) {
        ++nonzeroLeafNodes;

        leafParts += tree[tl].count;
        maxPPN = std::max((size_t) tree[tl].count, maxPPN);
      } else {
        ++zeroLeafNodes;
      }
    }
  }

  printf("\ttree post-build statistics:\n");
  printf("\t\tparticles: %lu\n", (size_t) particleCount);
  printf("\t\tnodes: %lu (allocated: %lu)\n", tree.size(), tree.capacity());
  printf("\t\tleaves: %lu (empty: %lu)\n", zeroLeafNodes+nonzeroLeafNodes, zeroLeafNodes);
  printf("\t\tmean ppn: %.2f (max ppn: %lu)\n", leafParts/((double) nonzeroLeafNodes), maxPPN);
}

static inline void cm(ID_T count, const POSVEL_T* __restrict xx, const POSVEL_T* __restrict yy,
                      const POSVEL_T* __restrict zz, const POSVEL_T* __restrict mass,
                      POSVEL_T* __restrict xmin, POSVEL_T* __restrict xmax, POSVEL_T* __restrict xc)
{
  // xmin/xmax are currently set to the whole bounding box, but this is too conservative, so we'll
  // set them based on the actual particle content.

  double x = 0, y = 0, z = 0, m = 0;

  for (int i = 0; i < count; ++i) {
    if (i == 0) {
      xmin[0] = xmax[0] = xx[0];
      xmin[1] = xmax[1] = yy[0];
      xmin[2] = xmax[2] = zz[0];
    } else {
      xmin[0] = std::min(xmin[0], xx[i]);
      xmax[0] = std::max(xmax[0], xx[i]);
      xmin[1] = std::min(xmin[1], yy[i]);
      xmax[1] = std::max(xmax[1], yy[i]);
      xmin[2] = std::min(xmin[2], zz[i]);
      xmax[2] = std::max(xmax[2], zz[i]);
    }

    POSVEL_T w = mass[i];
    x += w*xx[i];
    y += w*yy[i];
    z += w*zz[i];
    m += w;
  }

  xc[0] = (POSVEL_T) (x/m);
  xc[1] = (POSVEL_T) (y/m);
  xc[2] = (POSVEL_T) (z/m);
}

static inline POSVEL_T pptdr(const POSVEL_T* __restrict xmin, const POSVEL_T* __restrict xmax, const POSVEL_T* __restrict xc)
{
  return std::min(xmax[0] - xc[0], std::min(xmax[1] - xc[1], std::min(xmax[2] - xc[2], std::min(xc[0] - xmin[0],
                 std::min(xc[1] - xmin[1], xc[2] - xmin[2])))));
}

template <int TDPTS>
static inline void pppts(POSVEL_T tdr, const POSVEL_T* __restrict xc,
                         POSVEL_T* __restrict ppx, POSVEL_T* __restrict ppy, POSVEL_T* __restrict ppz)
{
  for (int i = 0; i < TDPTS; ++i) {
    ppx[i] = tdr*sphdesign<TDPTS>::x[i] + xc[0];
    ppy[i] = tdr*sphdesign<TDPTS>::y[i] + xc[1];
    ppz[i] = tdr*sphdesign<TDPTS>::z[i] + xc[2];
  }
}

template <int TDPTS>
static inline void pp(ID_T count, const POSVEL_T* __restrict xx, const POSVEL_T* __restrict yy,
                      const POSVEL_T* __restrict zz, const POSVEL_T* __restrict mass, const POSVEL_T* __restrict xc,
                      const POSVEL_T* __restrict ppx, const POSVEL_T* __restrict ppy, const POSVEL_T* __restrict ppz,
                      POSVEL_T* __restrict ppm, POSVEL_T tdr)
{
  POSVEL_T K = TDPTS;
  POSVEL_T odr0 = 1/K;

  for (int i = 0; i < count; ++i) {
    POSVEL_T xi = xx[i] - xc[0];
    POSVEL_T yi = yy[i] - xc[1];
    POSVEL_T zi = zz[i] - xc[2];
    POSVEL_T ri = sqrtf(xi*xi + yi*yi + zi*zi);

    for (int j = 0; j < TDPTS; ++j) {
      POSVEL_T xj = ppx[j] - xc[0];
      POSVEL_T yj = ppy[j] - xc[1];
      POSVEL_T zj = ppz[j] - xc[2];
      POSVEL_T rj2 = xj*xj + yj*yj + zj*zj;

      POSVEL_T odr1 = 0, odr2 = 0;
      if (rj2 != 0) {
        POSVEL_T rj  = sqrtf(rj2);
        POSVEL_T aij = (xi*xj + yi*yj + zi*zj)/(ri*rj);

        odr1 = (3/K)*(ri/tdr)*aij;
        odr2 = (5/K)*(ri/tdr)*(ri/tdr)*0.5*(3*aij*aij - 1);
      }

      ppm[j] += mass[i]*(odr0 + odr1 + odr2);
    }
  }
}

static inline void nbody1(ID_T count, ID_T count1, const POSVEL_T* __restrict xx, const POSVEL_T* __restrict yy,
                         const POSVEL_T* __restrict zz, const POSVEL_T* __restrict mass,
                         const POSVEL_T* __restrict xx1, const POSVEL_T* __restrict yy1,
                         const POSVEL_T* __restrict zz1, const POSVEL_T* __restrict mass1,
                         POSVEL_T* __restrict vx, POSVEL_T* __restrict vy, POSVEL_T* __restrict vz,
                         ForceLaw *fl, float fcoeff, float fsrrmax)
{
  POSVEL_T fsrrmax2 = fsrrmax*fsrrmax;

  for (int i = 0; i < count; ++i)
    for (int j = 0; j < count1; ++j) {
      POSVEL_T dx = xx1[j] - xx[i];
      POSVEL_T dy = yy1[j] - yy[i];
      POSVEL_T dz = zz1[j] - zz[i];
      POSVEL_T dist2 = dx*dx + dy*dy + dz*dz;
      POSVEL_T f_over_r = mass[i]*mass1[j] * fl->f_over_r(dist2);

      POSVEL_T updateq = 1.0;
      updateq *= (dist2 < fsrrmax2);

      vx[i] += updateq*fcoeff*f_over_r*dx;
      vy[i] += updateq*fcoeff*f_over_r*dy;
      vz[i] += updateq*fcoeff*f_over_r*dz;
    }
}

static inline void partition(ID_T n,
                             POSVEL_T* __restrict xx, POSVEL_T* __restrict yy, POSVEL_T* __restrict zz,
                             POSVEL_T* __restrict vx, POSVEL_T* __restrict vy, POSVEL_T* __restrict vz,
                             POSVEL_T* __restrict mass, POSVEL_T* __restrict phi,
                             ID_T* __restrict id, MASK_T* __restrict mask, const POSVEL_T* __restrict pv,
                             ID_T* __restrict is)
{
  // Initialize all split index values to 0. These must be kept ordered (is[i+1] >= is[i]).
  // All values less than is[0] have all three coords less than those in split. All values
  // less than is[1] have x and y less than split but z >= split[2], etc.
  for (int i = 0; i < NUM_CHILDREN-1; ++i) {
    is[i] = 0;
  }

  for (ID_T i = 0; i < n; ++i) {
    int p = 0;
    if (xx[i] >= pv[0]) p += NUM_CHILDREN/2;
    if (yy[i] >= pv[1]) p += NUM_CHILDREN/4;
    if (zz[i] >= pv[2]) p += NUM_CHILDREN/8;
    if (p >= NUM_CHILDREN-1) continue;

    // This value needs to be moved into the bin that ends one element before is[p].
    // If that is not the second-to-last bin then is[p] probably points to the first
    // element in is[p+1].

#define CHAIN_SWAP(var) \
    std::swap(var[is[NUM_CHILDREN-1]], var[i]); \
    for (int q = NUM_CHILDREN-1; q > p; --q) std::swap(var[is[q-1]], var[is[q]]); \
/**/
    CHAIN_SWAP(xx)
    CHAIN_SWAP(yy)
    CHAIN_SWAP(zz)
    CHAIN_SWAP(vx)
    CHAIN_SWAP(vy)
    CHAIN_SWAP(vz)
    CHAIN_SWAP(mass)
    CHAIN_SWAP(phi)
    CHAIN_SWAP(id)
    CHAIN_SWAP(mask)
#undef CHAIN_SWAP

    do { ++is[p++]; } while (p < NUM_CHILDREN-1);
  }
}
                             
template <int TDPTS>
void RCOForceTree<TDPTS>::createRCOForceSubtree(ID_T tl, const ID_T *__restrict tlc)
{
  ID_T is[NUM_CHILDREN-1];
  POSVEL_T split[DIMENSION];

  const bool geoSplit = false;
  for (int i = 0; i < DIMENSION; ++i) {
    split[i] = geoSplit ? (tree[tl].xmax[i]+tree[tl].xmin[i])/2 : tree[tl].xc[i];
  }

  ::partition(tree[tl].count, xx + tree[tl].offset, yy + tree[tl].offset, zz + tree[tl].offset,
              vx + tree[tl].offset, vy + tree[tl].offset, vz + tree[tl].offset,
              mass + tree[tl].offset, phi + tree[tl].offset,
              id + tree[tl].offset, mask + tree[tl].offset, split, is);

  bool noDiv = true;
  for (int i = 0; i < NUM_CHILDREN-1; ++i) {
    if (!(is[i] == 0 || is[i] == tree[tl].count)) {
      noDiv = false;
    }
  }
  if (noDiv) return;

  tree[tlc[0]].count = is[0];
  for (int i = 1; i < NUM_CHILDREN-1; ++i) {
    tree[tlc[i]].count = is[i] - is[i-1];
  }
  tree[tlc[NUM_CHILDREN-1]].count = tree[tl].count - is[NUM_CHILDREN-2];

  ID_T coff = tree[tl].offset;
  for (int i = 0; i < NUM_CHILDREN; ++i) {
    if (tree[tlc[i]].count > 0) {
      tree[tl].c[i] = tlc[i];
      tree[tlc[i]].offset = coff;
      // Note: tree[tlc[i]].xmax,xmin are not set here b/c cm will
      // compute them from the contained particles.
      coff += tree[tlc[i]].count;

      createRCOForceTreeInParallel(tlc[i]);
    }
  }
}

// This is basically the algorithm from (Gafton and Rosswog, 2011).
template <int TDPTS>
void RCOForceTree<TDPTS>::createRCOForceTreeInParallel(ID_T tl)
{
  ID_T cnt = tree[tl].count;
  ID_T off = tree[tl].offset;

  // Compute the center-of-mass coordinates (and recompute the min/max)
  ::cm(cnt, xx + off, yy + off, zz + off, mass + off,
       tree[tl].xmin, tree[tl].xmax, tree[tl].xc);

  if (cnt <= nDirect) {
    // The pseudoparticles
    tree[tl].tdr = ppContract*::pptdr(tree[tl].xmin, tree[tl].xmax, tree[tl].xc);
    memset(tree[tl].ppm, 0, sizeof(POSVEL_T)*TDPTS);
    if (cnt > TDPTS) { // Otherwise, the pseudoparticles are never used
      POSVEL_T ppx[TDPTS], ppy[TDPTS], ppz[TDPTS];
      pppts<TDPTS>(tree[tl].tdr, tree[tl].xc, ppx, ppy, ppz);
      pp<TDPTS>(cnt, xx + off, yy + off, zz + off, mass + off, tree[tl].xc,
                ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
    }

    return;
  }

  // Index of the right and left child levels
  ID_T tlc[NUM_CHILDREN];
  tlc[0] = tree.size();
  for (int i = 1; i < NUM_CHILDREN; ++i) {
    tlc[i] = tlc[0]+i;
  }
  tree.resize(tlc[NUM_CHILDREN-1]+1);
  memset(&tree[tlc[0]], 0, sizeof(TreeNode)*NUM_CHILDREN);

  // Both children have similar bounding boxes to the current node (the
  // parent), so copy the bounding box here, and then overwrite the changed
  // coordinate later.
  for (int i = 0; i < DIMENSION; ++i) {
    for (int j = 0; j < NUM_CHILDREN; ++j) {
      tree[tlc[j]].xmin[i] = tree[tl].xmin[i];
      tree[tlc[j]].xmax[i] = tree[tl].xmax[i];
    }
  }

  // Split all edges at the center of mass.
  createRCOForceSubtree(tl, tlc);

  // Compute the pseudoparticles based on those of the children
  POSVEL_T ppx[TDPTS], ppy[TDPTS], ppz[TDPTS];
  tree[tl].tdr = ppContract*::pptdr(tree[tl].xmin, tree[tl].xmax, tree[tl].xc);
  pppts<TDPTS>(tree[tl].tdr, tree[tl].xc, ppx, ppy, ppz);
  memset(tree[tl].ppm, 0, sizeof(POSVEL_T)*TDPTS);

  for (int i = 0; i < NUM_CHILDREN; ++i) {
    if (tree[tlc[i]].count > 0) {
      if (tree[tlc[i]].count <= TDPTS) {
        ID_T offc = tree[tlc[i]].offset;
        pp<TDPTS>(tree[tlc[i]].count, xx + offc, yy + offc, zz + offc, mass + offc,
                  tree[tl].xc, ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
      } else {
        POSVEL_T ppxc[TDPTS], ppyc[TDPTS], ppzc[TDPTS];
        pppts<TDPTS>(tree[tlc[i]].tdr, tree[tlc[i]].xc, ppxc, ppyc, ppzc);
        pp<TDPTS>(TDPTS, ppxc, ppyc, ppzc, tree[tlc[i]].ppm, tree[tl].xc,
                  ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
      }
    }
  }
}

template <int TDPTS>
void RCOForceTree<TDPTS>::createRCOForceTree()
{
  // The top tree is the entire box
  tree.resize(1);
  memset(&tree[0], 0, sizeof(TreeNode));

  tree[0].count = particleCount;
  tree[0].offset = 0;

  for (int i = 0; i < DIMENSION; ++i) {
    tree[0].xmin[i] = minRange[i];
    tree[0].xmax[i] = maxRange[i];
  }

  createRCOForceTreeInParallel();
}

template <int TDPTS>
void RCOForceTree<TDPTS>::calcInternodeForce(ID_T tl,
                                            const std::vector<ID_T> &parents) {
  POSVEL_T fsrrmax2 = fsrrmax*fsrrmax;

  int tid = 0;

  std::vector<ID_T> &q = iq[tid];
  q.clear();
  q.push_back(0);

  // The interaction list.
  std::vector<POSVEL_T> &nx = inx[tid];
  std::vector<POSVEL_T> &ny = iny[tid];
  std::vector<POSVEL_T> &nz = inz[tid];
  std::vector<POSVEL_T> &nm = inm[tid];
  nx.clear();
  ny.clear();
  nz.clear();
  nm.clear();
  nx.reserve(4096);
  ny.reserve(4096);
  nz.reserve(4096);
  nm.reserve(4096);

  while (!q.empty()) {
    ID_T tln = q.back();
    q.pop_back();

    // We should not interact with our own parents.
    if (tln < tl) {
      bool isParent = std::binary_search(parents.begin(), parents.end(), tln);
      if (isParent) {
        for (int i = 0; i < NUM_CHILDREN; ++i) {
          ID_T tlnc = tree[tln].c[i];

          if (tlnc != tl && tlnc > 0 && tree[tlnc].count > 0) {
            q.push_back(tlnc);
          }
        }

        continue;
      }
    }

    // Is this node have a small enough opening angle to interact with?
    POSVEL_T dx = tree[tln].xc[0] - tree[tl].xc[0];
    POSVEL_T dy = tree[tln].xc[1] - tree[tl].xc[1];
    POSVEL_T dz = tree[tln].xc[2] - tree[tl].xc[2];
    POSVEL_T dist2 = dx*dx + dy*dy + dz*dz;

    POSVEL_T sx = tree[tln].xmax[0]-tree[tln].xmin[0];
    POSVEL_T sy = tree[tln].xmax[1]-tree[tln].xmin[1];
    POSVEL_T sz = tree[tln].xmax[2]-tree[tln].xmin[2];
    POSVEL_T l2 = std::min(sx*sx, std::min(sy*sy, sz*sz)); // under-estimate

    POSVEL_T dtt2 = dist2*tanOpeningAngle*tanOpeningAngle; 
    bool looksBig;
    // l2/dist2 is really tan^2 theta, for small theta, tan(theta) ~ theta
    if (l2 > dtt2) {
      // the under-estimate is too big, so this is definitely too big
      looksBig = true;
    } else {
      // There are 8 corner points of the remote node, and the maximum angular
      // size will be from one of those points to its opposite points. So there
      // are 8 vector dot products to compute to determine the maximum angular
      // size at any given reference point. (do we need to do this for each point
      // in leaf node, or will the c.m. point be sufficient?).
      looksBig = false;
      for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) {
        POSVEL_T x1 = (i == 0 ? tree[tln].xmin : tree[tln].xmax)[0] - tree[tl].xc[0];
        POSVEL_T y1 = (j == 0 ? tree[tln].xmin : tree[tln].xmax)[1] - tree[tl].xc[1];
        POSVEL_T z1 = tree[tln].xmin[2] - tree[tl].xc[2];
  
        POSVEL_T x2 = (i == 0 ? tree[tln].xmax : tree[tln].xmin)[0] - tree[tl].xc[0];
        POSVEL_T y2 = (j == 0 ? tree[tln].xmax : tree[tln].xmin)[1] - tree[tl].xc[1];
        POSVEL_T z2 = tree[tln].xmax[2] - tree[tl].xc[2];
 
        const bool useRealOA = false;
        if (useRealOA) {
          // |a x b| = a*b*sin(theta)
          POSVEL_T cx = y1*z2 - z1*y2;
          POSVEL_T cy = z1*x2 - x1*z2;
          POSVEL_T cz = x1*y2 - y1*x2;
          if ((cx*cx + cy*cy + cz*cz) > sinOpeningAngle*sinOpeningAngle*
                                          (x1*x1 + y1*y1 + z1*z1)*(x2*x2 + y2*y2 + z2*z2)
             ) {
            looksBig = true;
            break;
          }
        } else {
          // Instead of using the real opening angle, use the tan approximation; this is
          // better than the opening-angle b/c it incorporates depth information.
          POSVEL_T ddx = x1 - x2, ddy = y1 - y2, ddz = z1 - z2;
          POSVEL_T dh2 = ddx*ddx + ddy*ddy + ddz*ddz;
          if (dh2 > dtt2) {
            looksBig = true;
            break;
          }
        }
      }
    }

    if (!looksBig) {
      if (dist2 > fsrrmax2) {
        // We could interact with this node, but it is too far away to make
        // any difference, so it will be skipped, along with all of its
        // children.
        continue;
      }

      // This node has fewer particles than pseudo particles, so just use the
      // particles that are actually there.
      if (tree[tln].count <= TDPTS) {
        ID_T offn = tree[tln].offset;
        ID_T cntn = tree[tln].count;
        size_t start = nx.size();
        nx.resize(nx.size() + cntn);
        ny.resize(ny.size() + cntn);
        nz.resize(nz.size() + cntn);
        nm.resize(nm.size() + cntn);
        for (size_t i = 0; i < (size_t) cntn; ++i) {
          nx[start + i] = xx[offn + i];
          ny[start + i] = yy[offn + i];
          nz[start + i] = zz[offn + i];
          nm[start + i] = mass[offn + i];
        }

        continue;
      }

      // Interact the particles in this node with the pseudoparticles of the
      // other node.
      size_t start = nx.size();
      nx.resize(nx.size() + TDPTS);
      ny.resize(ny.size() + TDPTS);
      nz.resize(nz.size() + TDPTS);
      nm.resize(nm.size() + TDPTS);
      pppts<TDPTS>(tree[tln].tdr, tree[tln].xc, &nx[start], &ny[start], &nz[start]);
      for (size_t i = 0; i < (size_t) TDPTS; ++i) {
        nm[start + i] = tree[tln].ppm[i];
      }

      continue;
    } else if (tree[tln].leaf()) {
      // This is a leaf node with which we must interact.
      ID_T offn = tree[tln].offset;
      ID_T cntn = tree[tln].count;
      size_t start = nx.size();
      nx.resize(nx.size() + cntn);
      ny.resize(ny.size() + cntn);
      nz.resize(nz.size() + cntn);
      nm.resize(nm.size() + cntn);
      for (size_t i = 0; i < (size_t) cntn; ++i) {
        nx[start + i] = xx[offn + i];
        ny[start + i] = yy[offn + i];
        nz[start + i] = zz[offn + i];
        nm[start + i] = mass[offn + i];
      }

      continue;
    }

    // This other node is not a leaf, but has too large an opening angle
    // for an approx. interaction: queue its children.

    for (int j = 0; j < NUM_CHILDREN; ++j) {
      ID_T tlnc = tree[tln].c[j];

      if (tlnc > 0 && tree[tlnc].count > 0) {
        bool close = true;
        for (int i = 0; i < DIMENSION; ++i) {
          POSVEL_T dist = 0;
          if (tree[tl].xmax[i] < tree[tlnc].xmin[i]) {
            dist = tree[tlnc].xmin[i] - tree[tl].xmax[i];
          } else if (tree[tl].xmin[i] > tree[tlnc].xmax[i]) {
            dist = tree[tl].xmin[i] - tree[tlnc].xmax[i];
          }
  
          if (dist > fsrrmax) {
            close = false;
            break;
          }
        }
  
        if (close) q.push_back(tlnc);
      }
    }
  }

  ID_T off = tree[tl].offset;
  ID_T cnt = tree[tl].count;

  // Add self interactions...
  size_t start = nx.size();
  nx.resize(nx.size() + cnt);
  ny.resize(ny.size() + cnt);
  nz.resize(nz.size() + cnt);
  nm.resize(nm.size() + cnt);
  for (size_t i = 0; i < (size_t) cnt; ++i) {
    nx[start + i] = xx[off + i];
    ny[start + i] = yy[off + i];
    nz[start + i] = zz[off + i];
    nm[start + i] = mass[off + i];
  }

  // Process the interaction list...
  ::nbody1(cnt, nx.size(),
           xx + off, yy + off, zz + off, mass + off,
           &nx[0], &ny[0], &nz[0], &nm[0],
           vx + off, vy + off, vz + off, m_fl, m_fcoeff, fsrrmax);
}

// Iterate through the tree nodes, for each leaf node, start a task.
// That task iterates through the tree nodes, skipping any node (and all
// of its children) if all corners are too far away. Then it compares the
// opening angle.
template <int TDPTS>
void RCOForceTree<TDPTS>::calcInternodeForces()
{
  std::vector<ID_T> q(1, 0);
  std::vector<ID_T> parents;
  while (!q.empty()) {
    ID_T tl = q.back();

    if (tree[tl].leaf()) {
      // This is a leaf node.
      q.pop_back();

      bool inside = true;
      for (int i = 0; i < DIMENSION; ++i) {
        inside &= (tree[tl].xmax[i] < maxForceRange[i] && tree[tl].xmax[i] > minForceRange[i]) ||
                  (tree[tl].xmin[i] < maxForceRange[i] && tree[tl].xmin[i] > minForceRange[i]);
      }

      if (inside) {
        calcInternodeForce(tl, parents);
      }
    } else if (parents.size() > 0 && parents.back() == tl) {
      // This is second time here; we've done with all children.
      parents.pop_back();
      q.pop_back();
    } else {
      // This is the first time at this parent node, queue the children.
      for (int i = 0; i < NUM_CHILDREN; ++i) {
        if (tree[tl].c[i] > 0) {
          q.push_back(tree[tl].c[i]);
        }
      }

      parents.push_back(tl);
    }
  }
}

// Explicit template instantiation...
template class RCOForceTree<QUADRUPOLE_TDPTS>;
template class RCOForceTree<MONOPOLE_TDPTS>;

