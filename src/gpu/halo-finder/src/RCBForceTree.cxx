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


/*
BG/Q tuned version of HACC: 69.2% of peak performance on 96 racks of Sequoia
Argonne Leadership Computing Facility, Argonne, IL 60439
Vitali Morozov (morozov@anl.gov)
Hal Finkel (hfinkel@anl.gov)
*/


#include "hip/hip_runtime.h"
#include "Timings.h"
#include "RCBForceTree.h"
#include "Partition.h"

#include <cstring>
#include <cstdio>
#include <ctime>
#include <stdexcept>
#include <assert.h>
using namespace std;

#ifdef __HIPCC__
#include <cudaUtil.h>

#define TILEX 4                       //Unroll factor in the x dimension, best if 2 or 4, could also add 8 but that is too many registers
#define TILEY 4                       //Unroll factor in the y dimension, best if 2 or 4, could add 8 but that is too many registers
#define BLOCKX 32                     //Block size in the x dimension (should be 32)
#define BLOCKY 4                      //Block size in the y dimension
#define MAXX 32                       //Maximum blocks in the X dimension, smaller=more reuse but less parallelism
#define MAXY 256                      //Maximum blocks in the Y dimension, there isn't much reason to make this smaller

#define ALIGNX(n) ((n+TILEX-1)/TILEX*TILEX)  //Rounds an integer to align with TILEX
#define ALIGNY(n) ((n+TILEY-1)/TILEY*TILEY)  //Rounds an integer to align with TILEY

cudaDeviceSelector __selector__;
#endif

#ifdef __HIPCC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif


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
RCBForceTree<TDPTS>::RCBForceTree(
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
                         POSVEL_T r,
                         POSVEL_T oa,
                         ID_T nd,
                         ID_T ds,
                         ID_T tmin,
                         ForceLaw *fl,
                         float fcoeff,
                         POSVEL_T ppc)
{
#ifdef __HIPCC__
  //nvtxRangeId_t r0;
  //r0=nvtxRangeStartA("RCBForceTree()");
#endif
  // Extract the contiguous data block from a vector pointer
  particleCount = count;

  xx = xLoc;
  yy = yLoc;
  zz = zLoc;
  vx = xVel;
  vy = yVel;
  vz = zVel;
  mass = ms;

  numThreads=1;

  // static size for the interaction list
  #define VMAX ALIGNY(16384)
  hipHostMalloc(&nx_v, VMAX*sizeof(POSVEL_T)*numThreads);
  hipHostMalloc(&ny_v, VMAX*sizeof(POSVEL_T)*numThreads);
  hipHostMalloc(&nz_v, VMAX*sizeof(POSVEL_T)*numThreads);
  hipHostMalloc(&nm_v, VMAX*sizeof(POSVEL_T)*numThreads);
  for(int i = 0; i < VMAX*numThreads; i++) {
    nx_v[i] = 0;
    ny_v[i] = 0;
    nz_v[i] = 0;
    nm_v[i] = 0;
  }


#ifdef __HIPCC__
  int size=ALIGNY(nd);
  hipHostMalloc(&d_xx,size*sizeof(POSVEL_T)*numThreads);
  hipHostMalloc(&d_yy,size*sizeof(POSVEL_T)*numThreads);
  hipHostMalloc(&d_zz,size*sizeof(POSVEL_T)*numThreads);
  hipHostMalloc(&d_vx,size*sizeof(POSVEL_T)*numThreads);
  hipHostMalloc(&d_vy,size*sizeof(POSVEL_T)*numThreads);
  hipHostMalloc(&d_vz,size*sizeof(POSVEL_T)*numThreads);
  hipHostMalloc(&d_mass,size*sizeof(POSVEL_T)*numThreads);
  for(int i = 0; i < size*numThreads; i++) {
    d_xx[i] = 0;
    d_yy[i] = 0;
    d_zz[i] = 0;
    d_vx[i] = 0;
    d_vy[i] = 0;
    d_vz[i] = 0;
    d_mass[i] = 0;
  }

  hipHostMalloc(&d_nx_v,VMAX*sizeof(POSVEL_T)*numThreads);
  hipHostMalloc(&d_ny_v,VMAX*sizeof(POSVEL_T)*numThreads);
  hipHostMalloc(&d_nz_v,VMAX*sizeof(POSVEL_T)*numThreads);
  hipHostMalloc(&d_nm_v,VMAX*sizeof(POSVEL_T)*numThreads);
  for(int i = 0; i < VMAX*numThreads; i++) {
    d_nx_v[i] = 0;
    d_ny_v[i] = 0;
    d_nz_v[i] = 0;
    d_nm_v[i] = 0;
  }
  cudaCheckError();


  event_v=(hipEvent_t*)malloc(sizeof(hipEvent_t)*numThreads);
  stream_v=(hipStream_t*)malloc(sizeof(hipStream_t)*numThreads);
  for(int i=0;i<numThreads;i++) {
    hipEventCreate(&event_v[i]);
    hipStreamCreate(&stream_v[i]);
  }
  cudaCheckError();
#endif

  phi = phiLoc;
  id = idLoc;
  mask = maskLoc;

  particleMass = avgMass;
  fsrrmax = fsm;
  rsm = r;
  sinOpeningAngle = sinf(oa);
  tanOpeningAngle = tanf(oa);
  nDirect = nd;
  depthSafety = ds;
  taskPartMin = tmin;
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
  // log_2(particleCount/nDirect). The tree needs to have (2^depth)+1 entries.
  // To that, a safety factor is added to the depth.
  ID_T nds = (((ID_T)(particleCount/(POSVEL_T)nDirect)) << depthSafety) + 1;
  tree.reserve(nds);

  int nthreads = 1;

  timespec b_start, b_end;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b_start);
  // Create the recursive RCB tree from the particle locations
#ifdef __HIPCC__
  //nvtxRangeId_t r1;
  //r1=nvtxRangeStartA("createRCBForceTree");
#endif
  createRCBForceTree();

#ifdef __HIPCC__
  //nvtxRangeEnd(r1);
#endif

  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b_end);
  double b_time = (b_end.tv_sec - b_start.tv_sec);
  b_time += 1e-9*(b_end.tv_nsec - b_start.tv_nsec);

  printStats(b_time);

  // Interaction lists.
  inx.resize(nthreads);
  iny.resize(nthreads);
  inz.resize(nthreads);
  inm.resize(nthreads);
  iq.resize(nthreads);

#ifdef __HIPCC__
  //r1=nvtxRangeStartA("calcInternodeForces");
#endif
  calcInternodeForces();
#ifdef __HIPCC__
  //nvtxRangeEnd(r1);
#endif

#ifdef __HIPCC__
  //nvtxRangeEnd(r0);
#endif
}

template <int TDPTS>
RCBForceTree<TDPTS>::~RCBForceTree()
{
#ifdef __HIPCC__
  //nvtxRangeId_t r0;
  //r0=nvtxRangeStartA("~RCBForceTree");
#endif
  if (m_own_fl) {
    delete m_fl;
  }
#ifdef __HIPCC__
  hipHostFree(d_xx);
  hipHostFree(d_yy);
  hipHostFree(d_zz);
  hipHostFree(d_vx);
  hipHostFree(d_vy);
  hipHostFree(d_vz);
  hipHostFree(d_mass);
  hipHostFree(d_nx_v);
  hipHostFree(d_ny_v);
  hipHostFree(d_nz_v);
  hipHostFree(d_nm_v);
  cudaCheckError();

  for(int i=0;i<numThreads;i++) {
    hipEventDestroy(event_v[i]);
    hipStreamDestroy(stream_v[i]);
  }
  cudaCheckError();

  free(event_v);
  free(stream_v);

#endif
  hipHostFree(nx_v);
  hipHostFree(ny_v);
  hipHostFree(nz_v);
  hipHostFree(nm_v);
#ifdef __HIPCC__
  //nvtxRangeEnd(r0);
#endif
}

template <int TDPTS>
void RCBForceTree<TDPTS>::printStats(double buildTime)
{
  size_t zeroLeafNodes = 0;
  size_t nonzeroLeafNodes = 0;
  size_t maxPPN = 0;
  size_t leafParts = 0;

  for (ID_T tl = 1; tl < (ID_T) tree.size(); ++tl) {
    if (tree[tl].cl == 0 && tree[tl].cr == 0) {
      if (tree[tl].count > 0) {
        ++nonzeroLeafNodes;

        leafParts += tree[tl].count;
        maxPPN = std::max((size_t) tree[tl].count, maxPPN);
      } else {
        ++zeroLeafNodes;
      }
    }
  }

  double localParticleCount = particleCount;
  double localTreeSize = tree.size();
  double localTreeCapacity = tree.capacity();
  double localLeaves = zeroLeafNodes+nonzeroLeafNodes;
  double localEmptyLeaves = zeroLeafNodes;
  double localMeanPPN = leafParts/((double) nonzeroLeafNodes);
  unsigned long localMaxPPN = maxPPN;
  double localBuildTime = buildTime;

  /*
  double globalParticleCount;
  double globalTreeSize;
  double globalTreeCapacity;
  double globalLeaves;
  double globalEmptyLeaves;
  double globalMeanPPN;
  unsigned long globalMaxPPN;
  double globalBuildTime;

  bool printHere = true;
  */

  if ( Partition::getMyProc() == 0 ) {
    printf("\ttree post-build statistics (local for rank 0):\n");
    printf("\t\tparticles: %.2f\n", localParticleCount);
    printf("\t\tnodes: %.2f (allocated:  %.2f)\n", localTreeSize, localTreeCapacity);
    printf("\t\tleaves: %.2f (empty: %.2f)\n", localLeaves, localEmptyLeaves);
    printf("\t\tmean ppn: %.2f (max ppn: %lu)\n", localMeanPPN, localMaxPPN);
    printf("\t\tbuild time: %g s\n", localBuildTime);
  }
}


extern "C" void cm(ID_T count, const POSVEL_T*  xx, const POSVEL_T*  yy,
                      const POSVEL_T*  zz, const POSVEL_T*  mass,
                      POSVEL_T*  xmin, POSVEL_T*  xmax, POSVEL_T*  xc);

static inline POSVEL_T pptdr(const POSVEL_T*  xmin, const POSVEL_T*  xmax, const POSVEL_T*  xc)
{
  return std::min(xmax[0] - xc[0], std::min(xmax[1] - xc[1], std::min(xmax[2] - xc[2], std::min(xc[0] - xmin[0],
                 std::min(xc[1] - xmin[1], xc[2] - xmin[2])))));
}

template <int TDPTS>
static inline void pppts(POSVEL_T tdr, const POSVEL_T*  xc,
                         POSVEL_T*  ppx, POSVEL_T*  ppy, POSVEL_T*  ppz)
{
  for (int i = 0; i < TDPTS; ++i) {
    ppx[i] = tdr*sphdesign<TDPTS>::x[i] + xc[0];
    ppy[i] = tdr*sphdesign<TDPTS>::y[i] + xc[1];
    ppz[i] = tdr*sphdesign<TDPTS>::z[i] + xc[2];
  }
}

template <int TDPTS>
static inline void pp(ID_T count, const POSVEL_T*  xx, const POSVEL_T*  yy,
                      const POSVEL_T*  zz, const POSVEL_T*  mass, const POSVEL_T*  xc,
                      const POSVEL_T*  ppx, const POSVEL_T*  ppy, const POSVEL_T*  ppz,
                      POSVEL_T*  ppm, POSVEL_T tdr)
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

#ifdef __HIPCC__

typedef long long int int64;

template<typename T>
__device__ __forceinline__
T load(T *t)
{
  return __ldg(t);  //texture load
}

template<int TILE_SIZE, typename T>__device__ void loadT(T *  out, const T *  in);

//generic version (inefficient)
template<int TILE_SIZE, typename T>
__device__ __forceinline__
void loadT(T *  out, const T *  in) {
  #pragma unroll
  for(int i=0;i<TILE_SIZE;i++) {
    out[i]=__ldg(in+i);
  }
}

//Vector loads
template<>
__device__ __forceinline__
void loadT<2,float>(float *  out, const float *  in) {
  *reinterpret_cast<float2*>(out)=load(reinterpret_cast<const float2*>(in));
}
template<>
__device__ __forceinline__
void loadT<4,float>(float *  out, const float *  in) {
  *reinterpret_cast<float4*>(out)=load(reinterpret_cast<const float4*>(in));
}

//static __device__ __forceinline__ float __internal_fast_rsqrtf(float a)
//{
//  float r;
//  asm ("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(a));
//  return r;
//}

//computes the forces between tiles i and j, adds the change in force to xi,yi,zi
template<int TX, int TY>
__device__ __forceinline__ void computeForces(POSVEL_T xxi[], POSVEL_T yyi[], POSVEL_T zzi[],
                                              POSVEL_T xxj[], POSVEL_T yyj[], POSVEL_T zzj[], POSVEL_T massj[],
                                              POSVEL_T xi[], POSVEL_T yi[], POSVEL_T zi[],
                                              POSVEL_T ma0, POSVEL_T ma1, POSVEL_T ma2, POSVEL_T ma3, POSVEL_T ma4, POSVEL_T ma5,
                                              POSVEL_T mp_rsm2, POSVEL_T fsrrmax2) {

  #pragma unroll
  for(int i=0;i<TY;i++) {
    #pragma unroll
    for(int j=0;j<TX;j++) {
      POSVEL_T dxc = xxj[j] - xxi[i];                                                                //1 FADD
      POSVEL_T dyc = yyj[j] - yyi[i];                                                                //1 FADD
      POSVEL_T dzc = zzj[j] - zzi[i];                                                                //1 FADD

      POSVEL_T r2 = dxc * dxc + dyc * dyc + dzc * dzc;                                               //1 FMUL 2 FMA
      POSVEL_T v=r2+mp_rsm2;                                                                         //1 FADD
      POSVEL_T v3=v*v*v;                                                                             //2 FMUL

      POSVEL_T f = __frsqrt_rn(v3);                                                        //1 MUFU,
	  // MDS: Should ask someone why this line is dangling
      //       - ( ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5)))));                          //5 FMA, 1 FADD
//#define BUG
#ifndef BUG
      f*=massj[j]*(r2<fsrrmax2 && r2>0.0f);                                                          //2 FMUL, 1 FSETP, 1 FCMP
#else
      f*=massj[j];                                                                                   //1 FMUL
      f*=(r2<fsrrmax2 && r2>0.0f);                                                                   //1 FMUL, 1 FSETP, 1 FCMP
#endif

      xi[i] = xi[i] + f * dxc;                                                                       //1 FMA
      yi[i] = yi[i] + f * dyc;                                                                       //1 FMA
      zi[i] = zi[i] + f * dzc;                                                                       //1 FMA
    }
  }
}

//loads a tile from memory.  Use checkBounds and loadMass to disable bounds check or mass load at compile time
template <bool checkBounds, bool loadMass, int TILE_SIZE>
__device__ __forceinline__
void loadTile(int i, int bounds,
              const POSVEL_T*  xx, const POSVEL_T*  yy, const POSVEL_T*  zz, const POSVEL_T*  mass,
              POSVEL_T xxi[], POSVEL_T yyi[], POSVEL_T zzi[], POSVEL_T massi[]) {
  if(checkBounds) {
  #pragma unroll
  for(int64 u=0;u<TILE_SIZE;u++) {
    int64 idx=TILE_SIZE*i+u;                                                                        // 1 IMAD

#if 1
      bool cond=idx<bounds;
      xxi[u] = (cond) ? load(xx+idx) : 0.0f;                                                     // 1 ISETP, 1 LDG, 2 IMAD, 1 MOV
      yyi[u] = (cond) ? load(yy+idx) : 0.0f;                                                     // 1 ISETP, 1 LDG, 2 IMAD, 1 MOV
      zzi[u] = (cond) ? load(zz+idx) : 0.0f;                                                     // 1 ISETP, 1 LDG, 2 IMAD, 1 MOV
      if(loadMass) massi[u] = (cond) ? load(mass+idx) : 0.0f;                                    // 1 ISETP, 1 LDG, 2 IMAD, 1 MOV
#else
      massi[u] = 0.0f;                                                                           //1 MOV
      if(idx<bounds) {                                                                           //1 ISETP, 1 BRA
        xxi[u] = load(xx+idx);                                                                   //1 LDG, 2 IMAD
        yyi[u] = load(yy+idx);                                                                   //1 LDG, 2 IMAD
        zzi[u] = load(zz+idx);                                                                   //1 LDG, 2 IMAD
        if(loadMass) massi[u] = load(mass+idx);                                                  //1 LDG, 2 IMAD
      }
#endif
    }
  } else {

    int idx=TILE_SIZE*i;
    loadT<TILE_SIZE>(xxi,xx+idx);                                                                //1 LDG, 2 IMAD
    loadT<TILE_SIZE>(yyi,yy+idx);                                                                //1 LDG, 2 IMAD
    loadT<TILE_SIZE>(zzi,zz+idx);                                                                //1 LDG, 2 IMAD
    if(loadMass) loadT<TILE_SIZE>(massi,mass+idx);                                               //1 LDG, 2 IMAD
  }
}

//applies the force in xi,yi,zi to update vx, vy, vz
//use checkBounds to disable bounds checking at compile time
template <bool checkBounds, int TILE_SIZE>
__device__ __forceinline__
void applyForce(int i, int bounds,POSVEL_T fcoeff,
                const POSVEL_T xi[], const POSVEL_T yi[], const POSVEL_T zi[],
                POSVEL_T *vx, POSVEL_T *vy, POSVEL_T *vz) {
    #pragma unroll
    for(int u=0;u<TILE_SIZE;u++) {
      int idx=TILE_SIZE*i+u;                                                                         //1 IMAD

      if(!checkBounds || idx<bounds)
      {                                                                                           //1 ISETP
        atomicWarpReduceAndUpdate(vx+idx,fcoeff * xi[u]);                                         //2 IMAD, 6 FADD
        atomicWarpReduceAndUpdate(vy+idx,fcoeff * yi[u]);                                         //2 IMAD, 6 FADD
        atomicWarpReduceAndUpdate(vz+idx,fcoeff * zi[u]);                                         //2 IMAD, 6 FADD
      }
    }
}

  //Tell the compiler how many blocks we expect to be active.
  //This gives the compiler a better idea of how many registers to use.  The second number is tunable.
__launch_bounds__(BLOCKX*BLOCKY,7)
__global__
void Step10_cuda_kernel(int count, int count1,
                        const POSVEL_T*  xx, const POSVEL_T*  yy,
                        const POSVEL_T*  zz, const POSVEL_T*  mass,
                        const POSVEL_T*  xx1, const POSVEL_T*  yy1,
                        const POSVEL_T*  zz1, const POSVEL_T*  mass1,
                        POSVEL_T*  vx, POSVEL_T*  vy,
                        POSVEL_T*  vz, POSVEL_T fsrrmax2, POSVEL_T mp_rsm2, POSVEL_T fcoeff)
{
  const POSVEL_T ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313, ma4 = 0.0000605491, ma5 = -0.00000147177;

  //Register arrays to hold tiles of data
  POSVEL_T xxi[TILEY];
  POSVEL_T yyi[TILEY];
  POSVEL_T zzi[TILEY];
  POSVEL_T xxj[TILEX];
  POSVEL_T yyj[TILEX];
  POSVEL_T zzj[TILEX];
  POSVEL_T massj[TILEX];

  // Consolidate variables to help fit within the register limit
  int x_idx = hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x;
  int y_idx = hipBlockIdx_y*hipBlockDim_y+hipThreadIdx_y;

  //loop over interior region and calculate forces.
  //for each tile i
 for(int i=y_idx;i<count/TILEY;i+=hipBlockDim_y*hipGridDim_y)                                //1 ISETP
  {
    POSVEL_T xi[TILEY]={0};                                                                                //TILEY MOV
    POSVEL_T yi[TILEY]={0};                                                                                //TILEY MOV
    POSVEL_T zi[TILEY]={0};                                                                                //TILEY MOV

    //load tile i,mass and bounds check are not needed
    loadTile<false,false,TILEY>(i,count,xx,yy,zz,NULL,xxi,yyi,zzi,NULL);

    //for each tile j
    for (int j=x_idx;j<count1/TILEX;j+=hipBlockDim_x*hipGridDim_x)                                  //1 ISETP
    {
      //load tile j, bounds check is not needed
      loadTile<false,true,TILEX>(j,count1,xx1,yy1,zz1,mass1,xxj,yyj,zzj,massj);

      //compute forces between tile i and tile j
      computeForces<TILEX,TILEY>(xxi,yyi,zzi,xxj,yyj,zzj,massj,xi,yi,zi,ma0,ma1,ma2,ma3,ma4,ma5,mp_rsm2,fsrrmax2);
    }

    //process remaining elements at the end, use TILEX=1
    for (int j=count1/TILEX*TILEX+x_idx;j<count1;j+=hipBlockDim_x*hipGridDim_x)                                  //1 ISETP
    {
      //load tile j, bounds check is needed, mass is needed
      loadTile<true,true,1>(j,count1,xx1,yy1,zz1,mass1,xxj,yyj,zzj,massj);

      //compute forces between tile i and tile j
      computeForces<1,TILEY>(xxi,yyi,zzi,xxj,yyj,zzj,massj,xi,yi,zi,ma0,ma1,ma2,ma3,ma4,ma5,mp_rsm2,fsrrmax2);
    }

    //apply the force we have calculated above, bounds check is not needed
    applyForce<false,TILEY>(i,count,fcoeff,xi,yi,zi,vx,vy,vz);
  }

  //At this point we have computed almost all interactions.
  //However we still need to add contributions for particles at the end

#if 1
  //process ramining elements in set TILEY=1
  //for each tile i
  for(int i=y_idx;i<count - count/TILEY*TILEY;i+=hipBlockDim_y*hipGridDim_y)                             //1 ISETP
  {
    // Taken out of the loop condition to help fit within the register limit
    int k = i + count/TILEY*TILEY;
    POSVEL_T xi[1]={0};                                                                                //1 MOV
    POSVEL_T yi[1]={0};                                                                                //1 MOV
    POSVEL_T zi[1]={0};                                                                                //1 MOV

    //load xxi, yyi, zzi tiles, mass is not needed, bounds check is needed
    loadTile<true,false,1>(k,count,xx,yy,zz,NULL,xxi,yyi,zzi,NULL);

    //for each tile j
    for (int j=x_idx;j<count1/TILEX;j+=hipBlockDim_x*hipGridDim_x)                                  //1 ISETP
    {
      //load tile j, bounds check is not needed
      loadTile<false,true,TILEX>(j,count1,xx1,yy1,zz1,mass1,xxj,yyj,zzj,massj);

      //compute forces between tile i and tile j
      computeForces<TILEX,1>(xxi,yyi,zzi,xxj,yyj,zzj,massj,xi,yi,zi,ma0,ma1,ma2,ma3,ma4,ma5,mp_rsm2,fsrrmax2);
    }

    //process remaining elements at the end, use TILEX=1
    for (int j=count1/TILEX*TILEX+x_idx;j<count1;j+=hipBlockDim_x*hipGridDim_x)                                  //1 ISETP
    {
      //load tile j, bounds check is needed, mass is needed
      loadTile<true,true,1>(j,count1,xx1,yy1,zz1,mass1,xxj,yyj,zzj,massj);

      //compute forces between tile i and tile j
      computeForces<1,1>(xxi,yyi,zzi,xxj,yyj,zzj,massj,xi,yi,zi,ma0,ma1,ma2,ma3,ma4,ma5,mp_rsm2,fsrrmax2);
    }

    applyForce<true,1>(k,count,fcoeff,xi,yi,zi,vx,vy,vz);
  }
#endif

}



#endif

#ifdef __bgq__
extern "C" Step16_int( int count1, float xxi, float yyi, float zzi, float fsrrmax2, float mp_rsm2, const float *xx1, const float *yy1, const float *zz1,const  float *mass1, float *ax, float *ay, float *az );
#endif

static inline void nbody1(ID_T count, ID_T count1, const POSVEL_T*  xx, const POSVEL_T*  yy,
                         const POSVEL_T*  zz, const POSVEL_T*  mass,
                         const POSVEL_T*  xx1, const POSVEL_T*  yy1,
                         const POSVEL_T*  zz1, const POSVEL_T*  mass1,
                         POSVEL_T*  vx, POSVEL_T*  vy, POSVEL_T*  vz,
                         ForceLaw *fl, float fcoeff, float fsrrmax, float rsm
#ifdef __HIPCC__
                         , hipStream_t stream
#endif
                         )
{
  POSVEL_T fsrrmax2 = fsrrmax*fsrrmax;
  POSVEL_T rsm2 = rsm*rsm;

#ifdef __bgq__
  float ax = 0.0f, ay = 0.0f, az = 0.0f;

  for (int i = 0; i < count; ++i)
  {

    Step16_int ( count1, xx[i],yy[i],zz[i], fsrrmax2,rsm2,xx1,yy1,zz1,mass1, &ax, &ay, &az );

    vx[i] = vx[i] + ax * fcoeff;
    vy[i] = vy[i] + ay * fcoeff;
    vz[i] = vz[i] + az * fcoeff;
  }

#else

#ifdef __HIPCC__
  //const int MAXX=64;
  //const int MAXY=64;

  dim3 threads(BLOCKX,BLOCKY);
  int blocksX=(count1+threads.x-1)/threads.x;
  int blocksY=(count+threads.y-1)/threads.y;
  dim3 blocks( min(blocksX,MAXX), min(blocksY,MAXY));

  //call kernel

  cudaCheckError();
  //printf("count: %d, count1: %d\n", count,count1);
#if 0
  checkCudaPtr(xx,"xx");
  checkCudaPtr(yy,"yy");
  checkCudaPtr(zz,"zz");
  checkCudaPtr(mass,"mass");
  checkCudaPtr(xx1,"xx1");
  checkCudaPtr(yy1,"yy1");
  checkCudaPtr(zz1,"zz1");
  checkCudaPtr(mass1,"mass1");
  checkCudaPtr(vx,"vx");
  checkCudaPtr(vy,"vy");
  checkCudaPtr(vz,"vz");
#endif

  hipLaunchKernelGGL(Step10_cuda_kernel, dim3(blocks), dim3(threads), 0, stream, count,count1,xx,yy,zz,mass,xx1,yy1,zz1,mass1, vx, vy, vz, fsrrmax2, rsm2, fcoeff);
  cudaCheckError();

  hipStreamSynchronize(stream);
  //exit(0);
#else

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
#endif //end __HIPCC__


#endif //end __bgq__
}


static inline ID_T partition(ID_T n,
                             POSVEL_T*  xx, POSVEL_T*  yy, POSVEL_T*  zz,
                             POSVEL_T*  vx, POSVEL_T*  vy, POSVEL_T*  vz,
                             POSVEL_T*  mass, POSVEL_T*  phi,
                             ID_T*  id, MASK_T*  mask, POSVEL_T pv
                            )
{
  float t0, t1, t2, t3, t4, t5, t6, t7;
  int32_t is, i, j;
  long i0;
  uint16_t i1;

  int idx[n];

  is = 0;
  for ( i = 0; i < n; i = i + 1 )
  {
    if (xx[i] < pv)
    {
      idx[is] = i;
      is = is + 1;
    }
  }

#pragma unroll 4
  for ( j = 0; j < is; j++ )
  {
      i = idx[j];

      t6 = mass[i]; mass[i] = mass[j]; mass[j] = t6;
      t7 = phi [i]; phi [i] = phi [j]; phi [j] = t7;
      i1 = mask[i]; mask[i] = mask[j]; mask[j] = i1;
      i0 = id  [i]; id  [i] = id  [j]; id  [j] = i0;
  }

#pragma unroll 4
  for ( j = 0; j < is; j++ )
  {
      i = idx[j];

      t0 = xx[i]; xx[i] = xx[j]; xx[j] = t0;
      t1 = yy[i]; yy[i] = yy[j]; yy[j] = t1;
      t2 = zz[i]; zz[i] = zz[j]; zz[j] = t2;
      t3 = vx[i]; vx[i] = vx[j]; vx[j] = t3;
      t4 = vy[i]; vy[i] = vy[j]; vy[j] = t4;
      t5 = vz[i]; vz[i] = vz[j]; vz[j] = t5;
  }

  return is;
}

template <int TDPTS>
void RCBForceTree<TDPTS>::createRCBForceSubtree(int d, ID_T tl, ID_T tlcl, ID_T tlcr)
{
  POSVEL_T *x1, *x2, *x3;
  switch (d) {
  case 0:
    x1 = xx;
    x2 = yy;
    x3 = zz;
  break;
  case 1:
    x1 = yy;
    x2 = zz;
    x3 = xx;
  break;
  /*case 2*/ default:
    x1 = zz;
    x2 = xx;
    x3 = yy;
  break;
  }

#ifdef __bgq__
  int tid = 0;

#endif
  const bool geoSplit = false;
  POSVEL_T split = geoSplit ? (tree[tl].xmax[d]+tree[tl].xmin[d])/2 : tree[tl].xc[d];
  ID_T is = ::partition(tree[tl].count, x1 + tree[tl].offset, x2 + tree[tl].offset, x3 + tree[tl].offset,
                        vx + tree[tl].offset, vy + tree[tl].offset, vz + tree[tl].offset,
                        mass + tree[tl].offset, phi + tree[tl].offset,
                        id + tree[tl].offset, mask + tree[tl].offset, split
                       );

  if (is == 0 || is == tree[tl].count) {
    return;
  }

  tree[tlcl].count = is;
  tree[tlcr].count = tree[tl].count - tree[tlcl].count;

  if (tree[tlcl].count > 0) {
    tree[tl].cl = tlcl;
    tree[tlcl].offset = tree[tl].offset;
    tree[tlcl].xmax[d] = split;

    createRCBForceTreeInParallel(tlcl);
  }

  if (tree[tlcr].count > 0) {
    tree[tl].cr = tlcr;
    tree[tlcr].offset = tree[tl].offset + tree[tlcl].count;
    tree[tlcr].xmin[d] = split;

    createRCBForceTreeInParallel(tlcr);
  }
}

// This is basically the algorithm from (Gafton and Rosswog, 2011).
template <int TDPTS>
void RCBForceTree<TDPTS>::createRCBForceTreeInParallel(ID_T tl)
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
  ID_T tlcl, tlcr;
  {
    tlcl = tree.size();
    tlcr = tlcl+1;
    size_t newSize = tlcr+1;
    tree.resize(newSize);
  }
  memset(&tree[tlcl], 0, sizeof(TreeNode)*2);

  // Both children have similar bounding boxes to the current node (the
  // parent), so copy the bounding box here, and then overwrite the changed
  // coordinate later.
  for (int i = 0; i < DIMENSION; ++i) {
          tree[tlcl].xmin[i] = tree[tl].xmin[i];
          tree[tlcr].xmin[i] = tree[tl].xmin[i];
          tree[tlcl].xmax[i] = tree[tl].xmax[i];
          tree[tlcr].xmax[i] = tree[tl].xmax[i];
  }

  // Split the longest edge at the center of mass.
  POSVEL_T xlen[DIMENSION];
  for (int i = 0; i < DIMENSION; ++i) {
    xlen[i] = tree[tl].xmax[i] - tree[tl].xmin[i];
  }

  int d;
  if (xlen[0] > xlen[1] && xlen[0] > xlen[2]) {
        d = 0; // Split in the x direction
  }
  else if (xlen[1] > xlen[2]) {
        d = 1; // Split in the y direction
  }
  else {
        d = 2; // Split in the z direction
  }

  createRCBForceSubtree(d, tl, tlcl, tlcr);

  // Compute the pseudoparticles based on those of the children
  POSVEL_T ppx[TDPTS], ppy[TDPTS], ppz[TDPTS];
  tree[tl].tdr = ppContract*::pptdr(tree[tl].xmin, tree[tl].xmax, tree[tl].xc);
  pppts<TDPTS>(tree[tl].tdr, tree[tl].xc, ppx, ppy, ppz);
  memset(tree[tl].ppm, 0, sizeof(POSVEL_T)*TDPTS);

  if (tree[tlcl].count > 0) {
    if (tree[tlcl].count <= TDPTS) {
      ID_T offc = tree[tlcl].offset;
      pp<TDPTS>(tree[tlcl].count, xx + offc, yy + offc, zz + offc, mass + offc,
                tree[tl].xc, ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
    } else {
      POSVEL_T ppxc[TDPTS], ppyc[TDPTS], ppzc[TDPTS];
      pppts<TDPTS>(tree[tlcl].tdr, tree[tlcl].xc, ppxc, ppyc, ppzc);
      pp<TDPTS>(TDPTS, ppxc, ppyc, ppzc, tree[tlcl].ppm, tree[tl].xc,
                ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
    }
  }
  if (tree[tlcr].count > 0) {
    if (tree[tlcr].count <= TDPTS) {
      ID_T offc = tree[tlcr].offset;
      pp<TDPTS>(tree[tlcr].count, xx + offc, yy + offc, zz + offc, mass + offc,
                tree[tl].xc, ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
    } else {
      POSVEL_T ppxc[TDPTS], ppyc[TDPTS], ppzc[TDPTS];
      pppts<TDPTS>(tree[tlcr].tdr, tree[tlcr].xc, ppxc, ppyc, ppzc);
      pp<TDPTS>(TDPTS, ppxc, ppyc, ppzc, tree[tlcr].ppm, tree[tl].xc,
                ppx, ppy, ppz, tree[tl].ppm, tree[tl].tdr);
    }
  }
}

template <int TDPTS>
void RCBForceTree<TDPTS>::createRCBForceTree()
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

  createRCBForceTreeInParallel();
}


template <int TDPTS>
void RCBForceTree<TDPTS>::calcInternodeForce(ID_T tl,
                                            const std::vector<ID_T> &parents) {

  POSVEL_T fsrrmax2 = fsrrmax*fsrrmax;
  const TreeNode* tree_ = &tree[0];

  int tid = 0;

  std::vector<ID_T> &q = iq[tid];
  q.clear();
  q.push_back(0);


  POSVEL_T *nx=nx_v+tid*VMAX;
  POSVEL_T *ny=ny_v+tid*VMAX;
  POSVEL_T *nz=nz_v+tid*VMAX;
  POSVEL_T *nm=nm_v+tid*VMAX;

#ifdef __HIPCC__
  //Adjust pointers to this threads workspace
  POSVEL_T *d_nx=d_nx_v+tid*VMAX;
  POSVEL_T *d_ny=d_ny_v+tid*VMAX;
  POSVEL_T *d_nz=d_nz_v+tid*VMAX;
  POSVEL_T *d_nm=d_nm_v+tid*VMAX;
  int size=ALIGNY(nDirect);
  POSVEL_T *d_xxl=d_xx+tid*size;
  POSVEL_T *d_yyl=d_yy+tid*size;
  POSVEL_T *d_zzl=d_zz+tid*size;
  POSVEL_T *d_massl=d_mass+tid*size;
  POSVEL_T *d_vxl=d_vx+tid*size;
  POSVEL_T *d_vyl=d_vy+tid*size;
  POSVEL_T *d_vzl=d_vz+tid*size;

  hipEvent_t& event=event_v[tid];
  hipStream_t& stream=stream_v[tid];
  hipEventSynchronize(event);  //wait for transfers from previous call to finish before overwriting nx,ny,nz,nm
  cudaCheckError();
#endif

  // The interaction list.
  int SIZE = 0; // current size of these arrays

  while (!q.empty()) {
    ID_T tln = q.back();
    q.pop_back();

    // We should not interact with our own parents.
    if (tln < tl) {
      bool isParent = std::binary_search(parents.begin(), parents.end(), tln);
      if (isParent) {
        ID_T tlncr = tree_[tln].cr;
        ID_T tlncl = tree_[tln].cl;

        if (tlncl != tl && tlncl > 0 && tree_[tlncl].count > 0) {
          q.push_back(tlncl);
        }
        if (tlncr != tl && tlncr > 0 && tree_[tlncr].count > 0) {
          q.push_back(tlncr);
        }

        continue;
      }
    }

    // Is this node have a small enough opening angle to interact with?
    POSVEL_T dx = tree_[tln].xc[0] - tree_[tl].xc[0];
    POSVEL_T dy = tree_[tln].xc[1] - tree_[tl].xc[1];
    POSVEL_T dz = tree_[tln].xc[2] - tree_[tl].xc[2];
    POSVEL_T dist2 = dx*dx + dy*dy + dz*dz;

    POSVEL_T sx = tree_[tln].xmax[0]-tree_[tln].xmin[0];
    POSVEL_T sy = tree_[tln].xmax[1]-tree_[tln].xmin[1];
    POSVEL_T sz = tree_[tln].xmax[2]-tree_[tln].xmin[2];
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
        POSVEL_T x1 = (i == 0 ? tree_[tln].xmin : tree_[tln].xmax)[0] - tree_[tl].xc[0];
        POSVEL_T y1 = (j == 0 ? tree_[tln].xmin : tree_[tln].xmax)[1] - tree_[tl].xc[1];
        POSVEL_T z1 = tree_[tln].xmin[2] - tree_[tl].xc[2];

        POSVEL_T x2 = (i == 0 ? tree_[tln].xmax : tree_[tln].xmin)[0] - tree_[tl].xc[0];
        POSVEL_T y2 = (j == 0 ? tree_[tln].xmax : tree_[tln].xmin)[1] - tree_[tl].xc[1];
        POSVEL_T z2 = tree_[tln].xmax[2] - tree_[tl].xc[2];

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
      if (tree_[tln].count <= TDPTS) {
        ID_T offn = tree_[tln].offset;
        ID_T cntn = tree_[tln].count;

        int start = SIZE;
        SIZE = SIZE + cntn;
        assert( SIZE < VMAX );

        for ( int i = 0; i < cntn; ++i) {
          nx[start + i] = xx[offn + i];
          ny[start + i] = yy[offn + i];
          nz[start + i] = zz[offn + i];
          nm[start + i] = mass[offn + i];
        }

        continue;
      }

      // Interact the particles in this node with the pseudoparticles of the
      // other node.
      int start = SIZE;
      SIZE = SIZE + TDPTS;
      assert( SIZE < VMAX );

      pppts<TDPTS>(tree_[tln].tdr, tree_[tln].xc, &nx[start], &ny[start], &nz[start]);
      for ( int i = 0; i < TDPTS; ++i) {
        nm[start + i] = tree_[tln].ppm[i];
      }

      continue;
    } else if (tree_[tln].cr == 0 && tree_[tln].cl == 0) {
      // This is a leaf node with which we must interact.
      ID_T offn = tree_[tln].offset;
      ID_T cntn = tree_[tln].count;

      int start = SIZE;
      SIZE = SIZE + cntn;
      assert( SIZE < VMAX );

      for ( int i = 0; i < cntn; ++i) {
        nx[start + i] = xx[offn + i];
        ny[start + i] = yy[offn + i];
        nz[start + i] = zz[offn + i];
        nm[start + i] = mass[offn + i];
      }

      continue;
    }

    // This other node is not a leaf, but has too large an opening angle
    // for an approx. interaction: queue its children.

    ID_T tlncr = tree_[tln].cr;
    ID_T tlncl = tree_[tln].cl;

    if (tlncl > 0 && tree_[tlncl].count > 0) {
      bool close = true;
      for (int i = 0; i < DIMENSION; ++i) {
        POSVEL_T dist = 0;
        if (tree_[tl].xmax[i] < tree_[tlncl].xmin[i]) {
          dist = tree_[tlncl].xmin[i] - tree_[tl].xmax[i];
        } else if (tree_[tl].xmin[i] > tree_[tlncl].xmax[i]) {
          dist = tree_[tl].xmin[i] - tree_[tlncl].xmax[i];
        }

        if (dist > fsrrmax) {
          close = false;
          break;
        }
      }

      if (close) q.push_back(tlncl);
    }
    if (tlncr > 0 && tree_[tlncr].count > 0) {
      bool close = true;
      for (int i = 0; i < DIMENSION; ++i) {
        POSVEL_T dist = 0;
        if (tree_[tl].xmax[i] < tree_[tlncr].xmin[i]) {
          dist = tree_[tlncr].xmin[i] - tree_[tl].xmax[i];
        } else if (tree_[tl].xmin[i] > tree_[tlncr].xmax[i]) {
          dist = tree_[tl].xmin[i] - tree_[tlncr].xmax[i];
        }

        if (dist > fsrrmax) {
          close = false;
          break;
        }
      }

      if (close) q.push_back(tlncr);
    }
  }

  ID_T off = tree_[tl].offset;
  ID_T cnt = tree_[tl].count;

  // Add self interactions...
  int start = SIZE;
  SIZE = SIZE + cnt;
  assert( SIZE < VMAX );

  for ( int i = 0; i < cnt; ++i) {
    nx[start + i] = xx[off + i];
    ny[start + i] = yy[off + i];
    nz[start + i] = zz[off + i];
    nm[start + i] = mass[off + i];
  }

#ifdef __HIPCC__
  hipMemcpyAsync(d_nx,nx,sizeof(POSVEL_T)*SIZE,hipMemcpyHostToDevice,stream);
  hipMemcpyAsync(d_ny,ny,sizeof(POSVEL_T)*SIZE,hipMemcpyHostToDevice,stream);
  hipMemcpyAsync(d_nz,nz,sizeof(POSVEL_T)*SIZE,hipMemcpyHostToDevice,stream);
  hipMemcpyAsync(d_nm,nm,sizeof(POSVEL_T)*SIZE,hipMemcpyHostToDevice,stream);
  hipEventRecord(event,stream);  //mark when transfers have finished
  cudaCheckError();
  hipStreamSynchronize(stream);


  hipMemcpyAsync(d_xxl,xx+off,sizeof(POSVEL_T)*cnt,hipMemcpyHostToDevice,stream);
  hipMemcpyAsync(d_yyl,yy+off,sizeof(POSVEL_T)*cnt,hipMemcpyHostToDevice,stream);
  hipMemcpyAsync(d_zzl,zz+off,sizeof(POSVEL_T)*cnt,hipMemcpyHostToDevice,stream);
  hipMemcpyAsync(d_massl,mass+off,sizeof(POSVEL_T)*cnt,hipMemcpyHostToDevice,stream);
  hipMemcpyAsync(d_vxl,vx+off,sizeof(POSVEL_T)*cnt,hipMemcpyHostToDevice,stream);
  hipMemcpyAsync(d_vyl,vy+off,sizeof(POSVEL_T)*cnt,hipMemcpyHostToDevice,stream);
  hipMemcpyAsync(d_vzl,vz+off,sizeof(POSVEL_T)*cnt,hipMemcpyHostToDevice,stream);
  cudaCheckError();
  hipStreamSynchronize(stream);
#endif

  // Process the interaction list...
#ifdef __HIPCC__
  ::nbody1(cnt, SIZE, d_xxl, d_yyl, d_zzl, d_massl, d_nx, d_ny, d_nz, d_nm, d_vxl, d_vyl, d_vzl, m_fl, m_fcoeff, fsrrmax, rsm, stream);
  hipStreamSynchronize(stream);
#else
  ::nbody1(cnt, SIZE, xx + off, yy + off, zz + off, mass + off, nx, ny, nz, nm, vx + off, vy + off, vz + off, m_fl, m_fcoeff, fsrrmax, rsm);
#endif

#ifdef __HIPCC__
  //transfer up vx vy vz
  hipMemcpyAsync(vx+off,d_vxl,sizeof(POSVEL_T)*cnt,hipMemcpyDeviceToHost,stream);
  hipMemcpyAsync(vy+off,d_vyl,sizeof(POSVEL_T)*cnt,hipMemcpyDeviceToHost,stream);
  hipMemcpyAsync(vz+off,d_vzl,sizeof(POSVEL_T)*cnt,hipMemcpyDeviceToHost,stream);
  cudaCheckError();
  hipStreamSynchronize(stream);
#endif

}

// Iterate through the tree nodes, for each leaf node, start a task.
// That task iterates through the tree nodes, skipping any node (and all
// of its children) if all corners are too far away. Then it compares the
// opening angle.
template <int TDPTS>
void RCBForceTree<TDPTS>::calcInternodeForces()
{

  std::vector<ID_T> q(1, 0);
  std::vector<ID_T> parents;
  while (!q.empty()) {
    ID_T tl = q.back();
    if (tree[tl].cr == 0 && tree[tl].cl == 0) {
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
      if (tree[tl].cl > 0) q.push_back(tree[tl].cl);
      if (tree[tl].cr > 0) q.push_back(tree[tl].cr);
      parents.push_back(tl);
    }
  }
}

// Explicit template instantiation...
template class RCBForceTree<QUADRUPOLE_TDPTS>;
template class RCBForceTree<MONOPOLE_TDPTS>;

