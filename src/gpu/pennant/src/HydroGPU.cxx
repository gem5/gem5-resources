/*
 * HydroGPU.cu
 *
 *  Created on: Aug 2, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

//#define HIP_ENABLE_PRINTF
#include "HydroGPU.hh"

#include <iostream>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <hip/hip_runtime.h>
#include <utility>
#include <vector>
#include <algorithm>

#include "Memory.hh"
#include "Vec2.hh"

using namespace std;

const int CHUNK_SIZE = 64;

__constant__ int nump;
__constant__ int numz;
__constant__ int nums;
__constant__ double *dt;
__constant__ double pgamma, pssmin;
__constant__ double talfa, tssmin;
__constant__ double qgamma, q1, q2;
__constant__ double hcfl, hcflv;
__constant__ double2 vfixx, vfixy;
__constant__ int numbcx, numbcy;
__constant__ double bcx[2], bcy[2];

__constant__ int *numsbad;
__constant__ double *dtnext;
__constant__ int *idtnext;

__constant__ const int* schsfirst;
__constant__ const int* schslast;
__constant__ const int* mapsp1;
__constant__ const int* mapsp2;
__constant__ const int* mapsz;
__constant__ const int* mapss4;
__constant__ const int *mappsfirst, *mapssnext;
__constant__ const int* znump;

__constant__ double2 *px, *pxp, *px0;
__constant__ double2 *zx, *zxp;
__constant__ double2 *pu, *pu0;
__constant__ double2* pap;
__constant__ double2* ssurf;
__constant__ const double* zm;
__constant__ double *zr, *zrp;
__constant__ double *ze, *zetot;
__constant__ double *zw, *zwrate;
__constant__ double *zp, *zss;
__constant__ const double* smf;
__constant__ double *careap, *sareap, *svolp, *zareap, *zvolp;
__constant__ double *sarea, *svol, *zarea, *zvol, *zvol0;
__constant__ double *zdl, *zdu;
__constant__ double *cmaswt, *pmaswt;
__constant__ double2 *sfp, *sft, *sfq, *cftot, *pf;
__constant__ double* cevol;
__constant__ double* cdu;
__constant__ double* cdiv;
__constant__ double2* zuc;
__constant__ double2* cqe;
__constant__ double* ccos;
__constant__ double* cw;

static int *numsbadD, *idtnextD;
static double *dtnextD;
static double *dtD;

static int numschH, numpchH, numzchH;
static int *schsfirstH, *schslastH, *schzfirstH, *schzlastH;
static int *schsfirstD, *schslastD, *schzfirstD, *schzlastD;
static int *mapsp1D, *mapsp2D, *mapszD, *mapss4D, *znumpD;
static int *mapspkeyD, *mapspvalD;
static int *mappsfirstD, *mapssnextD;
static double2 *pxD, *pxpD, *px0D, *zxD, *zxpD, *puD, *pu0D, *papD,
    *ssurfD, *sfpD, *sftD, *sfqD, *cftotD, *pfD, *zucD, *cqeD;
static double *zmD, *zrD, *zrpD,
    *sareaD, *svolD, *zareaD, *zvolD, *zvol0D, *zdlD, *zduD,
    *zeD, *zetot0D, *zetotD, *zwD, *zwrateD,
    *zpD, *zssD, *smfD, *careapD, *sareapD, *svolpD, *zareapD, *zvolpD;
static double *cmaswtD, *pmaswtD;
static double *cevolD, *cduD, *cdivD, *crmuD, *ccosD, *cwD;


int checkCudaError(const hipError_t err, const char* cmd)
{
    if(err) {
        printf("CUDA error in command '%s'\n", cmd); \
        printf("Error message: %s\n", hipGetErrorString(err)); \
    }
    return err;
}

#define CHKERR(cmd) checkCudaError(cmd, #cmd)

static __device__ void advPosHalf(
        const int p,
        const double2* __restrict__ px0,
        const double2* __restrict__ pu0,
        const double dt,
        double2* __restrict__ pxp) {

    pxp[p] = px0[p] + pu0[p] * dt;

}


static __device__ void calcZoneCtrs(
        const int s,
        const int s0,
        const int z,
        const int p1,
        const double2* __restrict__ px,
        double2* __restrict__ zx,
        int dss4[CHUNK_SIZE],
        double2 ctemp2[CHUNK_SIZE]) {

    ctemp2[s0] = px[p1];
    __syncthreads();

    double2 zxtot = ctemp2[s0];
    double zct = 1.;
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        zxtot += ctemp2[sn];
        zct += 1.;
    }
    zx[z] = zxtot / zct;

}


static __device__ void calcSideVols(
    const int s,
    const int z,
    const int p1,
    const int p2,
    const double2* __restrict__ px,
    const double2* __restrict__ zx,
    double* __restrict__ sarea,
    double* __restrict__ svol)
{
    const double third = 1. / 3.;
    double sa = 0.5 * cross(px[p2] - px[p1],  zx[z] - px[p1]);
    double sv = third * sa * (px[p1].x + px[p2].x + zx[z].x);
    sarea[s] = sa;
    svol[s] = sv;

    if (sv <= 0.) atomicAdd(numsbad, 1);
}


static __device__ void calcZoneVols(
    const int s,
    const int s0,
    const int z,
    const double* __restrict__ sarea,
    const double* __restrict__ svol,
    double* __restrict__ zarea,
    double* __restrict__ zvol)
{
    // make sure all side volumes have been stored
    __syncthreads();

    double zatot = sarea[s];
    double zvtot = svol[s];
    for (int sn = mapss4[s]; sn != s; sn = mapss4[sn]) {
        zatot += sarea[sn];
        zvtot += svol[sn];
    }
    zarea[z] = zatot;
    zvol[z] = zvtot;
}


static __device__ void meshCalcCharLen(
        const int s,
        const int s0,
        const int s3,
        const int z,
        const int p1,
        const int p2,
        const int* __restrict__ znump,
        const double2* __restrict__ px,
        const double2* __restrict__ zx,
        double* __restrict__ zdl,
        int dss4[CHUNK_SIZE],
        double ctemp[CHUNK_SIZE] ) {

    double area = 0.5 * cross(px[p2] - px[p1], zx[z] - px[p1]);
    double base = length(px[p2] - px[p1]);
    double fac = (znump[z] == 3 ? 3. : 4.);
    double sdl = fac * area / base;

    ctemp[s0] = sdl;
    __syncthreads();
    double sdlmin = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        sdlmin = fmin(sdlmin, ctemp[sn]);
    }
    zdl[z] = sdlmin;
}

static __device__ void hydroCalcRho(const int z,
        const double* __restrict__ zm,
        const double* __restrict__ zvol,
        double* __restrict__ zr)
{
    zr[z] = zm[z] / zvol[z];
}


static __device__ void pgasCalcForce(
        const int s,
        const int z,
        const double* __restrict__ zp,
        const double2* __restrict__ ssurf,
        double2* __restrict__ sf) {
    sf[s] = -zp[z] * ssurf[s];
}


static __device__ void ttsCalcForce(
        const int s,
        const int z,
        const double* __restrict__ zarea,
        const double* __restrict__ zr,
        const double* __restrict__ zss,
        const double* __restrict__ sarea,
        const double* __restrict__ smf,
        const double2* __restrict__ ssurf,
        double2* __restrict__ sf) {
    double svfacinv = zarea[z] / sarea[s];
    double srho = zr[z] * smf[s] * svfacinv;
    double sstmp = fmax(zss[z], tssmin);
    sstmp = talfa * sstmp * sstmp;
    double sdp = sstmp * (srho - zr[z]);
    sf[s] = -sdp * ssurf[s];
}


// Routine number [2]  in the full algorithm
//     [2.1] Find the corner divergence
//     [2.2] Compute the cos angle for c
//     [2.3] Find the evolution factor cevol(c)
//           and the Delta u(c) = du(c)
static __device__ void qcsSetCornerDiv(
        const int s,
        const int s0,
        const int s3,
        const int z,
        const int p1,
        const int p2,
        int dss4[CHUNK_SIZE],
        double2 ctemp2[CHUNK_SIZE]) {

    // [1] Compute a zone-centered velocity
    ctemp2[s0] = pu[p1];
    __syncthreads();

    double2 zutot = ctemp2[s0];
    double zct = 1.;
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        zutot += ctemp2[sn];
        zct += 1.;
    }
    zuc[z] = zutot / zct;

    // [2] Divergence at the corner
    // Associated zone, corner, point
    const int p0 = mapsp1[s3];
    double2 up0 = pu[p1];
    double2 xp0 = pxp[p1];
    double2 up1 = 0.5 * (pu[p1] + pu[p2]);
    double2 xp1 = 0.5 * (pxp[p1] + pxp[p2]);
    double2 up2 = zuc[z];
    double2 xp2 = zxp[z];
    double2 up3 = 0.5 * (pu[p0] + pu[p1]);
    double2 xp3 = 0.5 * (pxp[p0] + pxp[p1]);

    // position, velocity diffs along diagonals
    double2 up2m0 = up2 - up0;
    double2 xp2m0 = xp2 - xp0;
    double2 up3m1 = up3 - up1;
    double2 xp3m1 = xp3 - xp1;

    // average corner-centered velocity
    double2 duav = 0.25 * (up0 + up1 + up2 + up3);

    // compute cosine angle
    double2 v1 = xp1 - xp0;
    double2 v2 = xp3 - xp0;
    double de1 = length(v1);
    double de2 = length(v2);
    double minelen = 2.0 * fmin(de1, de2);
    ccos[s] = (minelen < 1.e-12 ? 0. : dot(v1, v2) / (de1 * de2));

    // compute 2d cartesian volume of corner
    double cvolume = 0.5 * cross(xp2m0, xp3m1);
    careap[s] = cvolume;

    // compute velocity divergence of corner
    cdiv[s] = (cross(up2m0, xp3m1) - cross(up3m1, xp2m0)) /
            (2.0 * cvolume);

    // compute delta velocity
    double dv1 = length2(up2m0 - up3m1);
    double dv2 = length2(up2m0 + up3m1);
    double du = sqrt(fmax(dv1, dv2));
    cdu[s]   = (cdiv[s] < 0.0 ? du   : 0.);

    // compute evolution factor
    double2 dxx1 = 0.5 * (xp2m0 - xp3m1);
    double2 dxx2 = 0.5 * (xp2m0 + xp3m1);
    double dx1 = length(dxx1);
    double dx2 = length(dxx2);

    double test1 = fabs(dot(dxx1, duav) * dx2);
    double test2 = fabs(dot(dxx2, duav) * dx1);
    double num = (test1 > test2 ? dx1 : dx2);
    double den = (test1 > test2 ? dx2 : dx1);
    double r = num / den;
    double evol = sqrt(4.0 * cvolume * r);
    evol = fmin(evol, 2.0 * minelen);
    cevol[s] = (cdiv[s] < 0.0 ? evol : 0.);

}


// Routine number [4]  in the full algorithm CS2DQforce(...)
static __device__ void qcsSetQCnForce(
        const int s,
        const int s3,
        const int z,
        const int p1,
        const int p2) {

    const double gammap1 = qgamma + 1.0;

    // [4.1] Compute the rmu (real Kurapatenko viscous scalar)
    // Kurapatenko form of the viscosity
    double ztmp2 = q2 * 0.25 * gammap1 * cdu[s];
    double ztmp1 = q1 * zss[z];
    double zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1);
    // Compute rmu for each corner
    double rmu = zkur * zrp[z] * cevol[s];
    rmu = (cdiv[s] > 0. ? 0. : rmu);

    // [4.2] Compute the cqe for each corner
    const int p0 = mapsp1[s3];
    const double elen1 = length(pxp[p1] - pxp[p0]);
    const double elen2 = length(pxp[p2] - pxp[p1]);
    // Compute: cqe(1,2,3)=edge 1, y component (2nd), 3rd corner
    //          cqe(2,1,3)=edge 2, x component (1st)
    cqe[2 * s]     = rmu * (pu[p1] - pu[p0]) / elen1;
    cqe[2 * s + 1] = rmu * (pu[p2] - pu[p1]) / elen2;
}


// Routine number [5]  in the full algorithm CS2DQforce(...)
static __device__ void qcsSetForce(
        const int s,
        const int s4,
        const int p1,
        const int p2) {

    // [5.1] Preparation of extra variables
    double csin2 = 1. - ccos[s] * ccos[s];
    cw[s]   = ((csin2 < 1.e-4) ? 0. : careap[s] / csin2);
    ccos[s] = ((csin2 < 1.e-4) ? 0. : ccos[s]);
    __syncthreads();

    // [5.2] Set-Up the forces on corners
    const double2 x1 = pxp[p1];
    const double2 x2 = pxp[p2];
    // Edge length for c1, c2 contribution to s
    double elen = length(x1 - x2);
    sfq[s] = (cw[s] * (cqe[2*s+1] + ccos[s] * cqe[2*s]) +
             cw[s4] * (cqe[2*s4] + ccos[s4] * cqe[2*s4+1]))
            / elen;
}


// Routine number [6]  in the full algorithm
static __device__ void qcsSetVelDiff(
        const int s,
        const int s0,
        const int p1,
        const int p2,
        const int z,
        int dss4[CHUNK_SIZE],
        double ctemp[CHUNK_SIZE] ) {

    double2 dx = pxp[p2] - pxp[p1];
    double2 du = pu[p2] - pu[p1];
    double lenx = length(dx);
    double dux = dot(du, dx);
    dux = (lenx > 0. ? fabs(dux) / lenx : 0.);

    ctemp[s0] = dux;
    __syncthreads();

    double ztmp = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        ztmp = fmax(ztmp, ctemp[sn]);
    }
    __syncthreads();

    zdu[z] = q1 * zss[z] + 2. * q2 * ztmp;
}


static __device__ void qcsCalcForce(
        const int s,
        const int s0,
        const int s3,
        const int s4,
        const int z,
        const int p1,
        const int p2,
        int dss3[CHUNK_SIZE],
        int dss4[CHUNK_SIZE],
        double ctemp[CHUNK_SIZE],
        double2 ctemp2[CHUNK_SIZE]) {
    // [1] Find the right, left, top, bottom  edges to use for the
    //     limiters
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [2] Compute corner divergence and related quantities
    qcsSetCornerDiv(s, s0, s3, z, p1, p2,dss4, ctemp2);

    // [3] Find the limiters Psi(c)
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [4] Compute the Q vector (corner based)
    qcsSetQCnForce(s, s3, z, p1, p2);

    // [5] Compute the Q forces
    qcsSetForce(s, s4, p1, p2);

    ctemp2[s0] = sfp[s] + sft[s] + sfq[s];
    __syncthreads();
    cftot[s] = ctemp2[s0] - ctemp2[s0 + dss3[s0]];

    // [6] Set velocity difference to use to compute timestep
    qcsSetVelDiff(s, s0, p1, p2, z, dss4, ctemp);

}


static __device__ void calcCrnrMass(
    const int s,
    const int s3,
    const int z,
    const double* __restrict__ zr,
    const double* __restrict__ zarea,
    const double* __restrict__ smf,
    double* __restrict__ cmaswt)
{
    double m = zr[z] * zarea[z] * 0.5 * (smf[s] + smf[s3]);
    cmaswt[s] = m;
}


static __device__ void pgasCalcEOS(
    const int z,
    const double* __restrict__ zr,
    const double* __restrict__ ze,
    double* __restrict__ zp,
    double& zper,
    double* __restrict__ zss)
{
    const double gm1 = pgamma - 1.;
    const double ss2 = fmax(pssmin * pssmin, 1.e-99);

    double rx = zr[z];
    double ex = fmax(ze[z], 0.0);
    double px = gm1 * rx * ex;
    double prex = gm1 * ex;
    double perx = gm1 * rx;
    double csqd = fmax(ss2, prex + perx * px / (rx * rx));
    zp[z] = px;
    zper = perx;
    zss[z] = sqrt(csqd);
}


static __device__ void pgasCalcStateAtHalf(
    const int z,
    const double* __restrict__ zr0,
    const double* __restrict__ zvolp,
    const double* __restrict__ zvol0,
    const double* __restrict__ ze,
    const double* __restrict__ zwrate,
    const double* __restrict__ zm,
    const double dt,
    double* __restrict__ zp,
    double* __restrict__ zss)
{
    double zper;
    pgasCalcEOS(z, zr0, ze, zp, zper, zss);

    const double dth = 0.5 * dt;
    const double zminv = 1. / zm[z];
    double dv = (zvolp[z] - zvol0[z]) * zminv;
    double bulk = zr0[z] * zss[z] * zss[z];
    double denom = 1. + 0.5 * zper * dv;
    double src = zwrate[z] * dth * zminv;
    zp[z] += (zper * src - zr0[z] * bulk * dv) / denom;
}


__global__ void gpuInvMap(
        const int* mapspkey,
        const int* mapspval,
        int* mappsfirst,
        int* mapssnext)
{
    const int i = hipBlockIdx_x * CHUNK_SIZE + hipThreadIdx_x;
    if (i >= nums) return;

    int p = mapspkey[i];
    int pp = mapspkey[i+1];
    int pm = i == 0 ? -1 : mapspkey[i-1];
    int s = mapspval[i];
    int sp = mapspval[i+1];

    if (i == 0 || p != pm)  mappsfirst[p] = s;
    if (i+1 == nums || p != pp)
        mapssnext[s] = -1;
    else
        mapssnext[s] = sp;

}


static __device__ void gatherToPoints(
        const int p,
        const double* __restrict__ cvar,
        double* __restrict__ pvar)
{
    double x = 0.;
    for (int s = mappsfirst[p]; s >= 0; s = mapssnext[s]) {
        x += cvar[s];
    }
    pvar[p] = x;
}


static __device__ void gatherToPoints(
        const int p,
        const double2* __restrict__ cvar,
        double2* __restrict__ pvar)
{
    double2 x = make_double2(0., 0.);
    for (int s = mappsfirst[p]; s >= 0; s = mapssnext[s]) {
        x += cvar[s];
    }
    pvar[p] = x;
}


static __device__ void applyFixedBC(
        const int p,
        const double2* __restrict__ px,
        double2* __restrict__ pu,
        double2* __restrict__ pf,
        const double2 vfix,
        const double bcconst) {

    const double eps = 1.e-12;
    double dp = dot(px[p], vfix);

    if (fabs(dp - bcconst) < eps) {
        pu[p] = project(pu[p], vfix);
        pf[p] = project(pf[p], vfix);
    }

}


static __device__ void calcAccel(
        const int p,
        const double2* __restrict__ pf,
        const double* __restrict__ pmass,
        double2* __restrict__ pa) {

    const double fuzz = 1.e-99;
    pa[p] = pf[p] / fmax(pmass[p], fuzz);

}


static __device__ void advPosFull(
        const int p,
        const double2* __restrict__ px0,
        const double2* __restrict__ pu0,
        const double2* __restrict__ pa,
        const double dt,
        double2* __restrict__ px,
        double2* __restrict__ pu) {

    pu[p] = pu0[p] + pa[p] * dt;
    px[p] = px0[p] + 0.5 * (pu[p] + pu0[p]) * dt;

}


static __device__ void hydroCalcWork(
        const int s,
        const int s0,
        const int s3,
        const int z,
        const int p1,
        const int p2,
        const double2* __restrict__ sf,
        const double2* __restrict__ sf2,
        const double2* __restrict__ pu0,
        const double2* __restrict__ pu,
        const double2* __restrict__ px,
        const double dt,
        double* __restrict__ zw,
        double* __restrict__ zetot,
        int dss4[CHUNK_SIZE],
        double ctemp[CHUNK_SIZE]) {

    // Compute the work done by finding, for each element/node pair
    //   dwork= force * vavg
    // where force is the force of the element on the node
    // and vavg is the average velocity of the node over the time period

    double sd1 = dot( (sf[s] + sf2[s]), (pu0[p1] + pu[p1]));
    double sd2 = dot(-(sf[s] + sf2[s]), (pu0[p2] + pu[p2]));
    double dwork = -0.5 * dt * (sd1 * px[p1].x + sd2 * px[p2].x);

    ctemp[s0] = dwork;
    double etot = zetot[z];
    __syncthreads();

    double dwtot = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        dwtot += ctemp[sn];
    }
    zetot[z] = etot + dwtot;
    zw[z] = dwtot;

}


static __device__ void hydroCalcWorkRate(
        const int z,
        const double* __restrict__ zvol0,
        const double* __restrict__ zvol,
        const double* __restrict__ zw,
        const double* __restrict__ zp,
        const double dt,
        double* __restrict__ zwrate) {

    double dvol = zvol[z] - zvol0[z];
    zwrate[z] = (zw[z] + zp[z] * dvol) / dt;

}


static __device__ void hydroCalcEnergy(
        const int z,
        const double* __restrict__ zetot,
        const double* __restrict__ zm,
        double* __restrict__ ze) {

    const double fuzz = 1.e-99;
    ze[z] = zetot[z] / (zm[z] + fuzz);

}


static __device__ void hydroCalcDtCourant(
        const int z,
        const double* __restrict__ zdu,
        const double* __restrict__ zss,
        const double* __restrict__ zdl,
        double& dtz,
        int& idtz) {

    const double fuzz = 1.e-99;
    double cdu = fmax(zdu[z], fmax(zss[z], fuzz));
    double dtzcour = zdl[z] * hcfl / cdu;
    dtz = dtzcour;
    idtz = z << 1;

}


static __device__ void hydroCalcDtVolume(
        const int z,
        const double* __restrict__ zvol,
        const double* __restrict__ zvol0,
        const double dtlast,
        double& dtz,
        int& idtz) {

    double zdvov = fabs((zvol[z] - zvol0[z]) / zvol0[z]);
    double dtzvol = dtlast * hcflv / zdvov;

    if (dtzvol < dtz) {
        dtz = dtzvol;
        idtz = (z << 1) | 1;
    }

}

static __device__ double atomicMin(double* address, double val)
{
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        // __longlong_as_double may be broken depending on the ROCm version
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(fmin(val,
                __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}


static __device__ void hydroFindMinDt(
        const int z,
        const int z0,
        const int zlength,
        const double dtz,
        const int idtz,
        double& dtnext,
        int& idtnext,
        double ctemp[CHUNK_SIZE],
        double2 ctemp2[CHUNK_SIZE]) {

    int* ctempi = (int*) ctemp2;

    ctemp[z0] = dtz;
    ctempi[z0] = idtz;
    __syncthreads();

    int len = zlength;
    int half = len >> 1;
    while (z0 < half) {
        len = half + (len & 1);
        if (ctemp[z0+len] < ctemp[z0]) {
            ctemp[z0]  = ctemp[z0+len];
            ctempi[z0] = ctempi[z0+len];
        }
        __syncthreads();
        half = len >> 1;
    }
    if (z0 == 0 && ctemp[0] < dtnext) {
        atomicMin(&dtnext, ctemp[0]);
        // This line isn't 100% thread-safe, but since it is only for
        // a debugging aid, I'm not going to worry about it.
        if (dtnext == ctemp[0]) idtnext = ctempi[0];
    }
}


static __device__ void hydroCalcDt(
        const int z,
        const int z0,
        const int zlength,
        const double* __restrict__ zdu,
        const double* __restrict__ zss,
        const double* __restrict__ zdl,
        const double* __restrict__ zvol,
        const double* __restrict__ zvol0,
        const double dtlast,
        double& dtnext,
        int& idtnext,
        double ctemp[CHUNK_SIZE],
        double2 ctemp2[CHUNK_SIZE]) {

    double dtz;
    int idtz;
    hydroCalcDtCourant(z, zdu, zss, zdl, dtz, idtz);
    hydroCalcDtVolume(z, zvol, zvol0, dt[0], dtz, idtz);
    hydroFindMinDt(z, z0, zlength, dtz, idtz, dtnext, idtnext, ctemp, ctemp2);

}


__global__ void gpuMain1(int dummy)
{
    const int p = hipBlockIdx_x * CHUNK_SIZE + hipThreadIdx_x;
    if (p >= nump) return;

    double dth = 0.5 * dt[0];

    // save off point variable values from previous cycle
    px0[p] = px[p];
    pu0[p] = pu[p];

    // ===== Predictor step =====
    // 1. advance mesh to center of time step
    advPosHalf(p, px0, pu0, dth, pxp);

}


__global__ void gpuMain2(int dummy)
{
    const int s0 = hipThreadIdx_x;
    const int sch = hipBlockIdx_x;
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int p1 = mapsp1[s];
    const int p2 = mapsp2[s];
    const int z  = mapsz[s];

    const int s4 = mapss4[s];
    const int s04 = s4 - schsfirst[sch];

    __shared__ int dss3[CHUNK_SIZE];
    __shared__ int dss4[CHUNK_SIZE];
    __shared__ double ctemp[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];

    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    __syncthreads();

    const int s3 = s + dss3[s0];

    // save off zone variable values from previous cycle
    zvol0[z] = zvol[z];

    // 1a. compute new mesh geometry
    calcZoneCtrs(s, s0, z, p1, pxp, zxp, dss4, ctemp2);
    meshCalcCharLen(s, s0, s3, z, p1, p2, znump, pxp, zxp, zdl, dss4, ctemp);

    ssurf[s] = rotateCCW(0.5 * (pxp[p1] + pxp[p2]) - zxp[z]);

    calcSideVols(s, z, p1, p2, pxp, zxp, sareap, svolp);
    calcZoneVols(s, s0, z, sareap, svolp, zareap, zvolp);

    // 2. compute corner masses
    hydroCalcRho(z, zm, zvolp, zrp);
    calcCrnrMass(s, s3, z, zrp, zareap, smf, cmaswt);

    // 3. compute material state (half-advanced)
    // call this routine from only one thread per zone
    if (s3 > s) pgasCalcStateAtHalf(z, zr, zvolp, zvol0, ze, zwrate,
            zm, dt[0], zp, zss);
    __syncthreads();

    // 4. compute forces
    pgasCalcForce(s, z, zp, ssurf, sfp);
    ttsCalcForce(s, z, zareap, zrp, zss, sareap, smf, ssurf, sft);
    qcsCalcForce(s, s0, s3, s4, z, p1, p2, dss3, dss4, ctemp, ctemp2);

}


__global__ void gpuMain3(int dummy)
{
    const int p = hipBlockIdx_x * CHUNK_SIZE + hipThreadIdx_x;
    if (p >= nump) return;

    // gather corner masses, forces to points
    gatherToPoints(p, cmaswt, pmaswt);
    gatherToPoints(p, cftot, pf);

    // 4a. apply boundary conditions
    for (int bc = 0; bc < numbcx; ++bc)
        applyFixedBC(p, pxp, pu0, pf, vfixx, bcx[bc]);
    for (int bc = 0; bc < numbcy; ++bc)
        applyFixedBC(p, pxp, pu0, pf, vfixy, bcy[bc]);

    // 5. compute accelerations
    calcAccel(p, pf, pmaswt, pap);

    // ===== Corrector step =====
    // 6. advance mesh to end of time step
    advPosFull(p, px0, pu0, pap, dt[0], px, pu);

}


__global__ void gpuMain4(int dummy)
{
    const int s0 = hipThreadIdx_x;
    const int sch = hipBlockIdx_x;
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int p1 = mapsp1[s];
    const int p2 = mapsp2[s];
    const int z  = mapsz[s];

    const int s4 = mapss4[s];
    const int s04 = s4 - schsfirst[sch];

    __shared__ int dss3[CHUNK_SIZE];
    __shared__ int dss4[CHUNK_SIZE];
    __shared__ double ctemp[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];

    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    __syncthreads();

    const int s3 = s + dss3[s0];

    // 6a. compute new mesh geometry
    calcZoneCtrs(s, s0, z, p1, px, zx, dss4, ctemp2);
    calcSideVols(s, z, p1, p2, px, zx, sarea, svol);
    calcZoneVols(s, s0, z, sarea, svol, zarea, zvol);

    // 7. compute work
    hydroCalcWork(s, s0, s3, z, p1, p2, sfp, sfq, pu0, pu, pxp, dt[0],
                  zw, zetot, dss4, ctemp);

}


__global__ void gpuMain5(int dummy)
{
    const int z = hipBlockIdx_x * CHUNK_SIZE + hipThreadIdx_x;
    if (z >= numz) return;

    const int z0 = hipThreadIdx_x;
    const int zlength = min((int)CHUNK_SIZE, (int)(numz - hipBlockIdx_x * CHUNK_SIZE));

    __shared__ double ctemp[CHUNK_SIZE];
    __shared__ double2 ctemp2[CHUNK_SIZE];

    // 7. compute work
    hydroCalcWorkRate(z, zvol0, zvol, zw, zp, dt[0], zwrate);

    // 8. update state variables
    hydroCalcEnergy(z, zetot, zm, ze);
    hydroCalcRho(z, zm, zvol, zr);

    // 9.  compute timestep for next cycle
    hydroCalcDt(z, z0, zlength, zdu, zss, zdl, zvol, zvol0, dt[0],
                dtnext[0], idtnext[0], ctemp, ctemp2);

}


void meshCheckBadSides() {

    int numsbadH;
    CHKERR(hipMemcpy(&numsbadH, numsbadD, sizeof(int), hipMemcpyDeviceToHost));
    // if there were negative side volumes, error exit
    if (numsbadH > 0) {
        cerr << "Error: " << numsbadH << " negative side volumes" << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

}


void computeChunks(
        const int nums,
        const int numz,
        const int* mapsz,
        const int chunksize,
        int& numsch,
        int*& schsfirst,
        int*& schslast,
        int*& schzfirst,
        int*& schzlast) {

    int* stemp1 = Memory::alloc<int>(nums/3+1);
    int* stemp2 = Memory::alloc<int>(nums/3+1);
    int* ztemp1 = Memory::alloc<int>(nums/3+1);
    int* ztemp2 = Memory::alloc<int>(nums/3+1);

    int nsch = 0;
    int s1;
    int s2 = 0;
    while (s2 < nums) {
        s1 = s2;
        s2 = min(s2 + chunksize, nums);
        if (s2 < nums) {
            while (mapsz[s2] == mapsz[s2-1]) --s2;
        }
        stemp1[nsch] = s1;
        stemp2[nsch] = s2;
        ztemp1[nsch] = mapsz[s1];
        ztemp2[nsch] = (s2 == nums ? numz : mapsz[s2]);
        ++nsch;
    }

    numsch = nsch;
    schsfirst = Memory::alloc<int>(numsch);
    schslast  = Memory::alloc<int>(numsch);
    schzfirst = Memory::alloc<int>(numsch);
    schzlast  = Memory::alloc<int>(numsch);
    copy(stemp1, stemp1 + numsch, schsfirst);
    copy(stemp2, stemp2 + numsch, schslast);
    copy(ztemp1, ztemp1 + numsch, schzfirst);
    copy(ztemp2, ztemp2 + numsch, schzlast);

    Memory::free(stemp1);
    Memory::free(stemp2);
    Memory::free(ztemp1);
    Memory::free(ztemp2);

}


void hydroInit(
        const int numpH,
        const int numzH,
        const int numsH,
        const int numcH,
        const int numeH,
        const double pgammaH,
        const double pssminH,
        const double talfaH,
        const double tssminH,
        const double qgammaH,
        const double q1H,
        const double q2H,
        const double hcflH,
        const double hcflvH,
        const int numbcxH,
        const double* bcxH,
        const int numbcyH,
        const double* bcyH,
        const double2* pxH,
        const double2* puH,
        const double* zmH,
        const double* zrH,
        const double* zvolH,
        const double* zeH,
        const double* zetotH,
        const double* zwrateH,
        const double* smfH,
        const int* mapsp1H,
        const int* mapsp2H,
        const int* mapszH,
        const int* mapss4H,
        const int* mapseH,
        const int* znumpH) {

    printf("Running Hydro on device...\n");

    computeChunks(numsH, numzH, mapszH, CHUNK_SIZE, numschH,
            schsfirstH, schslastH, schzfirstH, schzlastH);
    numpchH = (numpH+CHUNK_SIZE-1) / CHUNK_SIZE;
    numzchH = (numzH+CHUNK_SIZE-1) / CHUNK_SIZE;

    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(nump), &numpH, sizeof(int)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(numz), &numzH, sizeof(int)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(nums), &numsH, sizeof(int)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pgamma), &pgammaH, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pssmin), &pssminH, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(talfa), &talfaH, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(tssmin), &tssminH, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(qgamma), &qgammaH, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(q1), &q1H, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(q2), &q2H, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(hcfl), &hcflH, sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(hcflv), &hcflvH, sizeof(double)));

    const double2 vfixxH = make_double2(1., 0.);
    const double2 vfixyH = make_double2(0., 1.);
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(vfixx), &vfixxH, sizeof(double2)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(vfixy), &vfixyH, sizeof(double2)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(numbcx), &numbcxH, sizeof(int)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(numbcy), &numbcyH, sizeof(int)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(bcx), bcxH, numbcxH*sizeof(double)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(bcy), bcyH, numbcyH*sizeof(double)));


    CHKERR(hipMalloc(&schsfirstD, numschH*sizeof(int)));
    CHKERR(hipMalloc(&schslastD, numschH*sizeof(int)));
    CHKERR(hipMalloc(&schzfirstD, numschH*sizeof(int)));
    CHKERR(hipMalloc(&schzlastD, numschH*sizeof(int)));
    CHKERR(hipMalloc(&mapsp1D, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mapsp2D, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mapszD, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mapss4D, numsH*sizeof(int)));
    CHKERR(hipMalloc(&znumpD, numzH*sizeof(int)));

    CHKERR(hipMalloc(&pxD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&pxpD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&px0D, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&zxD, numzH*sizeof(double2)));
    CHKERR(hipMalloc(&zxpD, numzH*sizeof(double2)));
    CHKERR(hipMalloc(&puD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&pu0D, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&papD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&ssurfD, numsH*sizeof(double2)));
    CHKERR(hipMalloc(&zmD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zrD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zrpD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&sareaD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&svolD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&zareaD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zvolD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zvol0D, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zdlD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zduD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zeD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zetot0D, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zetotD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zwD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zwrateD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zpD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zssD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&smfD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&careapD, numcH*sizeof(double)));
    CHKERR(hipMalloc(&sareapD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&svolpD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&zareapD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&zvolpD, numzH*sizeof(double)));
    CHKERR(hipMalloc(&cmaswtD, numsH*sizeof(double)));
    CHKERR(hipMalloc(&pmaswtD, numpH*sizeof(double)));
    CHKERR(hipMalloc(&sfpD, numsH*sizeof(double2)));
    CHKERR(hipMalloc(&sftD, numsH*sizeof(double2)));
    CHKERR(hipMalloc(&sfqD, numsH*sizeof(double2)));
    CHKERR(hipMalloc(&cftotD, numcH*sizeof(double2)));
    CHKERR(hipMalloc(&pfD, numpH*sizeof(double2)));
    CHKERR(hipMalloc(&cevolD, numcH*sizeof(double)));
    CHKERR(hipMalloc(&cduD, numcH*sizeof(double)));
    CHKERR(hipMalloc(&cdivD, numcH*sizeof(double)));
    CHKERR(hipMalloc(&zucD, numzH*sizeof(double2)));
    CHKERR(hipMalloc(&crmuD, numcH*sizeof(double)));
    CHKERR(hipMalloc(&cqeD, 2*numcH*sizeof(double2)));
    CHKERR(hipMalloc(&ccosD, numcH*sizeof(double)));
    CHKERR(hipMalloc(&cwD, numcH*sizeof(double)));

    CHKERR(hipMalloc(&mapspkeyD, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mapspvalD, numsH*sizeof(int)));
    CHKERR(hipMalloc(&mappsfirstD, numpH*sizeof(int)));
    CHKERR(hipMalloc(&mapssnextD, numsH*sizeof(int)));


    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(schsfirst), &schsfirstD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(schslast), &schslastD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapsp1), &mapsp1D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapsp2), &mapsp2D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapsz), &mapszD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapss4), &mapss4D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mappsfirst), &mappsfirstD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(mapssnext), &mapssnextD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(znump), &znumpD, sizeof(void*)));

    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(px), &pxD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pxp), &pxpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(px0), &px0D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zx), &zxD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zxp), &zxpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pu), &puD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pu0), &pu0D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pap), &papD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(ssurf), &ssurfD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zm), &zmD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zr), &zrD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zrp), &zrpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(sarea), &sareaD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(svol), &svolD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zarea), &zareaD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvol), &zvolD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvol0), &zvol0D, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zdl), &zdlD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zdu), &zduD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(ze), &zeD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zetot), &zetotD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zw), &zwD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zwrate), &zwrateD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zp), &zpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zss), &zssD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(smf), &smfD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(careap), &careapD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(sareap), &sareapD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(svolp), &svolpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zareap), &zareapD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zvolp), &zvolpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cmaswt), &cmaswtD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pmaswt), &pmaswtD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(sfp), &sfpD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(sft), &sftD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(sfq), &sfqD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cftot), &cftotD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(pf), &pfD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cevol), &cevolD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cdu), &cduD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cdiv), &cdivD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(zuc), &zucD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cqe), &cqeD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(ccos), &ccosD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(cw), &cwD, sizeof(void*)));


    CHKERR(hipMemcpy(schsfirstD, schsfirstH, numschH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(schslastD, schslastH, numschH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(schzfirstD, schzfirstH, numschH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(schzlastD, schzlastH, numschH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(mapsp1D, mapsp1H, numsH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(mapsp2D, mapsp2H, numsH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(mapszD, mapszH, numsH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(mapss4D, mapss4H, numsH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(znumpD, znumpH, numzH*sizeof(int), hipMemcpyHostToDevice));

    CHKERR(hipMemcpy(zmD, zmH, numzH*sizeof(double), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(smfD, smfH, numsH*sizeof(double), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(pxD, pxH, numpH*sizeof(double2), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(puD, puH, numpH*sizeof(double2), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(zrD, zrH, numzH*sizeof(double), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(zvolD, zvolH, numzH*sizeof(double), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(zeD, zeH, numzH*sizeof(double), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(zetotD, zetotH, numzH*sizeof(double), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(zwrateD, zwrateH, numzH*sizeof(double), hipMemcpyHostToDevice));

    int *mapspkeyH, *mapspvalH;

    mapspkeyH = (int *)malloc(numsH*sizeof(int));
    mapspvalH = (int *)malloc(numsH*sizeof(int));

    std::vector<std::pair<int, int>> keyValuePair;
    for (int i = 0; i < numsH; i++) {
        keyValuePair.push_back(std::make_pair(mapsp1H[i], i));
    }

    std::stable_sort(keyValuePair.begin(), keyValuePair.end(), [](const std::pair<int, int> &lhs,
                                                                  const std::pair<int, int> &rhs)
    {
        return lhs.first < rhs.first;
    });

    for (int i = 0; i < numsH; i++){
        mapspkeyH[i] = keyValuePair[i].first;
        mapspvalH[i] = keyValuePair[i].second;
    }

    CHKERR(hipMemcpy(mapspkeyD, mapspkeyH, numsH*sizeof(int), hipMemcpyHostToDevice));
    CHKERR(hipMemcpy(mapspvalD, mapspvalH, numsH*sizeof(int), hipMemcpyHostToDevice));

    int gridSize = (numsH+CHUNK_SIZE - 1)/ CHUNK_SIZE;
    int chunkSize = CHUNK_SIZE;
    hipLaunchKernelGGL(gpuInvMap, dim3(gridSize), dim3(chunkSize), 0, 0, mapspkeyD, mapspvalD,
        mappsfirstD, mapssnextD);
    hipDeviceSynchronize();

    CHKERR(hipMalloc(&numsbadD, sizeof(int)));
    CHKERR(hipMalloc(&idtnextD, sizeof(int)));
    CHKERR(hipMalloc(&dtnextD, sizeof(double)));
    CHKERR(hipMalloc(&dtD, sizeof(double)));

    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(numsbad), &numsbadD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(idtnext), &idtnextD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(dtnext), &dtnextD, sizeof(void*)));
    CHKERR(hipMemcpyToSymbol(HIP_SYMBOL(dt), &dtD, sizeof(void*)));

    int zero = 0;
    CHKERR(hipMemcpy(numsbadD, &zero, sizeof(int), hipMemcpyHostToDevice));
}


void hydroDoCycle(
        const double dtH,
        double& dtnextH,
        int& idtnextH) {
    int gridSizeS, gridSizeP, gridSizeZ, chunkSize;

    CHKERR(hipMemcpy(dtD, &dtH, sizeof(double), hipMemcpyHostToDevice));

    gridSizeS = numschH;
    gridSizeP = numpchH;
    gridSizeZ = numzchH;
    chunkSize = CHUNK_SIZE;

    hipLaunchKernelGGL(gpuMain1, dim3(gridSizeP), dim3(chunkSize), 0, 0, 0);
    hipDeviceSynchronize();

    hipLaunchKernelGGL(gpuMain2, dim3(gridSizeS), dim3(chunkSize), 0, 0, 0);
    hipDeviceSynchronize();
    meshCheckBadSides();

    hipLaunchKernelGGL(gpuMain3, dim3(gridSizeP), dim3(chunkSize), 0, 0, 0);
    hipDeviceSynchronize();

    double bigval = 1.e99;
    CHKERR(hipMemcpy(dtnextD, &bigval, sizeof(double), hipMemcpyHostToDevice));


    hipLaunchKernelGGL(gpuMain4, dim3(gridSizeS), dim3(chunkSize), 0, 0, 0);
    hipDeviceSynchronize();

    hipLaunchKernelGGL(gpuMain5, dim3(gridSizeZ), dim3(chunkSize), 0, 0, 0);
    hipDeviceSynchronize();
    meshCheckBadSides();

    CHKERR(hipMemcpy(&dtnextH, dtnextD, sizeof(double), hipMemcpyDeviceToHost));
    CHKERR(hipMemcpy(&idtnextH, idtnextD, sizeof(int), hipMemcpyDeviceToHost));
}


void hydroGetData(
        const int numpH,
        const int numzH,
        double2* pxH,
        double* zrH,
        double* zeH,
        double* zpH) {

    CHKERR(hipMemcpy(pxH, pxD, numpH*sizeof(double2), hipMemcpyDeviceToHost));
    CHKERR(hipMemcpy(zrH, zrD, numzH*sizeof(double), hipMemcpyDeviceToHost));
    CHKERR(hipMemcpy(zeH, zeD, numzH*sizeof(double), hipMemcpyDeviceToHost));
    CHKERR(hipMemcpy(zpH, zpD, numzH*sizeof(double), hipMemcpyDeviceToHost));

}


void hydroInitGPU()
{
    int one = 1;

    CHKERR(hipDeviceSetCacheConfig(hipFuncCachePreferL1));

}


void hydroFinalGPU()
{
}

