//Adrian's test 2
//calculates force on particle due to sphere of particles
//can use full newton or short range force
//calculates direct particle-particle force for comparison
//newton should match theory prediction
//short range may not due to lack of gauss law

#include "hip/hip_runtime.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/time.h>

#include <cstdio>
#include <stdlib.h>
#include <string.h>

#include <ForceLaw.h>
#include <BHForceTree.h>
#include <RCBForceTree.h>
#include <Partition.h>

#include <fenv.h>
#if defined(__i386__) && defined(__SSE__)
#include <xmmintrin.h>
#endif

#include <mpi.h>

int main(int argc, char *argv[])
{
  using namespace std;

#if defined(FE_NOMASK_ENV) && !defined(__INTEL_COMPILER)
  fesetenv(FE_NOMASK_ENV);
  fedisableexcept(/* FE_OVERFLOW | */ FE_UNDERFLOW | FE_INEXACT);
#elif defined(__i386__) && defined(__SSE__)
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~(_MM_MASK_OVERFLOW|_MM_MASK_INVALID|_MM_MASK_DIV_ZERO));
#endif

#ifndef USE_SERIAL_COSMO
  MPI_Init(&argc, &argv);
#endif
  Partition::initialize();
  
  if(argc < 8) {
    fprintf(stderr,"USAGE: %s <L> <rSphere> <nSphere> <theta> <nTrials> <N|S> <seed> <bh|bhall|rcb|rcbm>\n",argv[0]);
    exit(-1);
  }

  POSVEL_T L = atof(argv[1]);
  POSVEL_T rSphere = atof(argv[2]);
  int nSphere = atoi(argv[3]);
  float m_openAngle = atof(argv[4]);  
  int trials = atoi(argv[5]);
  char* forceType = argv[6];
  long seed = atoi(argv[7]);

  int rpart = 0;
  int rcbN = 12;
  int tMin = 128;
  int useRCB = 0, bhAll = 0;
  if (argc > 8 && strncmp("rcb", argv[8], 3) == 0) {
    useRCB = argv[8][3] == 'm' ? 2 : 1;
    if (argc > 9) {
      rcbN = atoi(argv[9]);
      if (argc > 10) {
        tMin = atoi(argv[10]);
        if (argc > 11 && strcmp("r", argv[11]) == 0) {
          rpart = 1;
        }
      }
    }
  } else if (argc > 8 && strcmp("bhall", argv[8]) == 0) {
    bhAll = 1;
  }

  float m_rsm = 0.1;

  int Np = nSphere+1;

  POSVEL_T* m_xArr;
  hipHostMalloc(&m_xArr, Np*sizeof(POSVEL_T));
  POSVEL_T* m_yArr;
  hipHostMalloc(&m_yArr, Np*sizeof(POSVEL_T));
  POSVEL_T* m_zArr;
  hipHostMalloc(&m_zArr, Np*sizeof(POSVEL_T));
  POSVEL_T* m_vxArr;
  hipHostMalloc(&m_vxArr, Np*sizeof(POSVEL_T));
  POSVEL_T* m_vyArr;
  hipHostMalloc(&m_vyArr, Np*sizeof(POSVEL_T));
  POSVEL_T* m_vzArr;
  hipHostMalloc(&m_vzArr, Np*sizeof(POSVEL_T));
  POSVEL_T* m_massArr;
  hipHostMalloc(&m_massArr, Np*sizeof(POSVEL_T));
  for(int i = 0; i < Np; i++) {
        m_xArr[i] = 0;
        m_yArr[i] = 0;
        m_zArr[i] = 0;
        m_vxArr[i] = 0;
        m_vyArr[i] = 0;
        m_vzArr[i] = 0;
        m_massArr[i] = 0;
  }

  FGrid *m_fg = new FGrid();
  FGridEval *m_fgore = new FGridEvalFit(m_fg);

  ForceLaw *m_fl;
  if(forceType[0] == 'N')
    m_fl = new ForceLawNewton();
  else
    m_fl = new ForceLawSR(m_fgore, m_rsm);

  POSVEL_T xlo = 0.0, xhi = L;
  POSVEL_T ylo = 0.0, yhi = L;
  POSVEL_T zlo = 0.0, zhi = L;
  float ngltree[DIMENSION];
  ngltree[2] = ngltree[1] = ngltree[0] = L;
  float zero[DIMENSION] = {0.0, 0.0, 0.0};
  
  POSVEL_T c = 1.0;  
  float m_fsrrmax = m_fg->rmax();

  srand48(seed);  

  POSVEL_T pos_p[DIMENSION], pos_s[DIMENSION], vel_p[DIMENSION];

  int t=0;
  while(t<trials) {

    for(int i=0; i<DIMENSION; i++) {
      pos_p[i] = L*drand48();
      pos_s[i] = L*drand48();
    }

    float dx = pos_s[0]-pos_p[0];
    float dy = pos_s[1]-pos_p[1];
    float dz = pos_s[2]-pos_p[2];
    float r2 = dx*dx + dy*dy + dz*dz;
    float r = sqrt(r2);

    int inBounds = 1;
    inBounds *= (pos_s[0]>rSphere)*(pos_s[0]<L-rSphere);
    inBounds *= (pos_s[1]>rSphere)*(pos_s[1]<L-rSphere);
    inBounds *= (pos_s[2]>rSphere)*(pos_s[2]<L-rSphere);

/*
    if( (r<rSphere) || !inBounds)
      continue;
*/

    for (int i = 0; i < 3; ++i) {
      zero[i] = min(zero[i], pos_p[i]);
      ngltree[i] = max(ngltree[i], pos_p[i]);
    }

    t++;
    for (int i = 0; i < Np; ++i) {
      m_xArr[i] = 0.0;
      m_yArr[i] = 0.0;
      m_zArr[i] = 0.0;
      m_vxArr[i] = 0.0;
      m_vyArr[i] = 0.0;
      m_vzArr[i] = 0.0;
      m_massArr[i] = 1.0;
    }
    m_xArr[0] = pos_p[0];
    m_yArr[0] = pos_p[1];
    m_zArr[0] = pos_p[2];

    int p2=1;
    while(p2 < Np) {
      float sdx = 2.0*rSphere*(drand48()-0.5);
      float sdy = 2.0*rSphere*(drand48()-0.5);
      float sdz = 2.0*rSphere*(drand48()-0.5);

      if( (sdx*sdx + sdy*sdy + sdz*sdz) > rSphere*rSphere)
	continue;

      m_xArr[p2] = pos_s[0] + sdx;
      m_yArr[p2] = pos_s[1] + sdy;
      m_zArr[p2] = pos_s[2] + sdz;

      float pt[] = { m_xArr[p2], m_yArr[p2], m_zArr[p2] };
      for (int i = 0; i < 3; ++i) {
        zero[i] = min(zero[i], pt[i]);
        ngltree[i] = max(ngltree[i], pt[i]);
      }

      p2++;
    }

    //build tree
    if (useRCB) {
      POSVEL_T* m_phiArr = new POSVEL_T[Np];
      memset(m_phiArr, 0, sizeof(POSVEL_T)*Np);

      ID_T* m_idArr = new ID_T[Np];
      memset(m_idArr, 0, sizeof(ID_T)*Np);

      MASK_T* m_maskArr = new MASK_T[Np];
      memset(m_maskArr, 0, sizeof(MASK_T)*Np);

      if (useRCB == 2) {
        RCBMonopoleForceTree *sft = new RCBMonopoleForceTree(zero,
                                          ngltree,
                                          zero, ngltree,
                                          Np,
                                          m_xArr,
                                          m_yArr,
                                          m_zArr,
                                          m_vxArr,
                                          m_vyArr,
                                          m_vzArr,
                                          m_massArr,
                                          m_phiArr,
                                          m_idArr,
                                          m_maskArr,
                                          1.0,
                                          m_fsrrmax,
                                          m_rsm,
                                          m_openAngle,
                                          rcbN,
                                          2,
                                          tMin,
                                          m_fl,
                                          c);
        delete sft;
      } else {
        RCBQuadrupoleForceTree *sft = new RCBQuadrupoleForceTree(zero,
                                          ngltree,
                                          zero, ngltree,
                                          Np,
                                          m_xArr,
                                          m_yArr,
                                          m_zArr,
                                          m_vxArr,
                                          m_vyArr,
                                          m_vzArr,
                                          m_massArr,
                                          m_phiArr,
                                          m_idArr,
                                          m_maskArr,
                                          1.0,
                                          m_fsrrmax,
                                          m_rsm,
                                          m_openAngle,
                                          rcbN,
                                          2,
                                          tMin,
                                          m_fl,
                                          c);
        delete sft;
      }

      delete [] m_phiArr;
      delete [] m_idArr;
      delete [] m_maskArr;
    } else {
      BHForceTree *bhft = new BHForceTree(zero,
  					ngltree,
  					Np,
  					m_xArr,
  					m_yArr,
  					m_zArr,
  					m_vxArr,
  					m_vyArr,
  					m_vzArr,
  					m_massArr,
  					1.0,
  					m_fl,
  					c);
  
      bhft->treeForceGadgetTopDown(0, m_openAngle, m_fsrrmax);
      if (bhAll) for (int i = 1; i < Np; ++i) {
        bhft->treeForceGadgetTopDown(i, m_openAngle, m_fsrrmax);
      }
          
      //bhft->printForceValues();
      //bhft->printBHForceTree();
      
      delete bhft;
    }

    // The tree may have reordered the particles...
    ID_T pidx = 0;
    if (!rpart) {
      for (; m_xArr[pidx] != pos_p[0]; ++pidx);
    } else {
      pidx = (ID_T) (Np*drand48());
    }
    // printf("\ttest particle: (%f, %f, %f)\n", m_xArr[pidx], m_yArr[pidx], m_zArr[pidx]);

    float f0 = sqrt(m_vxArr[pidx]*m_vxArr[pidx] + 
		    m_vyArr[pidx]*m_vyArr[pidx] + 
		    m_vzArr[pidx]*m_vzArr[pidx]);

    vel_p[0] = vel_p[1] = vel_p[2] = 0.0;
    for(int i=0; i<Np; i++) {
      if (i == pidx) continue;
      float n2dx = m_xArr[i] - m_xArr[pidx];
      float n2dy = m_yArr[i] - m_yArr[pidx];
      float n2dz = m_zArr[i] - m_zArr[pidx];
      float n2r2 = n2dx*n2dx + n2dy*n2dy + n2dz*n2dz;
      float n2for = m_fl->f_over_r(n2r2);
      vel_p[0] += n2dx*n2for;
      vel_p[1] += n2dy*n2for;
      vel_p[2] += n2dz*n2for;
    }

    float f1 = sqrt(vel_p[0]*vel_p[0] +
		    vel_p[1]*vel_p[1] +
		    vel_p[2]*vel_p[2]);

    printf("%f\t%e\t%e\t%e\n",
	   r,
	   r2*f0,
	   r2*f1,
	   r2*r*m_fl->f_over_r(r2)*nSphere);
  }
  
  delete m_fl;
  delete m_fgore;
  delete m_fg;

  hipHostFree(m_xArr);
  hipHostFree(m_yArr);
  hipHostFree(m_zArr);
  hipHostFree(m_vxArr);
  hipHostFree(m_vyArr);
  hipHostFree(m_vzArr);
  hipHostFree(m_massArr);

#ifndef USE_SERIAL_COSMO
  MPI_Finalize();
#endif
  
  return 0;
}
