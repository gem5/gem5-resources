#include "ForceLaw.h"

#define MEM_ALIGN 64

//comment this out to leave out "inline" keywords
#define INLINEQ 1

#define POLY_ORDER 6

/*
int my_posix_memalign(void **memptr, size_t alignment, size_t size) {
  int ret;

  ret = posix_memalign(memptr, alignment, size);
  assert(ret==0);

  return ret;
}
*/

//-----------------------------------------------------------------------------

  FGrid::FGrid() :
  m_b(0.72),
  m_c(0.01),
  m_d(0.27),
  m_e(0.0001),
  m_f(360.0),
  m_g(100.0),
  m_h(0.67),
  m_l(17.0),
  m_rmax(3.116326355) {
};


#ifdef INLINEQ
inline
#endif
  float FGrid::fgor(float r) {
  float f0 = m_c + 2.0/3.0*m_b*m_b*m_b;
  float r2 = r*r;
  float r4 = r2*r2;
  float r6 = r4*r2;
  float coshbr = coshf(m_b*r);
  float r3fgor = tanhf(m_b*r) - m_b*r/coshbr/coshbr
    + m_c*r*r2*(1.0 + m_d*r2)*expf(-1.0*m_d*r2)
    + m_e*r2*(m_f*r2 + m_g*r4 + m_l*r6)*expf(-1.0*m_h*r2);
  float rdiv = r + 1.0*(r<=0.0);
  float rdiv3 = rdiv*rdiv*rdiv;
  return (r3fgor/rdiv3 + (r<=0.0)*f0)*(r<=m_rmax)*(r>=0.0);
}


  void FGrid::fgor_r2_interp(int nInterp, float **r2, float **f) {
  //my_posix_memalign((void **)r2, MEM_ALIGN, nInterp*sizeof(float) );
  //my_posix_memalign((void **)f, MEM_ALIGN, nInterp*sizeof(float) );
  *r2 = (float *)malloc(nInterp*sizeof(float));
  *f = (float *)malloc(nInterp*sizeof(float));

  double dr2 = (m_rmax*m_rmax)/(nInterp-1.0);
  for(int i=0; i<nInterp; i++) {
    (*r2)[i] = i*dr2;
    (*f)[i] = fgor(sqrt(i*dr2));
  }

  return;
}

//-----------------------------------------------------------------------------

  FGridEvalFit::FGridEvalFit(FGrid *fg) {
  m_fg = fg;
}


#ifdef INLINEQ
inline
#endif
  float FGridEvalFit::eval(float r2) {
  return m_fg->fgor(sqrt(r2));
}

//-----------------------------------------------------------------------------



// 0.263729 - 0.0686285 x + 0.00882248 x^2 - 0.000592487 x^3 + 0.0000164662 x^4

// 0.269327 - 0.0750978 x + 0.0114808 x^2 - 0.00109313 x^3 + 0.0000605491 x^4 - 1.47177*10^-6 x^5

// 0.271431 - 0.0783394 x + 0.0133122 x^2 - 0.00159485 x^3 + 0.000132336 x^4 - 6.63394*10^-6 x^5 + 1.47305*10^-7 x^6

  FGridEvalPoly::FGridEvalPoly(FGrid *fg) {
  m_fg = fg;

#if POLY_ORDER == 6
  //6th order
  m_a[0] =  0.271431;
  m_a[1] = -0.0783394;
  m_a[2] =  0.0133122;
  m_a[3] = -0.00159485;
  m_a[4] =  0.000132336;
  m_a[5] = -0.00000663394;
  m_a[6] =  0.000000147305;
#endif

#if POLY_ORDER == 5
  //5th order
  m_a[0] =  0.269327;
  m_a[1] = -0.0750978;
  m_a[2] =  0.0114808;
  m_a[3] = -0.00109313;
  m_a[4] =  0.0000605491;
  m_a[5] = -0.00000147177;
  m_a[6] =  0.0;
#endif

#if POLY_ORDER == 4
  //4th order
  m_a[0] =  0.263729;
  m_a[1] = -0.0686285;
  m_a[2] =  0.00882248;
  m_a[3] = -0.000592487;
  m_a[4] =  0.0000164622;
  m_a[5] = 0.0;
  m_a[6] = 0.0;
#endif

  m_r2min = 0.0;
  m_r2max = fg->rmax()*fg->rmax();
}


#ifdef INLINEQ
inline
#endif
  float FGridEvalPoly::eval(float r2) {
  float ret=0.0;
  ret = m_a[0] + r2*(m_a[1] + r2*(m_a[2] + r2*(m_a[3] + r2*(m_a[4] + r2*(m_a[5] + r2*m_a[6])))));
  return ret*(r2 >= m_r2min)*(r2 <= m_r2max);
}

//-----------------------------------------------------------------------------

  FGridEvalInterp::FGridEvalInterp(FGrid *fg, int nInterp) {
  m_nInterp = nInterp;
  fg->fgor_r2_interp(m_nInterp, &m_r2, &m_f);
  m_r2min = m_r2[0];
  m_r2max = m_r2[m_nInterp-1];
  m_dr2 = (m_r2max - m_r2min)/(m_nInterp - 1.0);
  m_oodr2 = 1.0/m_dr2;
}


  FGridEvalInterp::~FGridEvalInterp() {
  free(m_r2);
  free(m_f);
}


#ifdef INLINEQ
inline
#endif
  float FGridEvalInterp::eval(float r2) {
  int inRange, indx;
  float inRangef;
  inRange = (r2 > m_r2min)*(r2 < m_r2max);
  inRangef = 1.0*inRange;
  indx = int((r2 - m_r2min)*m_oodr2)*inRange;
  return inRangef*(m_f[indx]+(r2-m_r2[indx])*m_oodr2*(m_f[indx+1]-m_f[indx]));
}

//-----------------------------------------------------------------------------

  ForceLawSR::ForceLawSR(FGridEval *fgore, float rsm) {
  m_rsm = rsm;
  m_rsm2 = rsm*rsm;
  m_r2min = fgore->r2min();
  m_r2max = fgore->r2max();
  m_fgore = fgore;
}


#ifdef INLINEQ
inline
#endif
  float ForceLawSR::f_over_r(float r2) {
  float ret = powf(r2 + m_rsm2, -1.5) - m_fgore->eval(r2);
  ret *= (r2>=m_r2min)*(r2<=m_r2max);
  return ret;
  //return ( powf(r2 + m_rsm2, -1.5) );
}
