#ifndef FORCELAW_H
#define FORCELAW_H

#include <stdlib.h>
#include <math.h>
#include <assert.h>

using namespace std;

class FGrid {
public:
    FGrid();
    ~FGrid(){
  };

  float fgor(float r);
  void fgor_r2_interp(int nInterp, float **r2, float **f);
  float rmax(){return m_rmax;};

 protected:
  float m_b, m_c, m_d, m_e, m_f, m_g, m_h, m_l, m_rmax;
};



class FGridEval
{
  public:
    FGridEval() {};
    virtual ~FGridEval() {};
    virtual float eval(float)  = 0;
    virtual float r2min() = 0;
    virtual float r2max() = 0;
};



class FGridEvalFit : public FGridEval
{
  public:
    FGridEvalFit(FGrid *fg);
    ~FGridEvalFit() {};
    float eval(float);
    float r2min(){return 0.0;};
    float r2max(){return m_fg->rmax()*m_fg->rmax();};

  protected:
    FGrid *m_fg;
};



class FGridEvalPoly : public FGridEval
{
public:
    FGridEvalPoly(FGrid *fg);
    ~FGridEvalPoly() {
  };
    float eval(float);
    float r2min(){return 0.0;};
    float r2max(){return m_fg->rmax()*m_fg->rmax();};

 protected:
  FGrid *m_fg;
  float m_r2min, m_r2max;
  //float m_a0, m_a1, m_a2, m_a3, m_a4, m_a5, m_a6;
  float m_a[7];
};



class FGridEvalInterp : public FGridEval
{
  public:
    FGridEvalInterp(FGrid *fg, int nInterp);
    ~FGridEvalInterp();
    float eval(float);
    float r2min(){return m_r2min;}
    float r2max(){return m_r2max;}

    int nInterp() {return m_nInterp;}
    float* r2() {return m_r2;}
    float* f() {return m_f;}

  protected:
    float *m_r2;
    float *m_f;
    float m_r2min;
    float m_r2max;
    float m_dr2;
    float m_oodr2;
    int m_nInterp;  
};



class ForceLaw
{
 public:
    ForceLaw(){};
  virtual   ~ForceLaw(){};
  virtual   float f_over_r(float r2) = 0;
};



class ForceLawNewton : public ForceLaw
{
  public:
    ~ForceLawNewton() {
    };
    float f_over_r(float r2){return 1.0/r2/sqrt(r2);}

};



class ForceLawSR : public ForceLaw
{
 public:
    ForceLawSR(FGridEval *fgore, float rsm);
    ~ForceLawSR() {
  };
    float f_over_r(float r2);

 protected:
  float m_rsm;
  float m_rsm2;
  float m_r2min;
  float m_r2max;
  FGridEval *m_fgore;
};

#endif
