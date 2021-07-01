/*
BG/Q tuned version of HACC: 69.2% of peak performance on 96 racks of Sequoia
Argonne Leadership Computing Facility, Argonne, IL 60439
Vitali Morozov (morozov@anl.gov)
*/

//#undef __bgq__

#ifdef __bgq__

//#include </soft/compilers/ibmcmp-feb2012/vacpp/bg/12.1/include/builtins.h>
#include IBMCMP_BUILTINS

int isAligned(void *in);

void cm( int count, float *xx, float *yy, float *zz, float *mass, float *xmin, float *xmax, float *xc)
{
  // xmin/xmax are currently set to the whole bounding box, but this is too conservative, so we'll
  // set them based on the actual particle content.

  double x, y, z, m, w;
  double xa, ya, za, ma;
  int i, j, k;

  float x1, x2, y1, y2, z1, z2;

  vector4double xv, yv, zv, wv, dv0, dv1, dv2, dv3, dv4, dv5;
  vector4double xi0, xi1, yi0, yi1, zi0, zi1;
  vector4double xs, ys, zs, ms;

  int ALXX, ALYY, ALZZ, ALMM;

  ALXX = isAligned( (void *)xx );
  //ALYY = isAligned( (void *)yy );
  //ALZZ = isAligned( (void *)zz );
  //ALMM = isAligned( (void *)mass );


  i = 0; j = 0; k = 0;

  if ( ( ALXX == 4 ) && ( isAligned( (void *)&xx[1]) == 8  ) ) i = 3;
  if ( ( ALXX == 4 ) && ( isAligned( (void *)&xx[1]) >= 16 ) ) i = 1;
  if ( ( ALXX == 8 ) ) i = 2;

  ma = 0.; xa = 0.; ya = 0.; za = 0.;
  
  x1 = xx[0]; x2 = xx[0]; 
  y1 = yy[0]; y2 = yy[0];
  z1 = zz[0]; z2 = zz[0];

  for ( k = 0; k < i; k++ )
  {
    if ( x1 > xx[k] ) x1 = xx[k]; 
    if ( x2 < xx[k] ) x2 = xx[k]; 
    if ( y1 > yy[k] ) y1 = yy[k]; 
    if ( y2 < yy[k] ) y2 = yy[k]; 
    if ( z1 > zz[k] ) z1 = zz[k]; 
    if ( z2 < zz[k] ) z2 = zz[k]; 
    
    w = mass[k];
    xa = xa + w * xx[k];
    ya = ya + w * yy[k];
    za = za + w * zz[k];
    ma = ma + w;
  } 
  
  xi0 = vec_splats( (double)x1 );
  xi1 = vec_splats( (double)x2 );
  yi0 = vec_splats( (double)y1 );
  yi1 = vec_splats( (double)y2 );
  zi0 = vec_splats( (double)z1 );
  zi1 = vec_splats( (double)z2 );
  
  xs = vec_splats( 0. );
  ys = vec_splats( 0. );
  zs = vec_splats( 0. );
  ms = vec_splats( 0. );

  for ( i = k, j = k * 4; i < count-3; i = i + 4, j = j + 16 )
  {
    xv = vec_lda( j, xx );
    yv = vec_lda( j, yy );
    zv = vec_lda( j, zz );
    wv = vec_lda( j, mass );
    
    dv0 = vec_cmpgt( xi0, xv );
    dv1 = vec_cmplt( xi1, xv );
    dv2 = vec_cmpgt( yi0, yv );
    dv3 = vec_cmplt( yi1, yv );
    dv4 = vec_cmpgt( zi0, zv );
    dv5 = vec_cmplt( zi1, zv );
    
    xi0 = vec_sel( xi0, xv, dv0 );
    xi1 = vec_sel( xi1, xv, dv1 );
    yi0 = vec_sel( yi0, yv, dv2 );
    yi1 = vec_sel( yi1, yv, dv3 );
    zi0 = vec_sel( zi0, zv, dv4 );
    zi1 = vec_sel( zi1, zv, dv5 );
    
    xs = vec_madd( wv, xv, xs );
    ys = vec_madd( wv, yv, ys );
    zs = vec_madd( wv, zv, zs );
    ms = vec_add( ms, wv );
  }
  
  if ( i > 0 ) 
  {
      if ( x1 > xi0[0] ) x1 = xi0[0];
      if ( x1 > xi0[1] ) x1 = xi0[1];
      if ( x1 > xi0[2] ) x1 = xi0[2];
      if ( x1 > xi0[3] ) x1 = xi0[3];
    
      if ( x2 < xi1[0] ) x2 = xi1[0];
      if ( x2 < xi1[1] ) x2 = xi1[1];
      if ( x2 < xi1[2] ) x2 = xi1[2];
      if ( x2 < xi1[3] ) x2 = xi1[3];
    
      if ( y1 > yi0[0] ) y1 = yi0[0];
      if ( y1 > yi0[1] ) y1 = yi0[1];
      if ( y1 > yi0[2] ) y1 = yi0[2];
      if ( y1 > yi0[3] ) y1 = yi0[3];
    
      if ( y2 < yi1[0] ) y2 = yi1[0];
      if ( y2 < yi1[1] ) y2 = yi1[1];
      if ( y2 < yi1[2] ) y2 = yi1[2];
      if ( y2 < yi1[3] ) y2 = yi1[3];
    
      if ( z1 > zi0[0] ) z1 = zi0[0];
      if ( z1 > zi0[1] ) z1 = zi0[1];
      if ( z1 > zi0[2] ) z1 = zi0[2];
      if ( z1 > zi0[3] ) z1 = zi0[3];
    
      if ( z2 < zi1[0] ) z2 = zi1[0];
      if ( z2 < zi1[1] ) z2 = zi1[1];
      if ( z2 < zi1[2] ) z2 = zi1[2];
      if ( z2 < zi1[3] ) z2 = zi1[3];

      xa = xa + ( xs[0] + xs[1] + xs[2] + xs[3] );
      ya = ya + ( ys[0] + ys[1] + ys[2] + ys[3] );
      za = za + ( zs[0] + zs[1] + zs[2] + zs[3] );
      ma = ma + ( ms[0] + ms[1] + ms[2] + ms[3] );
  }    
  
  for ( k = i; k < count; k++ )
  {
    if ( x1 > xx[k] ) x1 = xx[k]; 
    if ( x2 < xx[k] ) x2 = xx[k]; 
    if ( y1 > yy[k] ) y1 = yy[k]; 
    if ( y2 < yy[k] ) y2 = yy[k]; 
    if ( z1 > zz[k] ) z1 = zz[k]; 
    if ( z2 < zz[k] ) z2 = zz[k]; 
    
    w = mass[k];
    xa = xa + w * xx[k];
    ya = ya + w * yy[k];
    za = za + w * zz[k];
    ma = ma + w;
  } 
  
  xmin[0] = x1; xmax[0] = x2;
  xmin[1] = y1; xmax[1] = y2;
  xmin[2] = z1; xmax[2] = z2;
  
  xc[0] = (float) ( xa / ma);
  xc[1] = (float) ( ya / ma);
  xc[2] = (float) ( za / ma);

  return;
}

#else

#include <math.h>

/*
static inline void cm(ID_T count, const POSVEL_T* __restrict xx, const POSVEL_T* __restrict yy,
                      const POSVEL_T* __restrict zz, const POSVEL_T* __restrict mass,
                      POSVEL_T* __restrict xmin, POSVEL_T* __restrict xmax, POSVEL_T* __restrict xc) 
*/

void cm( int count, float *xx, float *yy, float *zz, float *mass, float *xmin, float *xmax, float *xc)
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
      xmin[0] = fminf(xmin[0], xx[i]);
      xmax[0] = fmaxf(xmax[0], xx[i]);
      xmin[1] = fminf(xmin[1], yy[i]);
      xmax[1] = fmaxf(xmax[1], yy[i]);
      xmin[2] = fminf(xmin[2], zz[i]);
      xmax[2] = fmaxf(xmax[2], zz[i]);
    }

    float w = mass[i];
    x += w*xx[i];
    y += w*yy[i];
    z += w*zz[i];
    m += w;
  }

  xc[0] = (float) (x/m);
  xc[1] = (float) (y/m);
  xc[2] = (float) (z/m);
}

#endif

