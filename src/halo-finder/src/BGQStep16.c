/*
BG/Q tuned version of short force evaluation kernel: 81% of peak performance
Argonne Leadership Computing Facility, Argonne, IL 60439
Vitali Morozov (morozov@anl.gov)
*/


#include <assert.h>
#include <math.h>
#include <stdlib.h>
//#include <spi/include/l1p/sprefetch.h>
//#include </opt/ibmcmp/vacpp/bg/12.1/include/builtins.h>
//#include </soft/compilers/ibmcmp-may2012/vacpp/bg/12.1/include/builtins.h>

//#undef __bgq__

#ifdef __bgq__

#include IBMCMP_BUILTINS

int isAligned( void *in );

void Step16_int( int count1, float xxi, float yyi, float zzi, float fsrrmax2, float mp_rsm2, float *xx1, float *yy1, float *zz1, float *mass1, float *ax, float *ay, float *az )
{
    int i = 0, j, k;
    const int offset = 32; 
    
    vector4double b0, b1, b2, b3, b4, b5, b6;
    vector4double c0, c1, c2, c3, c4, c5, c6;
    vector4double a0 = (vector4double){ 0.5, 0.5, 0.5, 0.5 };
    vector4double a1 = (vector4double){ (double)xxi, (double)xxi, (double)xxi, (double)xxi };
    vector4double a2 = (vector4double){ (double)yyi, (double)yyi, (double)yyi, (double)yyi };
    vector4double a3 = (vector4double){ (double)zzi, (double)zzi, (double)zzi, (double)zzi };
    vector4double a4 = (vector4double){ (double)fsrrmax2, (double)fsrrmax2, (double)fsrrmax2, (double)fsrrmax2 };
    vector4double a5 = (vector4double){ (double)mp_rsm2, (double)mp_rsm2, (double)mp_rsm2, (double)mp_rsm2 };
    vector4double a6 = (vector4double){ 0., 0., 0., 0. };
    vector4double a7 = (vector4double){ 0., 0., 0., 0. };
    vector4double a8 = (vector4double){ 0., 0., 0., 0. };
    vector4double a9 = (vector4double){ 0., 0., 0., 0. };
    vector4double a10 = (vector4double){ 0.269327, 0.269327, 0.269327, 0.269327 };
    vector4double a11 = (vector4double){ -0.0750978, -0.0750978, -0.0750978, -0.0750978 };
    vector4double a12 = (vector4double){ 0.0114808, 0.0114808, 0.0114808, 0.0114808 };
    vector4double a13 = (vector4double){ -0.00109313, -0.00109313, -0.00109313, -0.00109313 };
    vector4double a14 = (vector4double){ 0.0000605491, 0.0000605491, 0.0000605491, 0.0000605491 };
    vector4double a15 = (vector4double){ -0.00000147177, -0.00000147177, -0.00000147177, -0.00000147177 };

    /*
    int32_t depth = 3;
    L1P_SetStreamPolicy( L1P_stream_confirmed );
    L1P_SetStreamDepth(depth);
    */

    for ( i = 0, j = 0; i < count1-7; i = i + 8, j = j + 32 ) 
    { 
    
        __dcbt( (void *)&xx1  [i+offset] );
        __dcbt( (void *)&yy1  [i+offset] );
        __dcbt( (void *)&zz1  [i+offset] );
        __dcbt( (void *)&mass1[i+offset] );
    

        b0 = vec_ld( j   , xx1 );
        c0 = vec_ld( j+16, xx1 );
        
        b1 = vec_ld( j   , yy1 );
        c1 = vec_ld( j+16, yy1 );

        b2 = vec_ld( j   , zz1 );
        c2 = vec_ld( j+16, zz1 );

        b3 = vec_sub( b0, a1 );
        c3 = vec_sub( c0, a1 );

        b4 = vec_sub( b1, a2 );
        c4 = vec_sub( c1, a2 );

        b5 = vec_sub( b2, a3 );
        c5 = vec_sub( c2, a3 );

        b0 = vec_madd( b3, b3, a6 );
        c0 = vec_madd( c3, c3, a6 );

        b0 = vec_madd( b4, b4, b0 );
        c0 = vec_madd( c4, c4, c0 );

        b6 = vec_madd( b5, b5, b0 );
        c6 = vec_madd( c5, c5, c0 );
        
        b0 = vec_madd( b5, b5, b0 );
        c0 = vec_madd( c5, c5, c0 );
        
        b0 = vec_add( b0, a5 );
        c0 = vec_add( c0, a5 );
        
        b1 = vec_madd( b0, b0, a6 );
        c1 = vec_madd( c0, c0, a6 );
        
        b0 = vec_madd( b1, b0, a6 );
        c0 = vec_madd( c1, c0, a6 );
        
        b1 = vec_rsqrte( b0 );
        c1 = vec_rsqrte( c0 );
        
        b2 = vec_madd( b1, b1, a6 );
        c2 = vec_madd( c1, c1, a6 );
        
        b0 = vec_madd( b0, b2, a6 );
        c0 = vec_madd( c0, c2, a6 );
        
        b0 = vec_nmsub( a0, b0, a0 );
        c0 = vec_nmsub( a0, c0, a0 );
        
        b0 = vec_madd( b1, b0, b1 );
        c0 = vec_madd( c1, c0, c1 );
        
        b1 = vec_madd( b6, a15, a14 );
        c1 = vec_madd( c6, a15, a14 );
        
        b1 = vec_madd( b6, b1, a13 );
        c1 = vec_madd( c6, c1, a13 );
        
        b1 = vec_madd( b6, b1, a12 );
        c1 = vec_madd( c6, c1, a12 );
        
        b1 = vec_madd( b6, b1, a11 );
        c1 = vec_madd( c6, c1, a11 );
        
        b1 = vec_madd( b6, b1, a10 );
        c1 = vec_madd( c6, c1, a10 );
        
        b0 = vec_sub( b0, b1 );
        c0 = vec_sub( c0, c1 );
        
        b1 = vec_ld( j   , mass1 );
        c1 = vec_ld( j+16, mass1 );
        
        b2 = vec_sub( b6, a4 );
        c2 = vec_sub( c6, a4 );
        
        b1 = vec_sel( b1, a6, b2 );
        c1 = vec_sel( c1, a6, c2 );
        
        b1 = vec_madd( b1, b0, a6 );
        c1 = vec_madd( c1, c0, a6 );
        
        b2 = vec_sub( a6, b6 );
        c2 = vec_sub( a6, c6 );
        
        b0 = vec_sel( b1, a6, b2 );
        c0 = vec_sel( c1, a6, c2 );
        
        a7 = vec_madd( b0, b3, a7 );
        a7 = vec_madd( c0, c3, a7 );
        
        a8 = vec_madd( b0, b4, a8 );
        a8 = vec_madd( c0, c4, a8 );
        
        a9 = vec_madd( b0, b5, a9 );
        a9 = vec_madd( c0, c5, a9 );
    }

    *ax = ( a7[0] + a7[1] + a7[2] + a7[3] );
    *ay = ( a8[0] + a8[1] + a8[2] + a8[3] );
    *az = ( a9[0] + a9[1] + a9[2] + a9[3] );

    
    const float ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313, ma4 = 0.0000605491, ma5 = -0.00000147177;
    float dxc, dyc, dzc, m, r2, f;

    for ( k = i; k < count1; k++ ) 
    {
        dxc = xx1[k] - xxi;
        dyc = yy1[k] - yyi;
        dzc = zz1[k] - zzi;
  
        r2 = dxc * dxc + dyc * dyc + dzc * dzc;
       
        m = ( r2 < fsrrmax2 ) ? mass1[k] : 0.0f;

        f =  pow( r2 + mp_rsm2, -1.5 ) - ( ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5)))));
        
        f = ( r2 > 0.0f ) ? m * f : 0.0f;

        *ax = *ax + f * dxc;
        *ay = *ay + f * dyc;
        *az = *az + f * dzc;
    }

}

int isAligned(void *in){
  const int mask_04 = 0xFFFFFFFC;
  const int mask_08 = 0xFFFFFFF8;
  const int mask_16 = 0xFFFFFFF0;
  const int mask_32 = 0xFFFFFFE0;


  if((int)in == ((int)in & mask_32))
    return 32;
  if((int)in == ((int)in & mask_16))
    return 16;
  if((int)in == ((int)in & mask_08))
    return 8;
  if((int)in == ((int)in & mask_04))
    return 4;

  return -1;
}

#endif

