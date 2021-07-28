#define POSVEL_T float
#define ID_T int

void cm_int(ID_T count, const POSVEL_T* __restrict xx, const POSVEL_T* __restrict yy,
                      const POSVEL_T* __restrict zz, const POSVEL_T* __restrict mass,
                      POSVEL_T* __restrict xmin, POSVEL_T* __restrict xmax, POSVEL_T* __restrict xc)
{
  // xmin/xmax are currently set to the whole bounding box, but this is too conservative, so we'll
  // set them based on the actual particle content.

  double x = 0, y = 0, z = 0, m = 0;

  POSVEL_T w,x1,x2,y1,y2,z1,z2;

  x1 = xx[0]; x2 = xx[0];
  y1 = yy[0]; y2 = yy[0];
  z1 = zz[0]; z2 = zz[0];

  for (int i = 0; i < count; ++i) 
  {
    if ( x1 > xx[i] ) x1 = xx[i]; /* x1 = min( xx[] ) */
    if ( x2 < xx[i] ) x2 = xx[i]; /* x2 = max( xx[] ) */
    if ( y1 > yy[i] ) y1 = yy[i]; /* y1 = min( yy[] ) */
    if ( y2 < yy[i] ) y2 = yy[i]; /* y2 = max( yy[] ) */
    if ( z1 > zz[i] ) z1 = zz[i]; /* z1 = min( zz[] ) */
    if ( z2 < zz[i] ) z2 = zz[i]; /* z2 = max( zz[] ) */

    w = mass[i];
    x = x + w * xx[i];
    y = y + w * yy[i];
    z = z + w * zz[i];
    m = m + w;
  }

  xc[0] = (POSVEL_T) (x/m);
  xc[1] = (POSVEL_T) (y/m);
  xc[2] = (POSVEL_T) (z/m);

  xmin[0] = x1; xmax[0] = x2;
  xmin[1] = y1; xmax[1] = y2;
  xmin[2] = z1; xmax[2] = z2;
}

