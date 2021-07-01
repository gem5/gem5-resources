// Compatibility file for C99 and C++ complex.  This header
// can be included by either C99 or ANSI C++ programs to
// allow complex arithmetic to be written in a common subset.
// Note that overloads for both the real and complex math
// functions are available after this header has been
// included.

#ifndef COMPLEX_TYPE_H
#define COMPLEX_TYPE_H

#ifdef __cplusplus

#include <cmath>
#include <complex>

typedef std::complex<double> complex_t;

#define I complex_t(0.0, 1.0)

#else

#include <complex.h>
#include <math.h>

typedef double complex complex_t;

#define complex_t(r,i) ((double)(r) + ((double)(i)) * I)

#define real(x) creal(x)
#define imag(x) cimag(x)
#define abs(x) fabs(x)
#define arg(x) carg(x)

#endif  // #ifdef __cplusplus

#endif  // #ifndef COMPLEX_TYPE_H
