#ifndef FP_H
#define FP_H

#include "complex-type.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

static bool fp_enable_check = true;

///
// Print warnings about funny floating point numbers
///

// pgCC doesn't yet play well with C99 constructs, so...
#if defined(__cplusplus) && defined(__PGI_)
static inline void fp_check(double x) { }
#else
static inline void fp_check(double x)
{
#if defined(__cplusplus)
    int type = std::fpclassify(x);
#else
    int type = fpclassify(x);
#endif

    if (type == FP_SUBNORMAL) {
        fprintf(stderr, "Warning: fpclassify: FP_SUBNORMAL\n");
    } else if (type == FP_INFINITE) {
        fprintf(stderr, "Warning: fpclassify: FP_INFINITE\n");
    } else if (type == FP_NAN) {
        fprintf(stderr, "Warning: fpclassify: FP_NAN\n");
    }
}
#endif

///
// Find the number of representable floating point numbers between two
// floating point numbers.  Assumes IEEE 754 and 2's complement
// arithmetic.
//
// A separation of 1 corresponds to a relative difference of 1 part
// in 2^53 ~ 10^-16.
///
static inline uint64_t fp_diff(double a, double b)
{
    int64_t ia = *(int64_t *) &a;
    int64_t ib = *(int64_t *) &b;

    if (ia < 0) {
        ia = 0x8000000000000000LL - ia;
    }
    if (ib < 0) {
        ib = 0x8000000000000000LL - ib;
    }
    if (ia > ib) {
        return (uint64_t) (ia - ib);
    } else {
        return (uint64_t) (ib - ia);
    }
}


///
// Compare floating point numbers based on the number of representable
// floating point numbers between them.
//
// A tolerance of 1 corresponds to a relative difference of 1 part
// in 2^53 ~ 10^-16.
///
static inline bool fp_isclose(double a, double b, uint64_t tolerance)
{
    if (fp_enable_check) {
        fp_check(a);
        fp_check(b);
    }
    return fp_diff(a, b) <= tolerance;
}


///
// Compare floating point numbers based on the number of representable
// floating point numbers between them.
//
// A tolerance of 1 corresponds to a relative difference of 1 part
// in 2^53 ~ 10^-16.
///
static inline bool fp_complex_isclose(complex_t a, complex_t b, uint64_t tolerance)
{
    return (fp_isclose(real(a), real(b), tolerance) &&
            fp_isclose(imag(a), imag(b), tolerance));
}

#endif
