/*
 * Vec2.hh
 *
 *  Created on: Dec 21, 2011
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef VEC2_HH_
#define VEC2_HH_

#include <cmath>
#include <string>
#include <sstream>
#include <ostream>


#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define FNQUALIFIERS __host__ __device__
#else
#define FNQUALIFIERS
#endif

// This class is defined with nearly all functions inline,
// to give the compiler maximum opportunity to optimize.
// Only the functions involving strings and I/O are
// out-of-line.

#ifndef __HIPCC__
// we are not in CUDA, so need to define our own double2 struct
struct double2
{
    typedef double value_type;
    double x, y;
    inline double2() {}
    inline double2(const double& x_, const double& y_) : x(x_), y(y_) {}
    inline double2(const double2& v2) : x(v2.x), y(v2.y) {}
    inline ~double2() {}

    inline double2& operator=(const double2& v2)
    {
        x = v2.x;
        y = v2.y;
        return(*this);
    }

    inline double2& operator+=(const double2& v2)
    {
        x += v2.x;
        y += v2.y;
        return(*this);
    }

    inline double2& operator-=(const double2& v2)
    {
        x -= v2.x;
        y -= v2.y;
        return(*this);
    }

    inline double2& operator*=(const double& r)
    {
        x *= r;
        y *= r;
        return(*this);
    }

    inline double2& operator/=(const double& r)
    {
        x /= r;
        y /= r;
        return(*this);
    }

}; // double2

inline double2 make_double2(const double& x_, const double& y_) {
    return(double2(x_, y_));
}

#else
// we are in CUDA; double2 is defined but needs op= operators
FNQUALIFIERS
inline double2& operator*=(double2& v, const double& r)
{
    v.x *= r;
    v.y *= r;
    return(v);
}

FNQUALIFIERS
inline double2& operator/=(double2& v, const double& r)
{
    v.x /= r;
    v.y /= r;
    return(v);
}
#endif



// unary operators:

// unary plus
FNQUALIFIERS
inline double2 operator+(const double2& v)
{
    return(v);
}

// unary minus
FNQUALIFIERS
inline double2 operator-(const double2& v)
{
    return(make_double2(-v.x, -v.y));
}


// binary operators:

// // multiply vector by scalar
// FNQUALIFIERS
// inline double2 operator*(const double2& v, const double& r)
// {
//     return(make_double2(v.x * r, v.y * r));
// }

// // multiply scalar by vector
// FNQUALIFIERS
// inline double2 operator*(const double& r, const double2& v)
// {
//     return(make_double2(v.x * r, v.y * r));
// }

// divide vector by scalar
FNQUALIFIERS
inline double2 operator/(const double2& v, const double& r)
{
    double rinv = (double) 1. / r;
    return(make_double2(v.x * rinv, v.y * rinv));
}


// other vector operations:

// dot product
FNQUALIFIERS
inline double dot(const double2& v1, const double2& v2)
{
    return(v1.x * v2.x + v1.y * v2.y);
}

// cross product (2D)
FNQUALIFIERS
inline double cross(const double2& v1, const double2& v2)
{
    return(v1.x * v2.y - v1.y * v2.x);
}

// length
FNQUALIFIERS
inline double length(const double2& v)
{
    return(sqrt(v.x * v.x + v.y * v.y));
}

// length squared
FNQUALIFIERS
inline double length2(const double2& v)
{
    return(v.x * v.x + v.y * v.y);
}

// rotate 90 degrees counterclockwise
FNQUALIFIERS
inline double2 rotateCCW(const double2& v)
{
    return(make_double2(-v.y, v.x));
}

// rotate 90 degrees clockwise
FNQUALIFIERS
inline double2 rotateCW(const double2& v)
{
    return(make_double2(v.y, -v.x));
}

// project v onto subspace perpendicular to u
// u must be a unit vector
FNQUALIFIERS
inline double2 project(double2& v, const double2& u)
{
    // assert(length2(u) == 1.);
    return v - dot(v, u) * u;
}


// miscellaneous:

//// output to stream
//std::ostream& operator<<(std::ostream& os, const double2& v)
//{
//    os << "(" << v.x << ", " << v.y << ")";
//    return os;
//}

//// conversion to string
//std::string makeString(const double2& v)
//{
//    std::ostringstream oss;
//    oss << v;
//    return oss.str();
//}


// convenience typedefs:

//typedef Vec2<float> float2;
//typedef Vec2<double> double2;


#endif /* VEC2_HH_ */
