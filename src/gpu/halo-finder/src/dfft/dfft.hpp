#ifndef DFFT_HPP
#define DFFT_HPP

///
// Distributed FFT
//
// This is a high-level interface providing FFT's on 3-d data
// distribution.  The same data distribution (partition) is assumed in
// both x and k space and is determined by the underlying Distribution
// class.
///

#include <vector>

#include "complex-type.h"

#ifdef FFTW3
#include <fftw3-mpi.h>
#else
#include <fftw_mpi.h>
#endif

#include "allocator.hpp"
#include "distribution.hpp"

#define FFTW_ADDR(X) reinterpret_cast<fftw_complex *>(&(X)[0])

class Dfft : public Distribution {

public:
    
    Dfft()
        : Distribution()
        {
        }

    Dfft(MPI_Comm comm, int ng, bool transposed_order = false)
        : Distribution(comm, ng)
        {
            std::vector<int> n;
            n.assign(3, ng);
            initialize(comm, &n[0], transposed_order);
        }

    Dfft(MPI_Comm comm, std::vector<int> const & n, bool transposed_order = false)
        : Distribution(comm, n)
        {
            initialize(comm, &n[0], transposed_order);
        }

    Dfft(MPI_Comm comm, int const n[], bool transposed_order = false)
        : Distribution(comm, n)
        {
            initialize(comm, n, transposed_order);
        }

    ~Dfft()
        {
#ifdef FFTW3
            fftw_destroy_plan(m_plan_f);
            fftw_destroy_plan(m_plan_b);
#else
            fftwnd_mpi_destroy_plan(m_plan_f);
            fftwnd_mpi_destroy_plan(m_plan_b);
#endif
        }

    void initialize(MPI_Comm comm, int const n[], bool transposed_order)
        {
            int padding[3] = { 0, 0, 0 };
            int flags_f;
            int flags_b;

#ifdef FFTW3
            fftw_mpi_init();
#endif
            distribution_init(comm, n, padding, &m_d, false);
            distribution_assert_commensurate(&m_d);

            m_buf1.resize(local_size());
            m_buf2.resize(local_size());

            // create plan for forward and backward DFT's
            flags_f = flags_b = FFTW_ESTIMATE;
#ifdef FFTW3
            if (transposed_order) {
                flags_f |= FFTW_MPI_TRANSPOSED_OUT;
                flags_b |= FFTW_MPI_TRANSPOSED_IN;
            }
            m_plan_f = fftw_mpi_plan_dft_3d(n[0], n[1], n[2], FFTW_ADDR(m_buf1), FFTW_ADDR(m_buf2),
                                            comm, FFTW_FORWARD, flags_f);
            m_plan_b = fftw_mpi_plan_dft_3d(n[0], n[1], n[2], FFTW_ADDR(m_buf1), FFTW_ADDR(m_buf2),
                                            comm, FFTW_BACKWARD, flags_b);
#else
            m_plan_f = fftw3d_mpi_create_plan(comm, n[0], n[1], n[2], FFTW_FORWARD, flags_f);
            m_plan_b = fftw3d_mpi_create_plan(comm, n[0], n[1], n[2], FFTW_BACKWARD, flags_b);
#endif
        }

    ///
    // Forward transform
    ///
    void forward(double const *in, complex_t *out)
        {
            complexify(in, &m_buf2[0], m_buf2.size());              // in   --> buf2
            distribution_3_to_1(&m_buf2[0], &m_buf1[0], &m_d);      // buf2 --> buf1
#ifdef FFTW3
            fftw_execute(m_plan_f);                                 // buf1 --> buf2
            distribution_1_to_3(&m_buf2[0], out, &m_d);             // buf2 --> out
#else
            fftwnd_mpi(m_plan_f, 1,
                       (fftw_complex *) &m_buf1[0],
                       (fftw_complex *) &m_buf2[0],
                       FFTW_NORMAL_ORDER);                          // buf1 -->buf1
            distribution_1_to_3(&m_buf1[0], out, &m_d);             // buf1 --> out
#endif
        }

    void forward(complex_t const *in, complex_t *out)
        {
            distribution_3_to_1(in, &m_buf1[0], &m_d);              // in --> buf1
#ifdef FFTW3
            fftw_execute(m_plan_f);                                 // buf1 --> buf2
            distribution_1_to_3(&m_buf2[0], out, &m_d);             // buf2 --> out
#else
            fftwnd_mpi(m_plan_f, 1,
                       (fftw_complex *) &m_buf1[0],
                       (fftw_complex *) &m_buf2[0],
                       FFTW_NORMAL_ORDER);                          // buf1 -->buf1
            distribution_1_to_3(&m_buf1[0], out, &m_d);             // buf1 --> out
#endif
        }

    void forward(std::vector<double> const & in, std::vector<complex_t> & out)
        {
            forward(&in[0], &out[0]);
        }

    void forward(std::vector<complex_t> const & in, std::vector<complex_t> & out)
        {
            forward(&in[0], &out[0]);
        }

    ///
    // Backward transform
    ///
    void backward(complex_t const *in, double *out)
        {
#ifdef FFTW3
            distribution_3_to_1(in, &m_buf1[0], &m_d);              // in --> buf1
            fftw_execute(m_plan_f);                                 // buf1 --> buf2
#else
            distribution_3_to_1(in, &m_buf1[0], &m_d);              // in --> buf2
            fftwnd_mpi(m_plan_f, 1,
                       (fftw_complex *) &m_buf2[0],
                       (fftw_complex *) &m_buf1[0],
                       FFTW_NORMAL_ORDER);                          // buf2 -->buf2
#endif
            distribution_1_to_3(&m_buf2[0], &m_buf1[0], &m_d);      // buf2 --> buf1
            decomplexify(&m_buf1[0], out, m_buf1.size());           // buf1 --> out
        }

    void backward(complex_t const *in, complex_t *out)
        {
#ifdef FFTW3
            distribution_3_to_1(in, &m_buf1[0], &m_d);              // in --> buf1
            fftw_execute(m_plan_f);                                 // buf1 --> buf2
#else
            distribution_3_to_1(in, &m_buf1[0], &m_d);              // in --> buf2
            fftwnd_mpi(m_plan_f, 1,
                       (fftw_complex *) &m_buf2[0],
                       (fftw_complex *) &m_buf1[0],
                       FFTW_NORMAL_ORDER);                          // buf2 -->buf2
#endif
            distribution_1_to_3(&m_buf2[0], out, &m_d);             // buf2 --> out
        }

    void backward(std::vector<complex_t> const & in, std::vector<double> & out)
        {
            backward(&in[0], &out[0]);
        }

    void backward(std::vector<complex_t> const & in, std::vector<complex_t> & out)
        {
            backward(&in[0], &out[0]);
        }

private:

    void complexify(double const *r, complex_t *z, size_t size)
        {
            for (size_t i = 0; i < size; ++i) {
                z[i] = r[i];
            }
        }

    void decomplexify(complex_t const *z, double *r, size_t size)
        {
            for (size_t i = 0; i < size; ++i) {
                r[i] = real(z[i]);
            }
        }

    std::vector<complex_t, fftw_allocator<complex_t> > m_buf1;
    std::vector<complex_t, fftw_allocator<complex_t> > m_buf2;
#ifdef FFTW3
    fftw_plan m_plan_f;
    fftw_plan m_plan_b;
#else
    fftwnd_mpi_plan m_plan_f;
    fftwnd_mpi_plan m_plan_b;
#endif
};

#endif
