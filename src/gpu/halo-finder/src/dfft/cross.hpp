#ifndef CROSS_HPP
#define CROSS_HPP

#include "complex-type.h"

#ifdef FFTW3
#include <fftw3-mpi.h>
#else
#include <fftw_mpi.h>
#endif

#include <algorithm>
#include <vector>

#include "allocator.hpp"
#include "distribution.hpp"
#include "solver.hpp"

#include <string.h>

// pgCC doesn't yet play well with C99 constructs, so...
#ifdef __PGI__
extern "C" long int lrint(double x);
#endif

#define FFTW_ADDR(X) reinterpret_cast<fftw_complex*>(&(X)[0])

class CrossBase : public Distribution {

public:

  // methods



  CrossBase()
  {
  }



  CrossBase(MPI_Comm comm, int ng)
    : Distribution(comm, ng)
  {
    std::vector<int> n;
    n.assign(3, ng);
    initialize(comm, n);
  }



  CrossBase(MPI_Comm comm, std::vector<int> const & n)
    : Distribution(comm, n)
  {
    initialize(comm, n);
  }


  
  virtual ~CrossBase()
  {
#ifdef FFTW3
    fftw_destroy_plan(m_plan_f);
    fftw_destroy_plan(m_plan_b);
#else
    fftwnd_mpi_destroy_plan(m_plan_f);
    fftwnd_mpi_destroy_plan(m_plan_b);
#endif
  }
  


  // solve interfaces



  //NOT SURE I DID THIS CORRECTLY FOR FFTW2 -- ADRIAN
  void forward(complex_t const *rho1, complex_t const *rho2)
  {

    //forward rho1
#ifdef FFTW3
    distribution_3_to_1(rho1, &m_buf1[0], &m_d);            // rho1 --> buf1
    fftw_execute(m_plan_f);                                 // buf1 --> buf2
#else
    distribution_3_to_1(rho1, &m_buf2[0], &m_d);            // rho1 --> buf2
    fftwnd_mpi(m_plan_f, 1,   FFTW_ADDR(m_buf2), FFTW_ADDR(m_buf3),
	       FFTW_NORMAL_ORDER);                          // buf2 --> buf3
#endif

    //copy transformed rho1 (in buf2) to a safe place (buf3)
    memcpy( &m_buf3[0], &m_buf2[0], local_size()*sizeof(complex_t) );

    //forward rho2
#ifdef FFTW3
    distribution_3_to_1(rho2, &m_buf1[0], &m_d);            // rho2 --> buf1
    fftw_execute(m_plan_f);                                 // buf1 --> buf2
#else
    distribution_3_to_1(rho2, &m_buf2[0], &m_d);            // rho2 --> buf2
    fftwnd_mpi(m_plan_f, 1,   FFTW_ADDR(m_buf2), FFTW_ADDR(m_buf3),
                       FFTW_NORMAL_ORDER);                  // buf2 --> buf3
#endif
  }



  void backward_xi(complex_t *phi) 
  {
    //now, transformed rho1 in buf3, transformed rho2 in buf2
    //need intermediate result in buf1
    for(int i=0; i<local_size(); i++) {
      m_buf1[i] = m_buf2[i] * conj(m_buf3[i]);
    }

    //it would be nice to set (0,0,0) mode to 0
    int index = 0;
    for (int local_k0 = 0; local_k0 < local_ng_1d(0); ++local_k0) {
      int k0 = local_k0 + self_1d(0) * local_ng_1d(0);
      for (int k1 = 0; k1 < local_ng_1d(1); ++k1) {
	for (int k2 = 0; k2 < local_ng_1d(2); ++k2) {
	  if(k0 == 0 && k1==0 && k2==0) {
	    m_buf1[index] *= 0.0;
	  }
	  index++;
	}
	index += m_d.padding[2];
      }
      index += m_d.padding[1];
    }

#ifdef FFTW3
    fftw_execute(m_plan_b);                                 // buf1 --> buf3
    distribution_1_to_3(&m_buf3[0], phi, &m_d);             // buf3 --> phi
#else
    fftwnd_mpi(m_plan_b, 1,
	       (fftw_complex *) &m_buf1[0],
	       (fftw_complex *) &m_buf3[0],
	       FFTW_NORMAL_ORDER);                           // buf1 -->buf1
    distribution_1_to_3(&m_buf1[0], phi, &m_d);              // buf3 --> phi
#endif
  }


  
  // interfaces for std::vector


  
  void forward(std::vector<complex_t> const & rho1,
	       std::vector<complex_t> const & rho2)
  {
    forward(&rho1[0], &rho2[0]);
  }



  void backward_xi(std::vector<complex_t> & phi)
  {
    backward_xi(&phi[0]);
  }



  // analysis interfaces
    ///
  // calculate the k-space power spectrum
  //   P(modk) = Sum { |rho(k)|^2 : |k| = modk, k <- [0, ng / 2)^3, periodically extended }
  ///
  void power_spectrum(std::vector<double> & power)
  {
    std::vector<complex_t, fftw_allocator<complex_t> > const & rho1 = m_buf3;
    std::vector<complex_t, fftw_allocator<complex_t> > const & rho2 = m_buf2;
    std::vector<int> ksq;
    std::vector<double> weight;
    int ng = m_d.n[0];
    double volume = 1.0 * ng * ng * ng; 
    
    // cache periodic ksq
    ksq.resize(ng);
    double ksq_max = 0;
    for (int k = 0; k < ng / 2; ++k) {
      ksq[k] = k * k;
      ksq_max = max(ksq_max, ksq[k]);
      ksq[k + ng / 2] = (k - ng / 2) * (k - ng / 2);
      ksq_max = max(ksq_max, ksq[k + ng / 2]);
    }
    long modk_max = lrint(sqrt(3 * ksq_max)); // round to nearest integer
    
    // calculate power spectrum
    power.resize(modk_max + 1);
    power.assign(modk_max + 1, 0.0);
    weight.resize(modk_max + 1);
    weight.assign(modk_max + 1, 0.0);
    int index = 0;
    for (int local_k0 = 0; local_k0 < local_ng_1d(0); ++local_k0) {
      int k0 = local_k0 + self_1d(0) * local_ng_1d(0);
      double ksq0 = ksq[k0];
      for (int k1 = 0; k1 < local_ng_1d(1); ++k1) {
	double ksq1 = ksq[k1];
	for (int k2 = 0; k2 < local_ng_1d(2); ++k2) {
	  double ksq2 = ksq[k2];
	  // round to nearest integer
	  long modk = lrint(sqrt(ksq0 + ksq1 + ksq2));
	  power[modk] += real(rho1[index] * conj(rho2[index]));
	  weight[modk] += volume;
	  index++;
	}
	index += m_d.padding[2];
      }
      index += m_d.padding[1];
    }
    
    // accumulate across processors
    MPI_Allreduce(MPI_IN_PLACE, &power[0], power.size(), MPI_DOUBLE, MPI_SUM, cart_1d());
    MPI_Allreduce(MPI_IN_PLACE, &weight[0], weight.size(), MPI_DOUBLE, MPI_SUM, cart_1d());
    
    //make sure we don't divide by zero
    for(int i = 0; i < weight.size(); ++i) {
      weight[i] += 1.0 * (weight[i] < 1.0);
    }
    
    // scale power by weight
    std::transform(power.begin(), power.end(), weight.begin(), power.begin(), std::divides<double>());
  }
  


  ///
  // General initialization
  ///
  void initialize(MPI_Comm comm, std::vector<int> n, bool transposed_order = false)
  {
    int flags_f;
    int flags_b;
    
    distribution_init(comm, &n[0], &n[0], &m_d, false);
    distribution_assert_commensurate(&m_d);
#ifdef FFTW3
    fftw_mpi_init();
#endif
    m_buf1.resize(local_size());
    m_buf2.resize(local_size());
    m_buf3.resize(local_size());
    
    // create plan for forward and backward DFT's
    flags_f = flags_b = FFTW_ESTIMATE;
#ifdef FFTW3
    if (transposed_order) {
      flags_f |= FFTW_MPI_TRANSPOSED_OUT;
      flags_b |= FFTW_MPI_TRANSPOSED_IN;
    }
    m_plan_f = fftw_mpi_plan_dft_3d(n[0], n[1], n[2],
				    FFTW_ADDR(m_buf1), FFTW_ADDR(m_buf2),
				    comm, FFTW_FORWARD, flags_f);
    m_plan_b = fftw_mpi_plan_dft_3d(n[0], n[1], n[2],
				    FFTW_ADDR(m_buf1), FFTW_ADDR(m_buf3),
				    comm, FFTW_BACKWARD, flags_b);
#else
    m_plan_f = fftw3d_mpi_create_plan(comm, n[0], n[1], n[2], FFTW_FORWARD, flags_f);
    m_plan_b = fftw3d_mpi_create_plan(comm, n[0], n[1], n[2], FFTW_BACKWARD, flags_b);
#endif
  }



protected:
    double max(double a, double b) { return a > b ? a : b; }
    std::vector<complex_t, fftw_allocator<complex_t> > m_buf1;
    std::vector<complex_t, fftw_allocator<complex_t> > m_buf2;
    std::vector<complex_t, fftw_allocator<complex_t> > m_buf3;
#ifdef FFTW3
    fftw_plan m_plan_f;
    fftw_plan m_plan_b;
#else
    fftwnd_mpi_plan m_plan_f;
    fftwnd_mpi_plan m_plan_b;
#endif
};

#endif
