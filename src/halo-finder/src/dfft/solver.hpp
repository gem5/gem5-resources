#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "complex-type.h"

#if !defined(FFTW3) && defined(PENCIL)
#error PENCIL FFT REQUIRES FFTW3
#endif

#if defined(FFTW3) && defined(PENCIL)
#ifdef ESSL_FFTW
#include <fftw3_essl.h>
#else
#include <fftw3.h>
#endif
#elif defined(FFTW3)
#include <fftw3-mpi.h>
#else
#include <fftw_mpi.h>
#endif

#include <algorithm>
#include <vector>
#include <cassert>

#include "allocator.hpp"
#include "distribution.hpp"

#include "bigchunk.h"

#define CERRILLOS_SLAB_HACK 1

const double pi = 3.14159265358979323846;

// pgCC doesn't yet play well with C99 constructs, so...
#ifdef __PGI__
extern "C" long int lrint(double x);
#endif

#define FFTW_ADDR(X) reinterpret_cast<fftw_complex*>(&(X)[0])

///
// Abstract base class for Poisson solvers.
//
// Derived classes must provide their own implementation of
// initialize_greens_function(), and, if necessary, override the
// backward_solve() and backward_solve_gradient() methods.
///
class SolverBase : public Distribution {

public:

  // methods
  
  SolverBase()
  {
  }
  
  SolverBase(MPI_Comm comm, int ng)
    : Distribution(comm, ng)
  {
    initialize(comm);
  }
  
  SolverBase(MPI_Comm comm, std::vector<int> const & n)
    : Distribution(comm, n)
  {
    initialize(comm);
  }
  
  virtual ~SolverBase()
  {
#if defined(FFTW3) && defined(PENCIL)
    fftw_destroy_plan(m_plan_f_x);
    fftw_destroy_plan(m_plan_f_y);
    fftw_destroy_plan(m_plan_f_z);
    fftw_destroy_plan(m_plan_b_x);
    fftw_destroy_plan(m_plan_b_y);
    fftw_destroy_plan(m_plan_b_z);
#elif defined(FFTW3)
    fftw_destroy_plan(m_plan_f);
    fftw_destroy_plan(m_plan_b);
#else
    fftwnd_mpi_destroy_plan(m_plan_f);
    fftwnd_mpi_destroy_plan(m_plan_b);
#endif
  }
  
  // solve interfaces
  
  void forward_solve(complex_t const *rho)
  {
#if defined(FFTW3) && defined(PENCIL)

    distribution_3_to_2(rho, &m_buf1[0], &m_d, 0);            // rho  --> buf1
    fftw_execute(m_plan_f_x);                                 // buf1 --> buf2
    distribution_2_to_3(&m_buf2[0], &m_buf1[0], &m_d, 0);     // buf2 --> buf1
    distribution_3_to_2(&m_buf1[0], &m_buf2[0], &m_d, 1);     // buf1 --> buf2
    fftw_execute(m_plan_f_y);                                 // buf2 --> buf1
    distribution_2_to_3(&m_buf1[0], &m_buf2[0], &m_d, 1);     // buf1 --> buf2
    distribution_3_to_2(&m_buf2[0], &m_buf1[0], &m_d, 2);     // buf2 --> buf1
    fftw_execute(m_plan_f_z);                                 // buf1 --> buf2

#elif defined(FFTW3)
    distribution_3_to_1(rho, &m_buf1[0], &m_d);             // rho  --> buf1
    fftw_execute(m_plan_f);                                 // buf1 --> buf2
#else
    distribution_3_to_1(rho, &m_buf2[0], &m_d);             // rho  --> buf2
    fftwnd_mpi(m_plan_f, 1,   
	       FFTW_ADDR(m_buf2), 
	       FFTW_ADDR(m_buf3),
	       FFTW_NORMAL_ORDER);                          // buf2 -->buf2
#endif
  }
  
  void backward_solve(complex_t *phi)
  {
    kspace_solve(&m_buf2[0], &m_buf1[0]);                   // buf2 --> buf1
#if defined(FFTW3) && defined(PENCIL)
    fftw_execute(m_plan_b_z);                               // buf1 --> buf3
    distribution_2_to_3(&m_buf3[0], &m_buf1[0], &m_d, 2);   // buf3 --> buf1
    distribution_3_to_2(&m_buf1[0], &m_buf3[0], &m_d, 1);   // buf1 --> buf3
    fftw_execute(m_plan_b_y);                               // buf3 --> buf1
    distribution_2_to_3(&m_buf1[0], &m_buf3[0], &m_d, 1);   // buf1 --> buf3
    distribution_3_to_2(&m_buf3[0], &m_buf1[0], &m_d, 0);   // buf3 --> buf1
    fftw_execute(m_plan_b_x);                               // buf1 --> buf3
    distribution_2_to_3(&m_buf3[0], phi, &m_d, 0);          // buf3 --> phi
#elif defined(FFTW3)
    fftw_execute(m_plan_b);                                 // buf1 --> buf3
    distribution_1_to_3(&m_buf3[0], phi, &m_d);             // buf3 --> phi
#else
    fftwnd_mpi(m_plan_b, 1,
	       (fftw_complex *) &m_buf1[0],
	       (fftw_complex *) &m_buf3[0],
	       FFTW_NORMAL_ORDER);                           // buf1 -->buf1
    distribution_1_to_3(&m_buf1[0], phi, &m_d);              // buf1 --> phi
#endif
  }
  
  void backward_solve_gradient(int axis, complex_t *grad_phi)
  {
    kspace_solve_gradient(axis, &m_buf2[0], &m_buf1[0]);    // buf2 --> buf1
#if defined(FFTW3) && defined(PENCIL)
    fftw_execute(m_plan_b_z);                               // buf1 --> buf3
    distribution_2_to_3(&m_buf3[0], &m_buf1[0], &m_d, 2);   // buf3 --> buf1
    distribution_3_to_2(&m_buf1[0], &m_buf3[0], &m_d, 1);   // buf1 --> buf3
    fftw_execute(m_plan_b_y);                               // buf3 --> buf1
    distribution_2_to_3(&m_buf1[0], &m_buf3[0], &m_d, 1);   // buf1 --> buf3
    distribution_3_to_2(&m_buf3[0], &m_buf1[0], &m_d, 0);   // buf3 --> buf1
    fftw_execute(m_plan_b_x);                               // buf1 --> buf3
    distribution_2_to_3(&m_buf3[0], grad_phi, &m_d, 0);     // buf3 --> grad_phi
#elif defined(FFTW3)
    fftw_execute(m_plan_b);                                 // buf1 --> buf3
    distribution_1_to_3(&m_buf3[0], grad_phi, &m_d);        // buf3 --> grad_phi
#else
    fftwnd_mpi(m_plan_b, 1,
	       (fftw_complex *) &m_buf1[0],
	       (fftw_complex *) &m_buf3[0],
	       FFTW_NORMAL_ORDER);                           // buf1 -> buf1
    distribution_1_to_3(&m_buf1[0], grad_phi, &m_d);         // buf1 --> grad_phi
#endif
  }
  
  void solve(const complex_t *rho, complex_t *phi)
  {
    forward_solve(rho);
    backward_solve(phi);
  }
  
  void solve_gradient(int axis, const complex_t *rho, complex_t *phi)
  {
    forward_solve(rho);
    backward_solve_gradient(axis, phi);
  }
  
  // interfaces for std::vector
  
  void forward_solve(std::vector<complex_t> const & rho)
  {
    forward_solve(&rho[0]);
  }
  
  void backward_solve(std::vector<complex_t> & phi)
  {
    backward_solve(&phi[0]);
  }
  
  void backward_solve_gradient(int axis, std::vector<complex_t> & phi)
  {
    backward_solve_gradient(axis, &phi[0]);
  }
  
  void solve(std::vector<complex_t> const & rho, std::vector<complex_t> & phi)
  {
    solve(&rho[0], &phi[0]);
  }
  
  void solve_gradient(int axis, std::vector<complex_t> const & rho, std::vector<complex_t> & phi)
  {
    solve_gradient(axis, &rho[0], &phi[0]);
  }
  
  
  // analysis interfaces
  
  ///
  // calculate the k-space power spectrum
  //   P(modk) = Sum { |rho(k)|^2 : |k| = modk, k <- [0, ng / 2)^3, periodically extended }
  ///
  void power_spectrum(std::vector<double> & power)
  {
    //intermediate in m_buf2 for both FFTW2 and FFTW3
    std::vector<complex_t, bigchunk_allocator<complex_t> > const & rho = m_buf2;

    int ng = m_d.n[0];
    double volume = 1.0 * ng * ng * ng; 
    double kk, tpi;

    tpi = 2.0*atan(1.0)*4.0;

    // cache periodic ksq
    m_pk_ksq.resize(ng);
    m_pk_cic.resize(ng);
    double ksq_max = 0;
    for (int k = 0; k < ng / 2; ++k) {

      m_pk_ksq[k] = k * k;
      ksq_max = max(ksq_max, m_pk_ksq[k]);

      m_pk_ksq[k + ng / 2] = (k - ng / 2) * (k - ng / 2);
      ksq_max = max(ksq_max, m_pk_ksq[k + ng / 2]);

      kk = tpi*k/ng;
      m_pk_cic[k] = pow(sin(0.5*kk)/(0.5*kk),-4.0);

      kk = tpi*(k-ng/2)/ng;
      m_pk_cic[k + ng/2] = pow(sin(0.5*kk)/(0.5*kk),-4.0);
    }
    m_pk_cic[0] = 1.0;

    long modk_max = lrint(sqrt(3 * ksq_max)); // round to nearest integer
    
    // calculate power spectrum
    power.resize(modk_max + 1);
    power.assign(modk_max + 1, 0.0);

    m_pk_weight.resize(modk_max + 1);
    m_pk_weight.assign(modk_max + 1, 0.0);
    
    /*
!-----3-D anti-CIC filter for deconvolution

      forall(ii=1:ng)kk(ii)=(ii-1)*tpi/(1.0*ng)

      do ii=1,ng
      if(ii.ge.ng/2+1)kk(ii)=(ii-ng-1)*tpi/(1.0*ng)
      enddo

      mult(1)=1.0
      
      forall(ii=2:ng)mult(ii)=
      #      1.0/(sin(kk(ii)/2.0)/(kk(ii)/2.0))**2
      
      forall(ii=1:ng,jj=1:ng,mm=1:ng)erhotr(ii,jj,mm)=
      #      mult(ii)*mult(jj)*mult(mm)*erhotr(ii,jj,mm)
    */
   
    int local_dim[3]; 
    int self_coord[3];
#if defined(FFTW3) && defined(PENCIL)
    self_coord[0]=self_2d_z(0);
    self_coord[1]=self_2d_z(1);
    self_coord[2]=self_2d_z(2);
    local_dim[0]=local_ng_2d_z(0);
    local_dim[1]=local_ng_2d_z(1);
    local_dim[2]=local_ng_2d_z(2);
#else
    self_coord[0]=self_1d(0);
    self_coord[1]=self_1d(1);
    self_coord[2]=self_1d(2);
    local_dim[0]=local_ng_1d(0);
    local_dim[1]=local_ng_1d(1);
    local_dim[2]=local_ng_1d(2);
#endif
    int index = 0;
    for (int local_k0 = 0; local_k0 < local_dim[0]; ++local_k0) {
      int k0 = local_k0 + self_coord[0] * local_dim[0];
      double ksq0 = m_pk_ksq[k0];

      for (int local_k1 = 0; local_k1 < local_dim[1]; ++local_k1) {
        int k1 = local_k1 + self_coord[1] * local_dim[1];
	double ksq1 = m_pk_ksq[k1];

	for (int local_k2 = 0; local_k2 < local_dim[2]; ++local_k2) {
          int k2 = local_k2 + self_coord[2] * local_dim[2];
	  double ksq2 = m_pk_ksq[k2];
	  long modk = lrint(sqrt(ksq0 + ksq1 + ksq2)); //round to nearest int
	  //power[modk] += real(rho[index] * conj(rho[index]));
	  power[modk] += std::real(rho[index] * conj(rho[index])) * m_pk_cic[k0] * m_pk_cic[k1] * m_pk_cic[k2];
	  m_pk_weight[modk] += volume;
	  index++;
	}
	index += m_d.padding[2];
      }
      index += m_d.padding[1];
    }
    
    // accumulate across processors
    MPI_Allreduce(MPI_IN_PLACE, &power[0], power.size(), 
		  MPI_DOUBLE, MPI_SUM, cart_1d());
    MPI_Allreduce(MPI_IN_PLACE, &m_pk_weight[0], m_pk_weight.size(), 
		  MPI_DOUBLE, MPI_SUM, cart_1d());
    
    //make sure we don't divide by zero
    for(size_t i = 0; i < m_pk_weight.size(); ++i) {
      m_pk_weight[i] += 1.0 * (m_pk_weight[i] < 1.0);
    }
    
    // scale power by weight
    std::transform(power.begin(), power.end(), 
		   m_pk_weight.begin(), power.begin(), 
		   std::divides<double>());
  }

  ///
  // General initialization
  ///
  void initialize(MPI_Comm comm, bool transposed_order = false)
  {
    int flags_f;
    int flags_b;
    
    // distribution_init(comm, &n[0], &n[0], &m_d, false);
    // distribution_assert_commensurate(&m_d);
#if defined(FFTW3) && !defined(PENCIL)
    fftw_mpi_init();
#endif
    m_greens_functions_initialized = false;
    m_buf1.resize(local_size());
    m_buf2.resize(local_size());
    m_buf3.resize(local_size());
    
    // create plan for forward and backward DFT's
    flags_f = flags_b = FFTW_ESTIMATE;
#if defined(FFTW3) && defined(PENCIL)
    m_plan_f_x = fftw_plan_many_dft(1, // rank
                                    &(m_d.process_topology_2_x.n[0]), // const int *n,
                                    m_d.process_topology_2_x.n[1] * m_d.process_topology_2_x.n[2], // howmany
                                    FFTW_ADDR(m_buf1),
                                    NULL, // const int *inembed,
                                    1, // int istride,
                                    m_d.process_topology_2_x.n[0], // int idist,
                                    FFTW_ADDR(m_buf2),
                                    NULL, // const int *onembed,
                                    1, // int ostride,
                                    m_d.process_topology_2_x.n[0], // int odist,
                                    FFTW_FORWARD, // int sign,
                                    0); // unsigned flags
    m_plan_f_y = fftw_plan_many_dft(1, // rank
                                    &(m_d.process_topology_2_y.n[1]), // const int *n,
                                    m_d.process_topology_2_y.n[0] * m_d.process_topology_2_y.n[2], // howmany
                                    FFTW_ADDR(m_buf2),
                                    NULL, // const int *inembed,
                                    1, // int istride,
                                    m_d.process_topology_2_y.n[1], // int idist,
                                    FFTW_ADDR(m_buf1),
                                    NULL, // const int *onembed,
                                    1, // int ostride,
                                    m_d.process_topology_2_y.n[1], // int odist,
                                    FFTW_FORWARD, // int sign,
                                    0); // unsigned flags
    m_plan_f_z = fftw_plan_many_dft(1, // rank
                                    &(m_d.process_topology_2_z.n[2]), // const int *n,
                                    m_d.process_topology_2_z.n[1] * m_d.process_topology_2_z.n[0], // howmany
                                    FFTW_ADDR(m_buf1),
                                    NULL, // const int *inembed,
                                    1, // int istride,
                                    m_d.process_topology_2_z.n[2], // int idist,
                                    FFTW_ADDR(m_buf2),
                                    NULL, // const int *onembed,
                                    1, // int ostride,
                                    m_d.process_topology_2_z.n[2], // int odist,
                                    FFTW_FORWARD, // int sign,
                                    0); // unsigned flags
    m_plan_b_x = fftw_plan_many_dft(1, // rank
                                    &(m_d.process_topology_2_x.n[0]), // const int *n,
                                    m_d.process_topology_2_x.n[1] * m_d.process_topology_2_x.n[2], // howmany
                                    FFTW_ADDR(m_buf1),
                                    NULL, // const int *inembed,
                                    1, // int istride,
                                    m_d.process_topology_2_x.n[0], // int idist,
                                    FFTW_ADDR(m_buf3),
                                    NULL, // const int *onembed,
                                    1, // int ostride,
                                    m_d.process_topology_2_x.n[0], // int odist,
                                    FFTW_BACKWARD, // int sign,
                                    0); // unsigned flags
    m_plan_b_y = fftw_plan_many_dft(1, // rank
                                    &(m_d.process_topology_2_y.n[1]), // const int *n,
                                    m_d.process_topology_2_y.n[0] * m_d.process_topology_2_y.n[2], // howmany
                                    FFTW_ADDR(m_buf3),
                                    NULL, // const int *inembed,
                                    1, // int istride,
                                    m_d.process_topology_2_y.n[1], // int idist,
                                    FFTW_ADDR(m_buf1),
                                    NULL, // const int *onembed,
                                    1, // int ostride,
                                    m_d.process_topology_2_y.n[1], // int odist,
                                    FFTW_BACKWARD, // int sign,
                                    0); // unsigned flags
    m_plan_b_z = fftw_plan_many_dft(1, // rank
                                    &(m_d.process_topology_2_z.n[2]), // const int *n,
                                    m_d.process_topology_2_z.n[1] * m_d.process_topology_2_z.n[0], // howmany
                                    FFTW_ADDR(m_buf1),
                                    NULL, // const int *inembed,
                                    1, // int istride,
                                    m_d.process_topology_2_z.n[2], // int idist,
                                    FFTW_ADDR(m_buf3),
                                    NULL, // const int *onembed,
                                    1, // int ostride,
                                    m_d.process_topology_2_z.n[2], // int odist,
                                    FFTW_BACKWARD, // int sign,
                                    0); // unsigned flags
#elif defined(FFTW3)
    if (transposed_order) {
      flags_f |= FFTW_MPI_TRANSPOSED_OUT;
      flags_b |= FFTW_MPI_TRANSPOSED_IN;
    }

#if CERRILLOS_SLAB_HACK == 1
    //attempt to make mpi fftw 3.3 work on more of cerrillos
    //because of green's function application, 
    //does not require changes to simulation code
    flags_f |= FFTW_DESTROY_INPUT;
    flags_b |= FFTW_DESTROY_INPUT;
#endif

    m_plan_f = fftw_mpi_plan_dft_3d(m_d.n[0], m_d.n[1], m_d.n[2],
				    FFTW_ADDR(m_buf1), FFTW_ADDR(m_buf2),
				    comm, FFTW_FORWARD, flags_f);
    m_plan_b = fftw_mpi_plan_dft_3d(m_d.n[0], m_d.n[1], m_d.n[2],
				    FFTW_ADDR(m_buf1), FFTW_ADDR(m_buf3),
				    comm, FFTW_BACKWARD, flags_b);
#else
    m_plan_f = fftw3d_mpi_create_plan(comm, m_d.n[0], m_d.n[1], m_d.n[2], FFTW_FORWARD, flags_f);
    m_plan_b = fftw3d_mpi_create_plan(comm, m_d.n[0], m_d.n[1], m_d.n[2], FFTW_BACKWARD, flags_b);
#endif

  }
  
  ///
  // Solve for the potential by applying the Green's function to the
  // density (in k-space)
  //   rho           density (input)
  //   phi           potential (output)
  ///
  void kspace_solve(const complex_t *rho, complex_t *phi)
  {
    int k[3];
    int index;
    
    initialize_greens_function();

    int local_dim[3]; 
    int self_coord[3];
#if defined(FFTW3) && defined(PENCIL)
    self_coord[0]=self_2d_z(0);
    self_coord[1]=self_2d_z(1);
    self_coord[2]=self_2d_z(2);
    local_dim[0]=local_ng_2d_z(0);
    local_dim[1]=local_ng_2d_z(1);
    local_dim[2]=local_ng_2d_z(2);
#else
    self_coord[0]=self_1d(0);
    self_coord[1]=self_1d(1);
    self_coord[2]=self_1d(2);
    local_dim[0]=local_ng_1d(0);
    local_dim[1]=local_ng_1d(1);
    local_dim[2]=local_ng_1d(2);
#endif
    
    index = 0;
    for (int local_k0 = 0; local_k0 < local_dim[0]; ++local_k0) {
      k[0] = local_k0 + self_coord[0] * local_dim[0];
      for (int local_k1 = 0; local_k1 < local_dim[1]; ++local_k1) {
          k[1] = local_k1 + self_coord[1] * local_dim[1];
	for (int local_k2 = 0;  local_k2 < local_dim[2]; ++local_k2) {
          k[2] = local_k2 + self_coord[2] * local_dim[2];
	  phi[index] = m_green[index] * rho[index];
	  index++;
	}
	index += m_d.padding[2];
      }
      index += m_d.padding[1];
    }
  }
  
  ///
  // Solve for the gradient of the potential along the given axis by
  // applying the derivative Green's function to the density (in
  // k-space)
  //   axis          the axis along which to take the gradient
  //   rho           density (input)
  //   grad_phi      the gradient of the potential (output)
  ///
  void kspace_solve_gradient(int axis, const complex_t *rho, complex_t *grad_phi)
  {
    int k[3];
    int index;

    initialize_greens_function();

    int local_dim[3]; 
    int self_coord[3];
#if defined(FFTW3) && defined(PENCIL)
    self_coord[0]=self_2d_z(0);
    self_coord[1]=self_2d_z(1);
    self_coord[2]=self_2d_z(2);
    local_dim[0]=local_ng_2d_z(0);
    local_dim[1]=local_ng_2d_z(1);
    local_dim[2]=local_ng_2d_z(2);
#else
    self_coord[0]=self_1d(0);
    self_coord[1]=self_1d(1);
    self_coord[2]=self_1d(2);
    local_dim[0]=local_ng_1d(0);
    local_dim[1]=local_ng_1d(1);
    local_dim[2]=local_ng_1d(2);
#endif
    
    index = 0;
    for (int local_k0 = 0; local_k0 < local_dim[0]; ++local_k0) {
      k[0] = local_k0 + self_coord[0] * local_dim[0];
      for (int local_k1 = 0; local_k1 < local_dim[1]; ++local_k1) {
          k[1] = local_k1 + self_coord[1] * local_dim[1];
	for (int local_k2 = 0;  local_k2 < local_dim[2]; ++local_k2) {
          k[2] = local_k2 + self_coord[2] * local_dim[2];
	  grad_phi[index] = I * (- m_gradient[k[axis]]) * m_green[index] * rho[index];
	  index++;
	}
	index += m_d.padding[2];
      }
      index += m_d.padding[1];
    }
  }
  
  ///
  // Allocate and pre-calculate isotropic Green's function and
  // gradient operator.
  ///
  virtual void initialize_greens_function() = 0;
  
protected:
  double max(double a, double b) { return a > b ? a : b; }
  std::vector<double> m_green; // green's function
  std::vector<double> m_gradient; //imaginary part of the gradient in grid units
  std::vector<double> m_pk_cic;
  std::vector<double> m_pk_weight;
  std::vector<int>    m_pk_ksq;
  std::vector<complex_t, bigchunk_allocator<complex_t> > m_buf1;
  std::vector<complex_t, bigchunk_allocator<complex_t> > m_buf2;
  std::vector<complex_t, bigchunk_allocator<complex_t> > m_buf3;
#if defined(FFTW3) && defined(PENCIL)
  fftw_plan m_plan_f_x;
  fftw_plan m_plan_f_y;
  fftw_plan m_plan_f_z;
  fftw_plan m_plan_b_x;
  fftw_plan m_plan_b_y;
  fftw_plan m_plan_b_z;
#elif defined(FFTW3)
  fftw_plan m_plan_f;
  fftw_plan m_plan_b;
#else
  fftwnd_mpi_plan m_plan_f;
  fftwnd_mpi_plan m_plan_b;
#endif
  bool m_greens_functions_initialized;
};


///
//  Poison solver using a 2nd-order discrete Green's function, and a
//  2nd-order derivative.
//
//  G(k) = 1 / (2 * (Sum_i cos(2 pi k_i / n) - 3))
//  (D_i f)(k) =  ( - i * sin(2 pi k / n) / (2 pi / n) )* f(k)
///
class SolverDiscrete : public SolverBase {

public:

  SolverDiscrete(MPI_Comm comm, int ng)
    : SolverBase(comm, ng)
  {
  }
  
  SolverDiscrete(MPI_Comm comm, std::vector<int> n)
    : SolverBase(comm, n)
  {
  }
  
  ///
  // Allocate and pre-calculate isotropic Green's function (1D
  // distribution) and trigonometric factors:
  //   1 / (2 * (Sum_i cos(2 pi k_i / n) - 3))
  ///
  void initialize_greens_function()
  {
    double kstep;
    int index;
    int k[3];
    std::vector<double> cosine;
    
    if (m_greens_functions_initialized) {
      return;
    }
    
    m_greens_functions_initialized = true;
    
    m_green.resize(local_size());
    m_gradient.resize(m_d.n[0]);
    cosine.resize(m_d.n[0]);
    
    // cache trigonometric factors and imaginary part of gradient
    kstep = 2.0 * pi / (double) m_d.n[0];
    for (int kk = 0; kk < m_d.n[0]; ++kk) {
      cosine[kk] = cos(kk * kstep);
      m_gradient[kk] = sin(kk * kstep); // imaginary part of gradient
    }
    
    // cache isotropic Green's function (1D or 2D distribution)
    int local_dim[3]; 
    int self_coord[3];
#if defined(FFTW3) && defined(PENCIL)
    self_coord[0]=self_2d_z(0);
    self_coord[1]=self_2d_z(1);
    self_coord[2]=self_2d_z(2);
    local_dim[0]=local_ng_2d_z(0);
    local_dim[1]=local_ng_2d_z(1);
    local_dim[2]=local_ng_2d_z(2);
#else
    self_coord[0]=self_1d(0);
    self_coord[1]=self_1d(1);
    self_coord[2]=self_1d(2);
    local_dim[0]=local_ng_1d(0);
    local_dim[1]=local_ng_1d(1);
    local_dim[2]=local_ng_1d(2);
#endif
    index = 0;
    double coeff = 0.5 / double(global_size());
    for (int local_k0 = 0; local_k0 < local_dim[0]; ++local_k0) {
      k[0] = local_k0 + self_coord[0] * local_dim[0];
      for (int local_k1 = 0; local_k1 < local_dim[1]; ++local_k1) {
          k[1] = local_k1 + self_coord[1] * local_dim[1];
	for (int local_k2 = 0;  local_k2 < local_dim[2]; ++local_k2) {
          k[2] = local_k2 + self_coord[2] * local_dim[2];
	  m_green[index] = coeff / (cosine[k[0]] + cosine[k[1]] + cosine[k[2]] - 3.0);
	  index++;
	}
	index += m_d.padding[2];
      }
      index += m_d.padding[1];
    }
    // handle the pole
    if (self() == 0) {
      m_green[0] = 0.0;
    }
  }
};


///
//  Poison solver using a 6th-order discrete Green's function with a
//  Gaussian noise-quieting filter function, and a 4th-order
//  derivative.
//
//    G(k) = W(k) * 45/128 / Sum_i [ cos (2 pi k / n)
//                                 - 5/64 cos(4 pi k / n)
//                                 + 1/1024 cos(8 pi k_i / n)
//                                 - 945/1024 ]
//  or
//    G(k) = W(k) / Sum_i ( a0 + a1 cos (2 pi k / n)
//                             + a2 cos (4 pi k / n)
//                             + a3 cos (8 pi k / n) )
//  where
//     a0 = -21/8
//     a1 =  128/45
//     a2 = -2/9
//     a3 =  1/360
//  satisfying
//     a0 + a1 + a2 + a3 = 0
//     (a1 + 4 a2 + 16 a3) = 2
//  and
//    W(k) = exp( - k^2 sigma^2 / 4) [ sin(k/2) / (k/2)]^n_s
//    sigma = 0.8
//    n_s = 3
//
//  The gradient operator (imaginary part, grid units) is
//    (gradient f)(k) =  b1 sin(2 pi k / n) + b2 sin( 4 pi k / n) f(k)
//  where
//     b1 = 4/3
//     b2 = -1/6
//  satisfying
//     b1 + 2 b2 = 1
//     b1 + 8 b2 = 0
///
class SolverQuiet : public SolverBase {

public:

  SolverQuiet(MPI_Comm comm, int ng)
    : SolverBase(comm, ng)
  {
  }
  
  SolverQuiet(MPI_Comm comm, std::vector<int> n)
    : SolverBase(comm, n)
  {
  }
  
  ///
  // Allocate and pre-calculate isotropic Green's function (1D
  // distribution) and trigonometric factors:
  //   1 / (2 * (Sum_i cos(2 pi k_i / n) - 3))
  ///
  void initialize_greens_function()
  {
    double kstep;
    int index;
    int k[3];
    std::vector<double> c1;
    std::vector<double> c2;
    std::vector<double> c3;
    std::vector<double> filter;
    std::vector<double> kperiodic;
    double const a0 = - 21.0 / 8.0;
    double const a1 =   128.0 / 45.0;
    double const a2 = - 2.0 / 9.0;
    double const a3 =   1.0 / 360.0;
    double const b1 =   4.0 / 3.0;
    double const b2 = - 1.0 / 6.0;
    double const sigma = 0.8;
    double const ns = 3.0;
    int ng = m_d.n[0];
    
    if (m_greens_functions_initialized) {
      return;
    }
    
    // check Taylor series coefficent conditions
    assert(fabs(a0 + a1 + a2 + a3) < 1.0e-12);
    assert(fabs(a1 + 4 * a2 + 16 * a3 - 2.0) < 1.0e-12);
    assert(fabs(b1 + 2 * b2 - 1.0) < 1.0e-12);
    assert(fabs(b1 + 8 * b2) < 1.0e-12);
    
    m_greens_functions_initialized = true;
    
    m_green.resize(local_size());
    m_gradient.resize(ng);
    c1.resize(ng);
    c2.resize(ng);
    c3.resize(ng);
    kperiodic.resize(ng);
    filter.resize(ng);
    
    // cache k array with the correct periodicity
    // cache trigonometric factors and imaginary part of gradient
    kstep = 2.0 * pi / static_cast<double>(ng);
    for (int kk = 0; kk < ng; ++kk) {
      c1[kk] = cos(kk * kstep);
      c2[kk] = cos(2 * kk * kstep);
      c3[kk] = cos(4 * kk * kstep);
      if (kk < ng / 2) {
	kperiodic[kk] = kk * kstep;
	kperiodic[kk + ng / 2] = (kk - ng / 2) * kstep;
      }
    }
    
    // cache Green's function filter, and 4th order k-space gradient operator
    filter[0] = 1.0;
    m_gradient[0] = 1.0;
    for (int kk = 1; kk < ng; ++kk) {
      filter[kk] = exp(- 0.25 * sigma * sigma * kperiodic[kk] * kperiodic[kk])
	* pow(sin(0.5 * kperiodic[kk]) / (0.5 * kperiodic[kk]), ns);
      m_gradient[kk] = (b1 * sin(kperiodic[kk]) + b2 * sin(2 * kperiodic[kk]));
    }
    
    // cache isotropic Green's function (1D or 2D distribution)
    int local_dim[3]; 
    int self_coord[3];
#if defined(FFTW3) && defined(PENCIL)
    self_coord[0]=self_2d_z(0);
    self_coord[1]=self_2d_z(1);
    self_coord[2]=self_2d_z(2);
    local_dim[0]=local_ng_2d_z(0);
    local_dim[1]=local_ng_2d_z(1);
    local_dim[2]=local_ng_2d_z(2);
#else
    self_coord[0]=self_1d(0);
    self_coord[1]=self_1d(1);
    self_coord[2]=self_1d(2);
    local_dim[0]=local_ng_1d(0);
    local_dim[1]=local_ng_1d(1);
    local_dim[2]=local_ng_1d(2);
#endif
    index = 0;
    double coeff = 1.0 / double(global_size());
    for (int local_k0 = 0; local_k0 < local_dim[0]; ++local_k0) {
      k[0] = local_k0 + self_coord[0] * local_dim[0];
      double d0 = a0 + a1 * c1[k[0]] + a2 * c2[k[0]] + a3 * c3[k[0]];
      for (int local_k1 = 0; local_k1 < local_dim[1]; ++local_k1) {
          k[1] = local_k1 + self_coord[1] * local_dim[1];
	  double d1 = a0 + a1 * c1[k[1]] + a2 * c2[k[1]] + a3 * c3[k[1]];
	for (int local_k2 = 0;  local_k2 < local_dim[2]; ++local_k2) {
          k[2] = local_k2 + self_coord[2] * local_dim[2];
	  double filt = coeff * filter[k[0]] * filter[k[1]] * filter[k[2]];
	  double d2 = a0 + a1 * c1[k[2]] + a2 * c2[k[2]] + a3 * c3[k[2]];
	  m_green[index] = filt / (d0 + d1 + d2);
	  index++;
	}
	index += m_d.padding[2];
      }
      index += m_d.padding[1];
    }
    
    // handle the pole
    if (self() == 0) {
      m_green[0] = 0.0;
    }
  }
};


///
//  Poisson solver class using continuum Green's function:
//    - 1 / k^2
///
class SolverContinuum : public SolverBase {

public:

  virtual void this_class_is_not_yet_tested_and_should_not_be_used() = 0;
  
  SolverContinuum(MPI_Comm comm, int ng)
    : SolverBase(comm, ng)
  {
  }
  
  SolverContinuum(MPI_Comm comm, std::vector<int> n)
    : SolverBase(comm, n)
  {
  }
  
  ///
  // Allocate and pre-calculate isotropic Green's function and gradient.
  ///
  void initialize_greens_function()
  {
    double kstep;
    double coeff;
    int index;
    int k[3];
    int ng = m_d.n[0];
    
    if (m_greens_functions_initialized) {
      return;
    }
    
    m_greens_functions_initialized = true;
    
    m_green.resize(local_size());
    m_gradient.resize(ng);
    
    // cache imaginary part of gradient, imposing symmetries by hand
    kstep = 2.0 * pi / (double) ng;
    for (int kk = 0; kk < ng / 2; ++kk) {
      m_gradient[kk] = kk * kstep;
      m_gradient[kk + ng / 2] = (kk - ng / 2) * kstep;
    }
    
    // cache isotropic Green's function (1D or 2D distribution)
    int local_dim[3]; 
    int self_coord[3];
#if defined(FFTW3) && defined(PENCIL)
    self_coord[0]=self_2d_z(0);
    self_coord[1]=self_2d_z(1);
    self_coord[2]=self_2d_z(2);
    local_dim[0]=local_ng_2d_z(0);
    local_dim[1]=local_ng_2d_z(1);
    local_dim[2]=local_ng_2d_z(2);
#else
    self_coord[0]=self_1d(0);
    self_coord[1]=self_1d(1);
    self_coord[2]=self_1d(2);
    local_dim[0]=local_ng_1d(0);
    local_dim[1]=local_ng_1d(1);
    local_dim[2]=local_ng_1d(2);
#endif
    index = 0;
    coeff = -1.0 / double(global_size());
    for (int local_k0 = 0; local_k0 < local_dim[0]; ++local_k0) {
      k[0] = local_k0 + self_coord[0] * local_dim[0];
      double k0sq = m_gradient[k[0]] * m_gradient[k[0]];
      for (int local_k1 = 0; local_k1 < local_dim[1]; ++local_k1) {
          k[1] = local_k1 + self_coord[1] * local_dim[1];
	  double k1sq = m_gradient[k[1]] * m_gradient[k[1]];
	for (int local_k2 = 0;  local_k2 < local_dim[2]; ++local_k2) {
          k[2] = local_k2 + self_coord[2] * local_dim[2];
	  double k2sq = m_gradient[k[2]] * m_gradient[k[2]];
	  m_green[index] = coeff / (k0sq + k1sq + k2sq);
	  index++;
	}
	index += m_d.padding[2];
      }
      index += m_d.padding[1];
    }
    // handle the pole
    if (self() == 0) {
      m_green[0] = 0.0;
    }
  }
};

#endif
