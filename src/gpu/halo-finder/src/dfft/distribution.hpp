#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include <vector>

///
// Distribution / partition / decomposition of data
//
// A C++ wrapper around distribution.h
///

#include "complex-type.h"
#include "distribution.h"

class Distribution {

public:
    Distribution()
        {
        }

    Distribution(MPI_Comm comm, int const n[], int n_padded[], bool debug = false)
        : m_debug(debug)
        {
            initialize(comm, n, n_padded);
        }

    Distribution(MPI_Comm comm, int const n[], bool debug = false)
        : m_debug(debug)
        {
            initialize(comm, n, n);
        }

    Distribution(MPI_Comm comm, std::vector<int> const & n, bool debug = false)
        : m_debug(debug)
        {
            initialize(comm, &n[0], &n[0]);
        }

    Distribution(MPI_Comm comm, int ng, bool debug = false)
        : m_debug(debug)
        {
            int n[3] = { ng, ng, ng };
            initialize(comm, n, n);
        }

    virtual ~Distribution()
        {
            distribution_fini(&m_d);
        }

    void initialize(MPI_Comm comm, int const n[], int const n_padded[])
        {
            int flag;
            MPI_Initialized(&flag);
            if (flag == 0) {
                MPI_Init(0, 0);
            }
            distribution_init(comm, n, n_padded, &m_d, m_debug);
        }

    void redistribute_1_to_3(const complex_t *a, complex_t *b)
        {
            distribution_1_to_3(a, b, &m_d);
        }

    void redistribute_1_to_3(std::vector<complex_t> const & a,
                             std::vector<complex_t> & b)
        {
            distribution_1_to_3(&a[0], &b[0], &m_d);
        }

    void redistribute_3_to_1(const complex_t *a, complex_t *b)
        {
            distribution_3_to_1(a, b, &m_d);
        }

    void redistribute_3_to_1(std::vector<complex_t> const & a,
                             std::vector<complex_t> & b)
        {
            distribution_3_to_1(&a[0], &b[0], &m_d);
        }

    void redistribute_2_to_3(const complex_t *a, complex_t *b, int axis)
        {
            distribution_2_to_3(a, b, &m_d, axis);
        }

    void redistribute_2_to_3(std::vector<complex_t> const & a,
                             std::vector<complex_t> & b, int axis)
        {
            distribution_2_to_3(&a[0], &b[0], &m_d, axis);
        }

    void redistribute_3_to_2(const complex_t *a, complex_t *b, int axis)
        {
            distribution_3_to_2(a, b, &m_d, axis);
        }

    void redistribute_3_to_2(std::vector<complex_t> const & a,
                             std::vector<complex_t> & b, int axis)
        {
            distribution_3_to_2(&a[0], &b[0], &m_d, axis);
        }

    size_t local_size() const
        {
            size_t size = 1;
            for (int i = 0; i < 3; ++i) {
                size *= (m_d.n[i] / m_d.process_topology_3.nproc[i]);
            }
            return size;
        }

    size_t global_size() const
        {
            size_t size = 1;
            for (int i = 0; i < 3; ++i) {
                size *= m_d.n[i];
            }
            return size;
        }

    int global_ng(int i) const
        {
            return m_d.n[i];
        }

    int local_ng_1d(int i) const
        {
            return m_d.process_topology_1.n[i];
        }

    int local_ng_2d_x(int i) const
        {
            return m_d.process_topology_2_x.n[i];
        }

    int local_ng_2d_y(int i) const
        {
            return m_d.process_topology_2_y.n[i];
        }

    int local_ng_2d_z(int i) const
        {
            return m_d.process_topology_2_z.n[i];
        }

    int local_ng_3d(int i) const
        {
            return m_d.process_topology_3.n[i];
        }

    int nproc() const
        {
            return m_d.process_topology_1.nproc[0];
        }

    int const (& nproc_1d() const)[3]
        {
            return m_d.process_topology_1.nproc;
        }

    int const (& nproc_2d_x() const)[3]
        {
            return m_d.process_topology_2_x.nproc;
        }

    int const (& nproc_2d_y() const)[3]
        {
            return m_d.process_topology_2_y.nproc;
        }

    int const (& nproc_2d_z() const)[3]
        {
            return m_d.process_topology_2_z.nproc;
        }

    int const (& nproc_3d() const)[3]
        {
            return m_d.process_topology_3.nproc;
        }

    int nproc_1d(int i) const
        {
            return m_d.process_topology_1.nproc[i];
        }

    int nproc_2d_x(int i) const
        {
            return m_d.process_topology_2_x.nproc[i];
        }

    int nproc_2d_y(int i) const
        {
            return m_d.process_topology_2_y.nproc[i];
        }

    int nproc_2d_z(int i) const
        {
            return m_d.process_topology_2_z.nproc[i];
        }

    int nproc_3d(int i) const
        {
            return m_d.process_topology_3.nproc[i];
        }

    int self() const
        {
            return m_d.process_topology_1.self[0];
        }

    int const (& self_1d() const)[3]
        {
            return m_d.process_topology_1.self;
        }

    int const (& self_2d_x() const)[3]
        {
            return m_d.process_topology_2_x.self;
        }

    int const (& self_2d_y() const)[3]
        {
            return m_d.process_topology_2_y.self;
        }

    int const (& self_2d_z() const)[3]
        {
            return m_d.process_topology_2_z.self;
        }

    int const (& self_3d() const)[3]
        {
            return m_d.process_topology_3.self;
        }

    int self_1d(int i) const
        {
            return m_d.process_topology_1.self[i];
        }

    int self_2d_x(int i) const
        {
            return m_d.process_topology_2_x.self[i];
        }

    int self_2d_y(int i) const
        {
            return m_d.process_topology_2_y.self[i];
        }

    int self_2d_z(int i) const
        {
            return m_d.process_topology_2_z.self[i];
        }

    int self_3d(int i) const
        {
            return m_d.process_topology_3.self[i];
        }

    MPI_Comm cart_1d() const
        {
            return m_d.process_topology_1.cart;
        }

    MPI_Comm cart_2d_x() const
        {
            return m_d.process_topology_2_x.cart;
        }

    MPI_Comm cart_2d_y() const
        {
            return m_d.process_topology_2_y.cart;
        }

    MPI_Comm cart_2d_z() const
        {
            return m_d.process_topology_2_z.cart;
        }

    MPI_Comm cart_3d() const
        {
            return m_d.process_topology_3.cart;
        }

    int rank_2d_x(int c[])
        {
            int r;

            Rank_x_pencils(&r, c, &m_d);
            return r;
        }

    int rank_2d_y(int c[])
        {
            int r;

            Rank_y_pencils(&r, c, &m_d);
            return r;
        }

    int rank_2d_z(int c[])
        {
            int r;

            Rank_z_pencils(&r, c, &m_d);
            return r;
        }

     void coords_2d_x(int r, int c[])
        {
            Coord_x_pencils(r, c, &m_d);
        }

     void coords_2d_y(int r, int c[])
        {
            Coord_y_pencils(r, c, &m_d);
        }

     void coords_2d_z(int r, int c[])
        {
            Coord_z_pencils(r, c, &m_d);
        }

protected:
    distribution_t m_d;
    bool m_debug;
};

#endif
