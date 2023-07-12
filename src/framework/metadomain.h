#ifndef FRAMEWORK_METADOMAIN_H
#define FRAMEWORK_METADOMAIN_H

#include "wrapper.h"

#if defined(MPI_ENABLED)
#  include <mpi.h>
#endif    // MPI_ENABLED

namespace ntt {
  template <Dimension D, typename T>
  using nd_tuple_t = typename std::conditional<
    D == Dim1,
    std::tuple<T>,
    typename std::conditional<
      D == Dim2,
      std::tuple<T, T>,
      typename std::conditional<D == Dim3, std::tuple<T, T, T>, std::nullptr_t>::type>::type>::type;

  template <Dimension D>
  using domaincoordinate_t = nd_tuple_t<D, unsigned int>;

  template <Dimension D>
  using direction_t                                  = nd_tuple_t<D, short>;

  const std::vector<direction_t<Dim1>> directions_1d = { { -1 }, { 1 } };
  const std::vector<direction_t<Dim2>> directions_2d = {
    {-1,  0},
    { 1,  0},
    { 0, -1},
    { 0,  1},
    {-1, -1},
    {-1,  1},
    { 1, -1},
    { 1,  1}
  };
  const std::vector<direction_t<Dim3>> directions_3d = {
    {-1,  0,  0},
    { 1,  0,  0},
    { 0, -1,  0},
    { 0,  1,  0},
    { 0,  0, -1},
    { 0,  0,  1},
    {-1, -1,  0},
    {-1,  1,  0},
    { 1, -1,  0},
    { 1,  1,  0},
    {-1,  0, -1},
    {-1,  0,  1},
    { 1,  0, -1},
    { 1,  0,  1},
    { 0, -1, -1},
    { 0, -1,  1},
    { 0,  1, -1},
    { 0,  1,  1},
    {-1, -1, -1},
    {-1, -1,  1},
    {-1,  1, -1},
    {-1,  1,  1},
    { 1, -1, -1},
    { 1, -1,  1},
    { 1,  1, -1},
    { 1,  1,  1}
  };

  template <Dimension D>
  class Domain {
    // index of the domain in the metadomain
    int                       m_index;
    // coordinate of the domain in the metadomain (# of domains in each dimension)
    std::vector<unsigned int> m_coordinate;
    // size of the domain (# of cells in each dimension)
    std::vector<unsigned int> m_ncells;
    // offset of the domain (# of cells in each dimension)
    std::vector<unsigned int> m_offset_cells;
    // extent of the domain (physical size in each dimension)
    std::vector<real_t>       m_extent;

#if defined(MPI_ENABLED)
    // MPI rank of the domain
    int m_mpi_rank;
#endif    // MPI_ENABLED

  public:
    std::map<direction_t<D>, std::shared_ptr<Domain<D>>> neighbors;

    Domain(const int&                       index,
           const std::vector<unsigned int>& coordinate,
           const std::vector<unsigned int>& ncells,
           const std::vector<unsigned int>& offset_cells,
           const std::vector<real_t>&       extent,
           const int&                       mpi_rank = 0)
      : m_index { index },
        m_coordinate { coordinate },
        m_ncells { ncells },
        m_offset_cells { offset_cells },
        m_extent { extent } {
#if defined(MPI_ENABLED)
      m_mpi_rank = mpi_rank;
#endif    // MPI_ENABLED
    }

#if defined(MPI_ENABLED)
    [[nodiscard]] auto mpiRank() const -> int {
      return m_mpi_rank;
    }
#endif    // MPI_ENABLED

    [[nodiscard]] auto coordinate() const -> std::vector<unsigned int> {
      return m_coordinate;
    }

    [[nodiscard]] auto ncells() const -> std::vector<unsigned int> {
      return m_ncells;
    }

    [[nodiscard]] auto offsetCells() const -> std::vector<unsigned int> {
      return m_offset_cells;
    }

    [[nodiscard]] auto extent() const -> std::vector<real_t> {
      return m_extent;
    }

    [[nodiscard]] auto index() const -> int {
      return m_index;
    }
  };

  template <Dimension D>
  class Metadomain {
    std::vector<unsigned int> m_global_ndomains;
    std::vector<unsigned int> m_global_ncells;
    std::vector<real_t>       m_global_extent;
#if defined(MPI_ENABLED)
    int m_mpisize;
    int m_mpirank;
#endif    // MPI_ENABLED

  public:
    std::vector<Domain<D>> domains;

    Metadomain(const std::vector<unsigned int>& global_ncells,
               const std::vector<real_t>&       global_extent,
               const std::vector<unsigned int>& global_decomposition)
      : m_global_ncells { global_ncells }, m_global_extent { global_extent } {
      /**
       * if MPI is enabled distribute domains among ranks
       * else just create one domain
       */
#if defined(MPI_ENABLED)
      MPI_Comm_size(MPI_COMM_WORLD, &m_mpisize);
      MPI_Comm_rank(MPI_COMM_WORLD, &m_mpirank);

      for (auto rnk { 0 }; rnk < m_mpisize; ++rnk) {
        // !TODO: distribute domains among ranks
        std::vector<unsigned int> l_coordinate;
        std::vector<unsigned int> l_ncells;
        std::vector<unsigned int> l_offset;
        std::vector<real_t>       l_extent;
        // index is the same as rank
        int                       l_index = rnk;

        // !TODO: define m_global_ndomains(i, j, k)

        domains.emplace_back(l_index, l_coordinate, l_ncells, l_offset, l_extent, rnk);
      }
#else     // MPI_ENABLED
      std::vector<unsigned int> no_coordinate;
      std::vector<unsigned int> no_offset;
      int                       no_index = 0;
      for (auto i { 0 }; i < (short)D; ++i) {
        no_coordinate.emplace_back(0);
        m_global_ndomains.emplace_back(1);
        no_offset.emplace_back(0);
      }
      domains.emplace_back(
        no_index, no_coordinate, m_global_ncells, no_offset, m_global_extent);
#endif    // MPI_ENABLED

      // assign neighbors
      // for (auto& domain : domains) {
      //   for (auto& direction : directions<D>) {
      //     // auto neighbor_coordinate { domain.m_coordinate };
      //     // auto neighbor_offset { domain.m_offset_cells };
      //     // auto neighbor_ncells { domain.m_ncells };
      //     // auto neighbor_extent { domain.m_extent };

      //     // for (auto i { 0 }; i < D; ++i) {
      //     //   neighbor_coordinate[i] += std::get<i>(direction);
      //     //   neighbor_offset[i] += std::get<i>(direction) * domain.m_ncells[i];
      //     // }

      //     // domains.emplace_back(
      //     //   domain.m_index + 1, neighbor_coordinate, neighbor_ncells, neighbor_offset,
      //     //   neighbor_extent);
      //   }
      // }
    }

    [[nodiscard]] auto globalNcells() const -> std::vector<unsigned int> {
      return m_global_ncells;
    }

    [[nodiscard]] auto globalExtent() const -> std::vector<real_t> {
      return m_global_extent;
    }

    [[nodiscard]] auto globalNdomains() const -> std::vector<unsigned int> {
      return m_global_ndomains;
    }

    auto domainByIndex(const int& index) const -> Domain<D> {
      return domains[index];
    }

    auto coord2index(const domaincoordinate_t<D>& d) const -> int {
      // !TODO: implement
      return 0;
    }

    auto index2coord(const int& index) const -> domaincoordinate_t<D> {
      // !TODO: implement
      domaincoordinate_t<D> d;
      for (auto i { 0 }; i < (short)D; ++i) {
        d[i] = 0;
      }
      return d;
    }

    auto localDomain() const -> Domain<D> {
#if defined(MPI_ENABLED)
      return domainByIndex(m_mpirank);
#else     // MPI_ENABLED
      return domainByIndex(0);
#endif    // MPI_ENABLED
    }

#if defined(MPI_ENABLED)
    [[nodiscard]] auto mpiSize() const -> int {
      return m_mpisize;
    }

    [[nodiscard]] auto mpiRank() const -> int {
      return m_mpirank;
    }
#endif    // MPI_ENABLED
  };

}    // namespace ntt

#endif    // FRAMEWORK_METADOMAIN_H