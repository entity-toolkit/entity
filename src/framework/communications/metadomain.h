#ifndef FRAMEWORK_COMM_METADOMAIN_H
#define FRAMEWORK_COMM_METADOMAIN_H

#include "wrapper.h"

#include "communications/decomposition.h"
#include "utils/qmath.h"
#include "utils/utils.h"

#include METRIC_HEADER

#include <iomanip>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

/**
 *
 * Metadomain
 * |---------|--------|------------------|------------------|
 * | D       | D      | D                | D                |
 * |         |        |                  |                  |
 * |---------|--------|------------------X------------------|
 * | D       | D      | Domain {index}   |\                 |
 * |         |        |                  | (extent[1],      |
 * |         |        |                  |  extent[3])      |
 * |         |        |                  |                  |
 * |         |        |                  |                  |
 * |         |        |                  |                  |
 * |         |        |                  |                  |
 * |<- offsetNcells ->|<---- ncells ---->|                  |
 * |---------|--------X------------------|------------------|
 * | D       | D      |\                 | D                |
 * |         |        | (extent[0],      |                  |
 * |         |        |  extent[2])      |                  |
 * |         |        |                  |                  |
 * |         |        |                  |                  |
 * |---------|--------|------------------|------------------|
 * ^                  ^
 * |--offsetNdomains--|
 *
 */

namespace ntt {

  template <Dimension D>
  class Domain {
    // index of the domain in the metadomain
    int                                         m_index;
    // offset of the domain in the metadomain (# of domains in each dimension)
    std::vector<unsigned int>                   m_offset_ndomains;
    // size of the domain (# of cells in each dimension)
    std::vector<unsigned int>                   m_ncells;
    // offset of the domain (# of cells in each dimension)
    std::vector<unsigned int>                   m_offset_ncells;
    // extent of the domain (physical size in each dimension)
    std::vector<real_t>                         m_extent;
    // boundary conditions of the domain
    std::vector<std::vector<BoundaryCondition>> m_boundaries;
    std::map<direction_t<D>, BoundaryCondition> m_boundaries_map;

    Metric<D> m_metric;

    // MPI rank of the domain (used only when MPI enabled)
    int m_mpi_rank;

    std::map<direction_t<D>, Domain<D>*> m_neighbors;

  public:
    Domain(const int&                                        index,
           const std::vector<unsigned int>&                  offset_ndomains,
           const std::vector<unsigned int>&                  ncells,
           const std::vector<unsigned int>&                  offset_ncells,
           const std::vector<real_t>&                        extent,
           const real_t*                                     metric_params,
           const std::vector<std::vector<BoundaryCondition>> boundaries,
           const int&                                        mpi_rank = -1) :
      m_index { index },
      m_offset_ndomains { offset_ndomains },
      m_ncells { ncells },
      m_offset_ncells { offset_ncells },
      m_extent { extent },
      m_boundaries { boundaries },
      m_metric { ncells, extent, metric_params },
      m_mpi_rank { mpi_rank } {}

#if defined(MPI_ENABLED)
    [[nodiscard]]
    auto mpiRank() const -> int {
      return m_mpi_rank;
    }
#endif // MPI_ENABLED

    auto assignNeighbor(const direction_t<D>& dir, Domain<D>* neighbor) -> void {
      m_neighbors[dir] = neighbor;
    }

    auto display() const -> void {
      std::cout << "Domain " << m_index << ":";
      std::cout << std::setw(20) << std::left << "\n  offset_ndomains: ";
      for (auto& off_nd : m_offset_ndomains) {
        std::cout << std::setw(10) << std::right << off_nd;
      }
      std::cout << std::setw(20) << std::left << "\n  ncells: ";
      for (auto& ncell : m_ncells) {
        std::cout << std::setw(10) << std::right << ncell;
      }
      std::cout << std::setw(20) << std::left << "\n  offset_ncells: ";
      for (auto& off_nc : m_offset_ncells) {
        std::cout << std::setw(10) << std::right << off_nc;
      }
      std::cout << std::setw(20) << std::left << "\n  extent: ";
      for (auto& ext : m_extent) {
        std::cout << std::setw(10) << std::right << ext;
      }
      std::cout << std::setw(20) << std::left << "\n  boundaries: ";
      for (auto& bound : m_boundaries) {
        for (auto& bc : bound) {
          std::cout << std::setw(10) << std::right
                    << stringizeBoundaryCondition(bc);
        }
      }
      std::cout << std::setw(20) << std::left << "\n  neighbors:";
      std::cout << "\n";
      for (auto& [dir, neighbor] : m_neighbors) {
        std::cout << "    ";
        for (auto& d : dir) {
          std::cout << std::setw(4) << std::right << d;
        }
        if (neighbor != nullptr) {
          std::cout << " -> " << neighbor->index() << "\n";
        } else {
          std::cout << " -> "
                    << "N/A"
                    << "\n";
        }
      }
      std::cout << "\n";
    }

    /**
     * Getters
     */

    [[nodiscard]]
    auto index() const -> int {
      return m_index;
    }

    [[nodiscard]]
    auto offsetNdomains() const -> const std::vector<unsigned int>& {
      return m_offset_ndomains;
    }

    [[nodiscard]]
    auto ncells() const -> const std::vector<unsigned int>& {
      return m_ncells;
    }

    [[nodiscard]]
    auto offsetNcells() const -> const std::vector<unsigned int>& {
      return m_offset_ncells;
    }

    [[nodiscard]]
    auto extent() const -> const std::vector<real_t>& {
      return m_extent;
    }

    [[nodiscard]]
    auto boundaries() const -> const std::vector<std::vector<BoundaryCondition>>& {
      return m_boundaries;
    }

    [[nodiscard]]
    auto neighbors(const direction_t<D>& dir) const -> const Domain<D>* {
      auto it = m_neighbors.find(dir);
      if (it != m_neighbors.end()) {
        return it->second;
      } else {
        NTTHostError("Neighbor not found");
      }
    }

    [[nodiscard]]
    auto boundaryIn(const direction_t<D>& dir) const -> BoundaryCondition {
      auto it = m_boundaries_map.find(dir);
      if (it != m_boundaries_map.end()) {
        return it->second;
      } else {
        NTTHostError("Boundary not found");
      }
    }

    [[nodiscard]]
    auto metric() const -> const Metric<D>& {
      return m_metric;
    }

    auto setBoundary(const direction_t<D>& dir, BoundaryCondition bc) -> void {
      m_boundaries_map[dir] = bc;
    }
  };

  template <Dimension D>
  class Metadomain {
    std::vector<unsigned int>                   m_global_ncells;
    std::vector<real_t>                         m_global_extent;
    Metric<D>                                   m_global_metric;
    std::vector<std::vector<BoundaryCondition>> m_global_boundaries;

    unsigned int                                      m_global_ndomains;
    std::vector<unsigned int>                         m_global_ndomains_per_dim;
    std::vector<std::vector<unsigned int>>            m_domain_offsets;
    std::map<std::vector<unsigned int>, unsigned int> m_domain_indices;
    real_t m_smallest_cell_size, m_fiducial_cell_volume;

#if defined(MPI_ENABLED)
    int m_mpisize;
    int m_mpirank;
#endif // MPI_ENABLED

  public:
    std::vector<Domain<D>> domains;

    Metadomain(const std::vector<unsigned int>& global_ncells,
               const std::vector<real_t>&       global_extent,
               const std::vector<unsigned int>& global_decomposition,
               const real_t*                    metric_params,
               const std::vector<std::vector<BoundaryCondition>>& global_boundaries,
               const bool allow_multidomain = false) :
      m_global_ncells { global_ncells },
      m_global_extent { global_extent },
      m_global_metric { global_ncells, global_extent, metric_params },
      m_global_boundaries { global_boundaries } {
#if defined(MPI_ENABLED)
      MPI_Comm_size(MPI_COMM_WORLD, &m_mpisize);
      MPI_Comm_rank(MPI_COMM_WORLD, &m_mpirank);
      m_global_ndomains = global_decomposition.empty()
                            ? m_mpisize
                            : std::accumulate(global_decomposition.begin(),
                                              global_decomposition.end(),
                                              1,
                                              std::multiplies<unsigned int>());
      NTTHostErrorIf(!allow_multidomain && ((int)m_global_ndomains != m_mpisize),
                     "ndomains != mpisize is not allowed");
      NTTHostErrorIf((int)m_global_ndomains < m_mpisize,
                     "ndomains < mpisize is not possible");
#else  // not MPI_ENABLED
      m_global_ndomains = global_decomposition.empty()
                            ? 1
                            : std::accumulate(global_decomposition.begin(),
                                              global_decomposition.end(),
                                              1,
                                              std::multiplies<unsigned int>());
      NTTHostErrorIf(!allow_multidomain and (m_global_ndomains != 1),
                     "ndomains > 1 is not allowed");
#endif // MPI_ENABLED

      auto d_ncells = Decompose(m_global_ndomains,
                                m_global_ncells,
                                global_decomposition);
      NTTHostErrorIf(d_ncells.size() != (short)D, "Invalid number of dimensions");
      auto d_offset_ncells = std::vector<std::vector<unsigned int>> {};
      auto d_offset_ndoms  = std::vector<std::vector<unsigned int>> {};
      for (auto& d : d_ncells) {
        m_global_ndomains_per_dim.push_back(d.size());
        auto offset_ncell = std::vector<unsigned int> { 0 };
        auto offset_ndom  = std::vector<unsigned int> { 0 };
        for (std::size_t i { 1 }; i < d.size(); ++i) {
          auto di = d[i - 1];
          offset_ncell.push_back(offset_ncell.back() + di);
          offset_ndom.push_back(offset_ndom.back() + 1);
        }
        d_offset_ncells.push_back(offset_ncell);
        d_offset_ndoms.push_back(offset_ndom);
      }

      // compute tensor products of the domain decompositions
      auto domain_ncells        = TensorProduct<unsigned int>(d_ncells);
      auto domain_offset_ncells = TensorProduct<unsigned int>(d_offset_ncells);
      auto domain_offset_ndoms  = TensorProduct<unsigned int>(d_offset_ndoms);

      m_domain_offsets = domain_offset_ndoms;

      // create domains
      for (std::size_t index { 0 }; index < m_global_ndomains; ++index) {
        auto l_offset_ndomains = domain_offset_ndoms[index];
        auto l_ncells          = domain_ncells[index];
        auto l_offset_ncells   = domain_offset_ncells[index];
        auto l_extent          = std::vector<real_t> {};
        auto l_boundaries = std::vector<std::vector<BoundaryCondition>> { {} };
        coord_t<D> low_corner_cu { ZERO }, up_corner_cu { ZERO },
          low_corner_ph { ZERO }, up_corner_ph { ZERO };
        for (auto d { 0 }; d < (short)D; ++d) {
          low_corner_cu[d] = (real_t)l_offset_ncells[d];
          up_corner_cu[d]  = (real_t)(l_offset_ncells[d] + l_ncells[d]);
        }
        m_global_metric.x_Code2Phys(low_corner_cu, low_corner_ph);
        m_global_metric.x_Code2Phys(up_corner_cu, up_corner_ph);

        for (auto d { 0 }; d < (short)D; ++d) {
          l_boundaries.push_back(std::vector<BoundaryCondition> {});
          l_extent.push_back(low_corner_ph[d]);
          l_extent.push_back(up_corner_ph[d]);
          if (l_offset_ndomains[d] != 0) {
            l_boundaries[d].push_back(BoundaryCondition::COMM);
          } else {
            l_boundaries[d].push_back(global_boundaries[d][0]);
          }
          if (l_offset_ndomains[d] + 1 != m_global_ndomains_per_dim[d]) {
            l_boundaries[d].push_back(BoundaryCondition::COMM);
          } else {
            l_boundaries[d].push_back(global_boundaries[d].size() > 1
                                        ? global_boundaries[d][1]
                                        : global_boundaries[d][0]);
          }
        }
        m_domain_indices[l_offset_ndomains] = index;
        domains.emplace_back(index,
                             l_offset_ndomains,
                             l_ncells,
                             l_offset_ncells,
                             l_extent,
                             metric_params,
                             l_boundaries,
                             index);
      }
      // populate the neighbors
      for (std::size_t index { 0 }; index < m_global_ndomains; ++index) {
        auto current_offset = domains[index].offsetNdomains();
        for (auto& direction : Directions<D>::all) {
          auto neighbor_offset = current_offset;
          auto no_neighbor     = false;
          for (auto d { 0 }; d < (short)D; ++d) {
            auto dir = direction[d];
            if ((dir == -1) && (current_offset[d] == 0)) {
              neighbor_offset[d] = m_global_ndomains_per_dim[d] - 1;
            } else if ((dir == 1) &&
                       (current_offset[d] == m_global_ndomains_per_dim[d] - 1)) {
              neighbor_offset[d] = 0;
            } else {
              neighbor_offset[d] += dir;
            }
            if ((dir == -1) &&
                (domains[index].boundaries()[d][0] != BoundaryCondition::COMM) &&
                (domains[index].boundaries()[d][0] != BoundaryCondition::PERIODIC)) {
              no_neighbor = true;
            }
            if ((dir == 1) &&
                (domains[index].boundaries()[d][1] != BoundaryCondition::COMM) &&
                (domains[index].boundaries()[d][1] != BoundaryCondition::PERIODIC)) {
              no_neighbor = true;
            }
          }
          domains[index].setBoundary(direction,
                                     no_neighbor ? BoundaryCondition::UNDEFINED
                                                 : BoundaryCondition::COMM);
          domains[index].assignNeighbor(
            direction,
            no_neighbor ? nullptr : &domains[offset2index(neighbor_offset)]);
        }
      }
      m_smallest_cell_size = m_global_metric.dxMin();
      coord_t<D> x_corner { ZERO };
      for (short d { 0 }; d < (short)D; ++d) {
        x_corner[d] = HALF;
      }
      m_fiducial_cell_volume = m_global_metric.sqrt_det_h(x_corner);
      // sanity check
      NTTHostErrorIf(!AlmostEqual(m_fiducial_cell_volume,
                                  domains[0].metric().sqrt_det_h(x_corner)),
                     fmt::format("fiducial cell volume is not the same across "
                                 "all domains: %.6e != %.6e",
                                 m_fiducial_cell_volume,
                                 domains[0].metric().sqrt_det_h(x_corner)));

#if defined(MPI_ENABLED)
      // sanity check
      auto smallest_cell_sizes       = std::vector<real_t>(m_global_ndomains);
      smallest_cell_sizes[m_mpirank] = m_smallest_cell_size;
      MPI_Allgather(&smallest_cell_sizes[m_mpirank],
                    1,
                    mpi_get_type<real_t>(),
                    smallest_cell_sizes.data(),
                    1,
                    mpi_get_type<real_t>(),
                    MPI_COMM_WORLD);
      for (const auto& sz : smallest_cell_sizes) {
        NTTHostErrorIf(
          !AlmostEqual(sz, m_smallest_cell_size),
          "smallest cell size is not the same across all MPI ranks");
      }

#endif
    }

    auto domainByIndex(const int& index) const -> const Domain<D>* {
      return &(domains[index]);
    }

    auto domainByOffset(const std::vector<unsigned int>& d) const
      -> const Domain<D>* {
      return domainByIndex(offset2index(d));
    }

    auto offset2index(const std::vector<unsigned int>& d) const -> int {
      return m_domain_indices.at(d);
    }

    auto index2offset(const int& index) const -> std::vector<unsigned int> {
      return m_domain_offsets[index];
    }

    auto localDomain() const -> const Domain<D>* {
      // !MULTIDOMAIN: this has to be more general
#if defined(MPI_ENABLED)
      return domainByIndex(m_mpirank);
#else  // not MPI_ENABLED
      return domainByIndex(0);
#endif // MPI_ENABLED
    }

    /**
     * Getters
     */

    [[nodiscard]]
    auto globalNcells() const -> const std::vector<unsigned int>& {
      return m_global_ncells;
    }

    [[nodiscard]]
    auto globalExtent() const -> const std::vector<real_t>& {
      return m_global_extent;
    }

    [[nodiscard]]
    auto globalNdomains() const -> unsigned int {
      return m_global_ndomains;
    }

    [[nodiscard]]
    auto globalNdomainsPerDim() const -> const std::vector<unsigned int>& {
      return m_global_ndomains_per_dim;
    }

    [[nodiscard]]
    auto domainOffsets() const -> const std::vector<std::vector<unsigned int>>& {
      return m_domain_offsets;
    }

    [[nodiscard]]
    auto domainIndices() const
      -> const std::map<std::vector<unsigned int>, unsigned int>& {
      return m_domain_indices;
    }

    [[nodiscard]]
    auto globalBoundaries() const
      -> const std::vector<std::vector<BoundaryCondition>>& {
      return m_global_boundaries;
    }

    [[nodiscard]]
    auto globalMetric() const -> const Metric<D>& {
      return m_global_metric;
    }

    [[nodiscard]]
    auto smallestCellSize() const -> real_t {
      return m_smallest_cell_size;
    }

    [[nodiscard]]
    auto fiducialCellVolume() const -> real_t {
      return m_fiducial_cell_volume;
    }

#if defined(MPI_ENABLED)
    [[nodiscard]]
    auto mpiSize() const -> int {
      return m_mpisize;
    }

    [[nodiscard]]
    auto mpiRank() const -> int {
      return m_mpirank;
    }
#endif // MPI_ENABLED
  };

} // namespace ntt

#endif // FRAMEWORK_COMM_METADOMAIN_H