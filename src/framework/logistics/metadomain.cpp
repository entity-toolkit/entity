#include "framework/logistics/metadomain.h"

#include "enums.h"
#include "global.h"

#include "arch/mpi_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/tools.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <map>
#include <string>
#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
  Metadomain<S, M>::Metadomain(unsigned int            global_ndomains,
                               const std::vector<int>& global_decomposition,
                               const std::vector<std::size_t>& global_ncells,
                               const boundaries_t<real_t>&     global_extent,
                               const boundaries_t<FldsBC>&     global_flds_bc,
                               const boundaries_t<PrtlBC>&     global_prtl_bc,
                               const std::map<std::string, real_t>& metric_params,
                               const std::vector<ParticleSpecies>& species_params) :
    g_ndomains { global_ndomains },
    g_decomposition { global_decomposition },
    g_ncells { global_ncells },
    g_extent { global_extent },
    g_flds_bc { global_flds_bc },
    g_prtl_bc { global_prtl_bc },
    g_metric { g_ncells, g_extent, metric_params },
    g_metric_params { metric_params },
    g_species_params { species_params } {

    initialValidityCheck();

    createEmptyDomains();
    redefineNeighbors();
    redefineBoundaries();

    finalValidityCheck();
    // metricCompatibilityCheck();
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::initialValidityCheck() const {
    // ensure everything has the correct shape
    raise::ErrorIf(g_decomposition.size() != (std::size_t)D,
                   "Invalid number of dimensions in g_decomposition",
                   HERE);
    raise::ErrorIf(g_ncells.size() != (std::size_t)D,
                   "Invalid number of dimensions in g_ncells",
                   HERE);
    raise::ErrorIf(g_extent.size() != (std::size_t)D,
                   "Invalid number of dimensions in g_extent",
                   HERE);
    raise::ErrorIf(g_flds_bc.size() != (std::size_t)D,
                   "Invalid number of dimensions in g_flds_bc",
                   HERE);
    raise::ErrorIf(g_prtl_bc.size() != (std::size_t)D,
                   "Invalid number of dimensions in g_prtl_bc",
                   HERE);
#if defined(MPI_ENABLED)
    int init_flag;
    int status = MPI_Initialized(init_flag);
    raise::ErrorIf((status != MPI_SUCCESS) || (init_flag != 1),
                   "MPI not initialized",
                   HERE);
    int mpisize, mpirank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    raise::ErrorIf(mpisize != g_ndomains,
                   "ndomains != mpisize is not implemented with MPI",
                   HERE);

#else // not MPI_ENABLED
  #if !defined(DEBUG)
    raise::ErrorIf(g_ndomains != 1,
                   "ndomains > 1 is not implemented for non-DEBUG purposes",
                   HERE);
  #endif
#endif // MPI_ENABLED
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::createEmptyDomains() {
    /* decompose and compute cell & domain offsets ------------------------ */
    auto d_ncells = tools::Decompose(g_ndomains, g_ncells, g_decomposition);
    raise::ErrorIf(d_ncells.size() != (std::size_t)D,
                   "Invalid number of dimensions received",
                   HERE);
    auto d_offset_ncells = std::vector<std::vector<std::size_t>> {};
    auto d_offset_ndoms  = std::vector<std::vector<unsigned int>> {};
    for (auto& d : d_ncells) {
      g_ndomains_per_dim.push_back(d.size());
      auto offset_ncell = std::vector<std::size_t> { 0 };
      auto offset_ndom  = std::vector<unsigned int> { 0 };
      for (std::size_t i { 1 }; i < d.size(); ++i) {
        auto di = d[i - 1];
        offset_ncell.push_back(offset_ncell.back() + di);
        offset_ndom.push_back(offset_ndom.back() + 1);
      }
      d_offset_ncells.push_back(offset_ncell);
      d_offset_ndoms.push_back(offset_ndom);
    }

    /* compute tensor products of the domain decompositions --------------- */
    // works similar to np.meshgrid()
    const auto domain_ncells = tools::TensorProduct<std::size_t>(d_ncells);
    const auto domain_offset_ncells = tools::TensorProduct<std::size_t>(
      d_offset_ncells);
    const auto domain_offset_ndoms = tools::TensorProduct<unsigned int>(
      d_offset_ndoms);

    g_domain_offsets = domain_offset_ndoms;

    /* create the domains ------------------------------------------------- */
    if (not g_subdomains.empty()) {
      g_subdomains.clear();
    }
    for (std::size_t idx { 0 }; idx < g_ndomains; ++idx) {
      auto                 l_offset_ndomains = domain_offset_ndoms[idx];
      auto                 l_ncells          = domain_ncells[idx];
      auto                 l_offset_ncells   = domain_offset_ncells[idx];
      boundaries_t<real_t> l_extent;
      coord_t<D>           low_corner_Code { ZERO }, up_corner_Code { ZERO };
      coord_t<D>           low_corner_Phys { ZERO }, up_corner_Phys { ZERO };
      for (unsigned short d { 0 }; d < (unsigned short)D; ++d) {
        low_corner_Code[d] = (real_t)l_offset_ncells[d];
        up_corner_Code[d]  = (real_t)(l_offset_ncells[d] + l_ncells[d]);
      }
      g_metric.template convert<Crd::Cd, Crd::Ph>(low_corner_Code, low_corner_Phys);
      g_metric.template convert<Crd::Cd, Crd::Ph>(up_corner_Code, up_corner_Phys);
      for (auto d { 0 }; d < (short)D; ++d) {
        l_extent.push_back({ low_corner_Phys[d], up_corner_Phys[d] });
      }
      g_subdomains.emplace_back(idx,
                                l_offset_ndomains,
                                l_offset_ncells,
                                l_ncells,
                                l_extent,
                                g_metric_params,
                                g_species_params);
      g_domain_offset2index[l_offset_ndomains] = idx;
    }
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::redefineNeighbors() {
    for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
      // offset of the subdomain[idx]
      auto       current_domain = &g_subdomains[idx];
      const auto current_offset = current_domain->offset_ndomains();
      for (const auto& direction : dir::Directions<D>::all) {
        // find the neighbor by its offset in specific direction
        auto nghbr_offset = current_offset;
        for (auto d { 0 }; d < (short)D; ++d) { // loop over all dimensions
          auto dir = direction[d];
          // find the neighbor offset in the direction dir, dimension d
          if ((dir == -1) && (current_offset[d] == 0)) {
            // left edge
            // rollover from left wall
            nghbr_offset[d] = g_ndomains_per_dim[d] - 1;
          } else if ((dir == 1) &&
                     (current_offset[d] == g_ndomains_per_dim[d] - 1)) {
            // rollover from right wall
            nghbr_offset[d] = 0;
          } else {
            // otherwise just shift the offset in the right direction
            // to get the neighbor
            nghbr_offset[d] += dir;
          }
        }
        // pointer to the neighbor candidate
        auto nghbr_candidate = &g_subdomains[g_domain_offset2index.at(nghbr_offset)];
        raise::ErrorIf(nghbr_candidate == nullptr,
                       "Neighbor candidate is nullptr",
                       HERE);
        current_domain->setNeighbor(direction, nghbr_candidate);
      }
    }
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::redefineBoundaries() {
    for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
      // offset of the subdomain[idx]
      auto       current_domain = &g_subdomains[idx];
      const auto current_offset = current_domain->offset_ndomains();
      for (auto direction : dir::Directions<D>::orth) {
        FldsBC flds_bc { FldsBC::INVALID };
        PrtlBC prtl_bc { PrtlBC::INVALID };
        for (auto d { 0 }; d < (short)D; ++d) {
          // !TODO: not setting CommBC for now
          auto dir = direction[d];
          if (dir == -1) {
            if (current_offset[d] == 0) {
              // left edge
              flds_bc = g_flds_bc[d].first;
              prtl_bc = g_prtl_bc[d].first;
            } else {
              // not left edge
              flds_bc = FldsBC::SYNC;
              prtl_bc = PrtlBC::SYNC;
            }
            break;
          } else if (dir == 1) {
            if (current_offset[d] == g_ndomains_per_dim[d] - 1) {
              // right edge
              flds_bc = g_flds_bc[d].second;
              prtl_bc = g_prtl_bc[d].second;
            } else {
              // not right edge
              flds_bc = FldsBC::SYNC;
              prtl_bc = PrtlBC::SYNC;
            }
            break;
          }
        }
        // if sending to a different domain & periodic, then sync
        if (current_domain->neighbor_in(direction)->index() != idx) {
          if (flds_bc == FldsBC::PERIODIC) {
            flds_bc = FldsBC::SYNC;
          }
          if (prtl_bc == PrtlBC::PERIODIC) {
            prtl_bc = PrtlBC::SYNC;
          }
        }
        current_domain->mesh.setFldsBc(direction, flds_bc);
        current_domain->mesh.setPrtlBc(direction, prtl_bc);
      }
    }
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::finalValidityCheck() const {
    for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
      const auto current_domain = &g_subdomains[idx];
      // check that all neighbors are set properly
      for (const auto& direction : dir::Directions<D>::all) {
        raise::ErrorIf(current_domain->neighbor_in(direction) == nullptr,
                       "Neighbor not set properly",
                       HERE);
        raise::ErrorIf(current_domain->neighbor_in(direction)->neighbor_in(
                         -direction) != current_domain,
                       "Neighbor not set properly",
                       HERE);
      }
      // check that all boundaries are set properly
      for (const auto& direction : dir::Directions<D>::orth) {
        raise::ErrorIf(current_domain->mesh.flds_bc_in(direction) == FldsBC::INVALID,
                       "Invalid boundary condition for fields",
                       HERE);
        raise::ErrorIf(current_domain->mesh.prtl_bc_in(direction) == PrtlBC::INVALID,
                       "Invalid boundary condition for particles",
                       HERE);
      }
    }
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::metricCompatibilityCheck() const {
    const auto dx_min = g_metric.dxMin();
    for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
      const auto current_domain = &g_subdomains[idx];
      const auto current_dx_min = current_domain->mesh.metric.dxMin();
      raise::ErrorIf(!cmp::AlmostEqual(dx_min, current_dx_min),
                     "dx_min is not the same across all domains",
                     HERE);
    }
#if defined(MPI_ENABLED)
    auto dx_mins       = std::vector<real_t>(g_ndomains);
    dx_mins[m_mpirank] = dx_min;
    MPI_Allgather(&dx_mins[m_mpirank],
                  1,
                  mpi::get_type<real_t>(),
                  dx_mins.data(),
                  1,
                  mpi::get_type<real_t>(),
                  MPI_COMM_WORLD);
    for (const auto& dx : dx_mins) {
      raise::ErrorIf(!cmp::AlmostEqual(dx, dx_min),
                     "dx_min is not the same across all MPI ranks",
                     HERE);
    }
#endif
  }

  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_3D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Spherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::QSpherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>>;

} // namespace ntt