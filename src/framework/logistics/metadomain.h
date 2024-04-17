/**
 * @file framework/logistics/metadomain.h
 * @brief ...
 * @implements
 *   - ntt::Metadomain<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - utils/error.h
 *   - utils/tools.h
 *   - utils/comparators.h
 *   - arch/mpi_aliases.h
 *   - framework/logistics/domain.h
 *   - framework/containers/species.h
 *   - metrics/kerr_schild.h
 *   - metrics/kerr_schild_0.h
 *   - metrics/minkowski.h
 *   - metrics/qkerr_schild.h
 *   - metrics/qspherical.h
 *   - metrics/spherical.h
 * @cpp:
 *   - metadomain.cpp
 * @namespaces:
 *   - ntt::
 */

#ifndef FRAMEWORK_LOGISTICS_METADOMAIN_H
#define FRAMEWORK_LOGISTICS_METADOMAIN_H

#include "enums.h"
#include "global.h"

#include "framework/containers/species.h"
#include "framework/logistics/domain.h"

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
  struct Metadomain {
    static_assert(M::is_metric,
                  "template arg for Metadomain class has to be a metric");
    static constexpr Dimension D { M::Dim };

    void initialValidityCheck() const;
    void finalValidityCheck() const;
    void metricCompatibilityCheck() const;

    /**
     * @brief Populates the g_subdomains vector with ...
     * ... domains of proper shape, extent, index, and offset
     */
    void createEmptyDomains();

    /**
     * @brief Populates the neighbor-pointers of each domain in g_subdomains
     */
    void redefineNeighbors();

    /**
     * @brief Populates the boundary-conditions of each domain in g_subdomains
     */
    void redefineBoundaries();

    /**
     * @param global_ndomains total number of domains
     * @param global_decomposition decomposition of the global domain
     * @param global_ncells number of cells in each dimension
     * @param global_extent physical extent of the global domain
     * @param global_flds_bc boundary conditions for fields
     * @param global_prtl_bc boundary conditions for particles
     * @param metric_params parameters for the metric
     * @param species_params parameters for the particle species
     */
    Metadomain(unsigned int,
               const std::vector<int>&,
               const std::vector<std::size_t>&,
               const boundaries_t<real_t>&,
               const boundaries_t<FldsBC>&,
               const boundaries_t<PrtlBC>&,
               const std::map<std::string, real_t>&,
               const std::vector<ParticleSpecies>&);

    ~Metadomain() = default;

    /* getters -------------------------------------------------------------- */
    [[nodiscard]]
    auto ndomains() const -> unsigned int {
      return g_ndomains;
    }

    [[nodiscard]]
    auto ndomains_per_dim() const -> const std::vector<unsigned int>& {
      return g_ndomains_per_dim;
    }

    [[nodiscard]]
    auto idx2subdomain(unsigned int idx) const -> const Domain<S, M>& {
      return g_subdomains.at(idx);
    }

    [[nodiscard]]
    auto species_params() const -> const std::vector<ParticleSpecies>& {
      return g_species_params;
    }

  private:
    // domain information
    unsigned int g_ndomains;

    std::vector<int>                                  g_decomposition;
    std::vector<unsigned int>                         g_ndomains_per_dim;
    std::vector<std::vector<unsigned int>>            g_domain_offsets;
    std::map<std::vector<unsigned int>, unsigned int> g_domain_offset2index;

    std::vector<Domain<S, M>>                  g_subdomains;
    std::vector<std::shared_ptr<Domain<S, M>>> g_local_subdomains;

    // grid information
    std::vector<std::size_t> g_ncells;

    // physical domain information
    boundaries_t<real_t> g_extent;
    boundaries_t<FldsBC> g_flds_bc;
    boundaries_t<PrtlBC> g_prtl_bc;

    M                                   g_metric;
    const std::map<std::string, real_t> g_metric_params;
    const std::vector<ParticleSpecies>  g_species_params;

#if defined(MPI_ENABLED)
    int g_mpi_rank, g_mpi_size;
#endif
  };

} // namespace ntt

#endif // FRAMEWORK_LOGISTICS_METADOMAIN_H