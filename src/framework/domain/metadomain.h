/**
 * @file framework/domain/metadomain.h
 * @brief ...
 * @implements
 *   - ntt::Metadomain<>
 * @cpp:
 *   - metadomain.cpp
 * @namespaces:
 *   - ntt::
 * @macros:
 *   - MPI_ENABLED
 *   - OUTPUT_ENABLED
 */

#ifndef FRAMEWORK_DOMAIN_METADOMAIN_H
#define FRAMEWORK_DOMAIN_METADOMAIN_H

#include "enums.h"
#include "global.h"

#include "utils/timer.h"

#include "framework/containers/species.h"
#include "framework/domain/domain.h"
#include "framework/domain/mesh.h"
#include "framework/parameters.h"

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

#if defined OUTPUT_ENABLED
  #include "output/writer.h"
#endif

#include <map>
#include <string>
#include <utility>
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

    template <typename Func, typename... Args>
    void runOnLocalDomains(Func func, Args&&... args) {
      for (auto& ldidx : g_local_subdomain_indices) {
        func(g_subdomains[ldidx], std::forward<Args>(args)...);
      }
    }

    template <typename Func, typename... Args>
    void runOnLocalDomainsConst(Func func, Args&&... args) const {
      for (auto& ldidx : g_local_subdomain_indices) {
        func(g_subdomains[ldidx], std::forward<Args>(args)...);
      }
    }

    void CommunicateFields(Domain<S, M>&, CommTags);
    void SynchronizeFields(Domain<S, M>&, CommTags, const range_tuple_t& = { 0, 0 });
    void CommunicateParticles(Domain<S, M>&, timer::Timers*);

    /**
     * @param global_ndomains total number of domains
     * @param global_decomposition decomposition of the global domain
     * @param global_ncells number of cells in each dimension
     * @param global_extent physical extent of the global domain
     * @param global_flds_bc boundary conditions for fields
     * @param global_prtl_bc boundary conditions for particles
     * @param metric_params parameters for the metric
     * @param species_params parameters for the particle species
     * @param output_params parameters for the output
     */
    Metadomain(unsigned int,
               const std::vector<int>&,
               const std::vector<std::size_t>&,
               const boundaries_t<real_t>&,
               const boundaries_t<FldsBC>&,
               const boundaries_t<PrtlBC>&,
               const std::map<std::string, real_t>&,
               const std::vector<ParticleSpecies>&
#if defined(OUTPUT_ENABLED)
               ,
               const std::string&
#endif
    );

#if defined(OUTPUT_ENABLED)
    void InitWriter(const SimulationParams&);
    auto Write(const SimulationParams&, std::size_t, long double) -> bool;
#endif

    Metadomain(const Metadomain&)            = delete;
    Metadomain& operator=(const Metadomain&) = delete;

    ~Metadomain() = default;

    /* setters -------------------------------------------------------------- */

    /* getters -------------------------------------------------------------- */
    [[nodiscard]]
    auto ndomains() const -> unsigned int {
      return g_ndomains;
    }

    [[nodiscard]]
    auto ndomains_per_dim() const -> std::vector<unsigned int> {
      return g_ndomains_per_dim;
    }

    [[nodiscard]]
    auto subdomain(unsigned int idx) const -> const Domain<S, M>& {
      raise::ErrorIf(idx >= g_subdomains.size(), "subdomain() failed", HERE);
      return g_subdomains[idx];
    }

    [[nodiscard]]
    auto subdomain_ptr(unsigned int idx) -> Domain<S, M>* {
      raise::ErrorIf(idx >= g_subdomains.size(), "subdomain_ptr() failed", HERE);
      return &g_subdomains[idx];
    }

    [[nodiscard]]
    auto mesh() const -> const Mesh<M>& {
      return g_mesh;
    }

    [[nodiscard]]
    auto species_params() const -> const std::vector<ParticleSpecies>& {
      return g_species_params;
    }

    [[nodiscard]]
    auto local_subdomain_indices() const -> std::vector<unsigned int> {
      return g_local_subdomain_indices;
    }

  private:
    // domain information
    unsigned int g_ndomains;

    std::vector<int>                                  g_decomposition;
    std::vector<unsigned int>                         g_ndomains_per_dim;
    std::vector<std::vector<unsigned int>>            g_domain_offsets;
    std::map<std::vector<unsigned int>, unsigned int> g_domain_offset2index;

    std::vector<Domain<S, M>> g_subdomains;
    std::vector<unsigned int> g_local_subdomain_indices;

    Mesh<M>                             g_mesh;
    const std::map<std::string, real_t> g_metric_params;
    const std::vector<ParticleSpecies>  g_species_params;

#if defined(OUTPUT_ENABLED)
    out::Writer g_writer;
#endif

#if defined(MPI_ENABLED)
    int g_mpi_rank, g_mpi_size;
#endif
  };

} // namespace ntt

#endif // FRAMEWORK_DOMAIN_METADOMAIN_H
