/**
 * @file framework/containers/particles.h
 * @brief Definition of the particle container class
 * @implements
 *   - ntt::Particles<> : ntt::ParticleSpecies
 * @cpp:
 *   - particles.cpp
 * @macros:
 *   - MPI_ENABLED
 */

#ifndef FRAMEWORK_CONTAINERS_PARTICLES_H
#define FRAMEWORK_CONTAINERS_PARTICLES_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"

#include "framework/containers/species.h"

#include <Kokkos_Core.hpp>

#include <string>
#include <vector>

namespace ntt {

  /**
   * @brief Container class to carry particle information for a specific species
   * @tparam D The dimension of the simulation
   * @tparam S The simulation engine being used
   */
  template <Dimension D, Coord::type C>
  struct Particles : public ParticleSpecies {
  private:
    // Number of currently active (used) particles
    std::size_t m_npart { 0 };
    bool        m_is_sorted { false };

#if !defined(MPI_ENABLED)
    const std::size_t m_ntags { 2 };
#else // MPI_ENABLED
    const std::size_t m_ntags { (std::size_t)(2 + math::pow(3, (int)D) - 1) };
#endif

  public:
    // Cell indices of the current particle
    array_t<int*>                 i1, i2, i3;
    // Displacement of a particle within the cell
    array_t<prtldx_t*>            dx1, dx2, dx3;
    // Three spatial components of the covariant 4-velocity (physical units)
    array_t<real_t*>              ux1, ux2, ux3;
    // Particle weights.
    array_t<real_t*>              weight;
    // Previous timestep coordinates
    array_t<int*>                 i1_prev, i2_prev, i3_prev;
    array_t<prtldx_t*>            dx1_prev, dx2_prev, dx3_prev;
    // Array to tag the particles
    array_t<short*>               tag;
    // Array to store the particle load
    std::vector<array_t<real_t*>> pld;
    // phi coordinate (for axisymmetry)
    array_t<real_t*>              phi;

    // host mirrors
    array_mirror_t<int*>                 i1_h, i2_h, i3_h;
    array_mirror_t<prtldx_t*>            dx1_h, dx2_h, dx3_h;
    array_mirror_t<real_t*>              ux1_h, ux2_h, ux3_h;
    array_mirror_t<real_t*>              weight_h;
    array_mirror_t<real_t*>              phi_h;
    array_mirror_t<short*>               tag_h;
    std::vector<array_mirror_t<real_t*>> pld_h;

    // for empty allocation
    Particles() {}

    /**
     * @brief Constructor for the particle container
     * @param index The index of the species (starts from 1)
     * @param label The label for the species
     * @param m The mass of the species
     * @param ch The charge of the species
     * @param maxnpart The maximum number of allocated particles for the species
     * @param pusher The pusher assigned for the species
     * @param use_gca Use hybrid GCA pusher for the species
     * @param cooling The cooling mechanism assigned for the species
     * @param npld The number of payloads for the species
     */
    Particles(unsigned short     index,
              const std::string& label,
              float              m,
              float              ch,
              std::size_t        maxnpart,
              const PrtlPusher&  pusher,
              bool               use_gca,
              const Cooling&     cooling,
              unsigned short     npld = 0);

    /**
     * @brief Constructor for the particle container
     * @overload
     * @param spec The particle species container
     */
    Particles(const ParticleSpecies& spec)
      : Particles(spec.index(),
                  spec.label(),
                  spec.mass(),
                  spec.charge(),
                  spec.maxnpart(),
                  spec.pusher(),
                  spec.use_gca(),
                  spec.cooling(),
                  spec.npld()) {}

    Particles(const Particles&)            = delete;
    Particles& operator=(const Particles&) = delete;

    ~Particles() = default;

    /**
     * @brief Loop over all active particles
     * @returns A 1D Kokkos range policy of size of `npart`
     */
    inline auto rangeActiveParticles() const -> range_t<Dim::_1D> {
      return CreateRangePolicy<Dim::_1D>({ 0 }, { npart() });
    }

    /**
     * @brief Loop over all particles
     * @returns A 1D Kokkos range policy of size of `npart`
     */
    inline auto rangeAllParticles() const -> range_t<Dim::_1D> {
      return CreateRangePolicy<Dim::_1D>({ 0 }, { maxnpart() });
    }

    /* getters -------------------------------------------------------------- */
    /**
     * @brief Get the number of active particles
     */
    [[nodiscard]]
    auto npart() const -> std::size_t {
      return m_npart;
    }

    [[nodiscard]]
    auto is_sorted() const -> bool {
      return m_is_sorted;
    }

    /**
     * @brief Get the number of distinct tags possible
     */
    [[nodiscard]]
    auto ntags() const -> std::size_t {
      return m_ntags;
    }

    [[nodiscard]]
    auto memory_footprint() const -> std::size_t {
      std::size_t footprint  = 0;
      footprint             += sizeof(int) * i1.extent(0);
      footprint             += sizeof(int) * i2.extent(0);
      footprint             += sizeof(int) * i3.extent(0);
      footprint             += sizeof(prtldx_t) * dx1.extent(0);
      footprint             += sizeof(prtldx_t) * dx2.extent(0);
      footprint             += sizeof(prtldx_t) * dx3.extent(0);
      footprint             += sizeof(real_t) * ux1.extent(0);
      footprint             += sizeof(real_t) * ux2.extent(0);
      footprint             += sizeof(real_t) * ux3.extent(0);
      footprint             += sizeof(real_t) * weight.extent(0);
      footprint             += sizeof(int) * i1_prev.extent(0);
      footprint             += sizeof(int) * i2_prev.extent(0);
      footprint             += sizeof(int) * i3_prev.extent(0);
      footprint             += sizeof(prtldx_t) * dx1_prev.extent(0);
      footprint             += sizeof(prtldx_t) * dx2_prev.extent(0);
      footprint             += sizeof(prtldx_t) * dx3_prev.extent(0);
      footprint             += sizeof(short) * tag.extent(0);
      for (auto& p : pld) {
        footprint += sizeof(real_t) * p.extent(0);
      }
      footprint += sizeof(real_t) * phi.extent(0);
      return footprint;
    }

    /**
     * @brief Count the number of particles with a specific tag.
     * @return The vector of counts for each tag.
     */
    [[nodiscard]]
    auto npart_per_tag() const -> std::vector<std::size_t>;

    /* setters -------------------------------------------------------------- */
    /**
     * @brief Set the number of particles
     * @param npart The number of particles as a std::size_t
     */
    void set_npart(std::size_t n) {
      raise::ErrorIf(
        n > maxnpart(),
        fmt::format(
          "Trying to set npart to %d which is larger than maxnpart %d",
          n,
          maxnpart()),
        HERE);
      m_npart = n;
    }

    void set_unsorted() {
      m_is_sorted = false;
    }

    /**
     * @brief Sort particles by their tags.
     * @return The vector of counts per each tag.
     */
    auto SortByTags() -> std::vector<std::size_t>;

    /**
     * @brief Copy particle data from device to host.
     */
    void SyncHostDevice();
  };

} // namespace ntt

#endif // FRAMEWORK_CONTAINERS_PARTICLES_H
