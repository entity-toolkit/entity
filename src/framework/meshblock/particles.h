#ifndef FRAMEWORK_PARTICLES_H
#define FRAMEWORK_PARTICLES_H

#include "wrapper.h"

#include "meshblock/species.h"

#include <cstddef>
#include <string>

namespace ntt {
  enum ParticleTag : short { dead = 0, alive, NTags };

  /**
   * @brief Container class to carry particle information for a specific species.
   * @tparam D The dimension of the simulation.
   * @tparam S The simulation engine being used.
   */
  template <Dimension D, SimulationEngine S>
  struct Particles : public ParticleSpecies {
  private:
    // Number of currently active (used) particles.
    std::size_t m_npart { 0 };

  public:
    /**
     * Arrays containing particle data.
     */
    // Cell indices of the current particle.
    array_t<int*>             i1, i2, i3;
    // Displacement of a particle within the cell.
    array_t<prtldx_t*>        dx1, dx2, dx3;
    // Three spatial components of the covariant 4-velocity (physical units).
    array_t<real_t*>          ux1, ux2, ux3;
    // Particle weights.
    array_t<real_t*>          weight;
    // Additional variables (specific to different cases).
    // previous coordinates (GR specific)
    array_t<real_t*>          i1_prev, i2_prev, i3_prev;
    array_t<prtldx_t*>        dx1_prev, dx2_prev, dx3_prev;
    // phi coordinate (for axisymmetry)
    array_t<real_t*>          phi;
    // Array to tag the particles.
    array_t<short*>           tag;

    // host mirrors
    array_mirror_t<int*>      i1_h, i2_h, i3_h;
    array_mirror_t<prtldx_t*> dx1_h, dx2_h, dx3_h;
    array_mirror_t<real_t*>   ux1_h, ux2_h, ux3_h;
    array_mirror_t<real_t*>   weight_h;
    array_mirror_t<real_t*>   phi_h;
    array_mirror_t<short*>    tag_h;

    /**
     * @brief Constructor for the particle container.
     * @param index The index of the species in the meshblock::particles vector (index + 1).
     * @param label The label for the species.
     * @param m The mass of the species.
     * @param ch The charge of the species.
     * @param maxnpart The maximum number of allocated particles for the species.
     * @param pusher The pusher assigned for the species.
     */
    Particles(const int&            index,
              const std::string&    label,
              const float&          m,
              const float&          ch,
              const std::size_t&    maxnpart,
              const ParticlePusher& pusher);

    /**
     * @brief Constructor for the particle container.
     * @overload
     * @param spec The particle species container.
     */
    Particles(const ParticleSpecies& spec);
    ~Particles() = default;

    /**
     * @brief Loop over all active particles.
     * @returns A 1D Kokkos range policy of size of `npart`.
     */
    auto               rangeActiveParticles() -> range_t<Dim1>;

    /**
     * @brief Loop over all particles.
     * @returns A 1D Kokkos range policy of size of `npart`.
     */
    auto               rangeAllParticles() -> range_t<Dim1>;

    /**
     * @brief Get the number of active particles.
     * @return The number of active particles as a std::size_t.
     */
    [[nodiscard]] auto npart() const -> std::size_t {
      return m_npart;
    }

    /**
     * @brief Set the number of particles.
     * @param npart The number of particles as a std::size_t.
     */
    void setNpart(const std::size_t& npart) {
      NTTHostErrorIf(npart > maxnpart(),
                     fmt::format("Trying to set npart to {} which is larger than maxnpart {}.",
                                 npart,
                                 maxnpart()));
      m_npart = npart;
    }

    /**
     * @brief Count the number of particles with a specific tag.
     * @return The vector of counts for each tag.
     */
    auto CountTaggedParticles() const -> std::vector<std::size_t>;

    /**
     * @brief Reshuffle particles by their tags.
     */
    void ReshuffleByTags();
  };

}    // namespace ntt

#endif
