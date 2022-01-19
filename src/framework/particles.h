#ifndef FRAMEWORK_PARTICLES_H
#define FRAMEWORK_PARTICLES_H

#include "global.h"

#include <cstddef>
#include <string>

namespace ntt {
  /**
   * Container for the information about the particle species.
   */
  class ParticleSpecies {
  protected:
    // Species label.
    std::string m_label;
    // Species mass in units of m0.
    float m_mass;
    // Species charge in units of q0.
    float m_charge;
    // Max number of allocated particles for the species.
    std::size_t m_maxnpart;
    // Pusher assigned for the species.
    ParticlePusher m_pusher;

  public:
    /**
     * Constructor for the particle species container.
     *
     * @param label species label.
     * @param m species mass.
     * @param ch species charge.
     * @param maxnpart max number of allocated particles.
     * @param pusher pusher assigned for the species.
     */
    ParticleSpecies(
        std::string label,
        const float& m,
        const float& ch,
        const std::size_t& maxnpart,
        const ParticlePusher& pusher);
    /**
     * Constructor for the particle species container which deduces the pusher itself.
     *
     * @overload
     * @param label species label.
     * @param m species mass.
     * @param ch species charge.
     * @param maxnpart max number of allocated particles.
     */
    ParticleSpecies(std::string, const float&, const float&, const std::size_t&);
    /**
     * Copy constructor for the particle species container.
     *
     * @overload
     * @param spec 
     */
    ParticleSpecies(const ParticleSpecies&) = default;
    ~ParticleSpecies() = default;

    /**
     * Getters
     */
    [[nodiscard]] auto label() const -> std::string { return m_label; }
    [[nodiscard]] auto mass() const -> float { return m_mass; }
    [[nodiscard]] auto charge() const -> float { return m_charge; }
    [[nodiscard]] auto maxnpart() const -> std::size_t { return m_maxnpart; }
    [[nodiscard]] auto pusher() const -> ParticlePusher { return m_pusher; }
  };

  /**
   * Container class to carry particle information for a specific species.
   *
   * @tparam D Dimension.
   * @tparam S Simulation type.
   */
  template <Dimension D, SimulationType S>
  struct Particles : public ParticleSpecies {
  private:
    // Number of currently active (used) particles.
    std::size_t m_npart {0};
  public:
    /**
     * 1D arrays with particle data
     * 
     * @todo: Perhaps try VPIC-style arrays.
     */
    // Cell number of the current particle.
    NTTArray<int*> i1, i2, i3;
    // Displacement of a particle within the shell.
    NTTArray<float*> dx1, dx2, dx3;
    // Three spatial components of the covariant 4-velocity (physical units).
    NTTArray<real_t*> ux1, ux2, ux3;
    // Particle weights.
    NTTArray<float*> weight;

    /**
     * Constructor for the particle container.
     *
     * @param label species label.
     * @param m species mass.
     * @param ch species charge.
     * @param maxnpart max number of allocated particles.
     */
    Particles(const std::string&, const float&, const float&, const std::size_t&);
    /**
     * Constructor for the particle container.
     *
     * @overload
     * @param spec species container.
     */
    Particles(const ParticleSpecies&);
    ~Particles() = default;

    /**
     * Loop over all active particles.
     *
     * @returns 1D Kokkos range policy of size of `npart`.
     */
    auto loopParticles() -> RangeND<Dimension::ONE_D>;
    /**
     * Getters
     */
    [[nodiscard]] auto npart() const -> const std::size_t& { return m_npart; }
  };

} // namespace ntt

#endif
