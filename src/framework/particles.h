#ifndef FRAMEWORK_PARTICLES_H
#define FRAMEWORK_PARTICLES_H

#include "wrapper.h"

#include <cstddef>
#include <string>

namespace ntt {
  /**
   * @brief Container for the information about the particle species.
   */
  class ParticleSpecies {
  protected:
    // Species label.
    std::string    m_label;
    // Species mass in units of m0.
    float          m_mass;
    // Species charge in units of q0.
    float          m_charge;
    // Max number of allocated particles for the species.
    std::size_t    m_maxnpart;
    // Pusher assigned for the species.
    ParticlePusher m_pusher;

  public:
    const int m_index;

    /**
     * @brief Constructor for the particle species container.
     *
     * @param index The index of the species in the meshblock::particles vector (index + 1).
     * @param label The label for the species.
     * @param m The mass of the species.
     * @param ch The charge of the species.
     * @param maxnpart The maximum number of allocated particles for the species.
     * @param pusher The pusher assigned for the species.
     */
    ParticleSpecies(const int&            index,
                    const std::string&    label,
                    const float&          m,
                    const float&          ch,
                    const std::size_t&    maxnpart,
                    const ParticlePusher& pusher);

    /**
     * @brief Constructor for the particle species container which deduces the pusher itself.
     *
     * @overload
     * @param index The index of the species in the meshblock::particles vector (index + 1).
     * @param label The label for the species.
     * @param m The mass of the species.
     * @param ch The charge of the species.
     * @param maxnpart The maximum number of allocated particles for the species.
     */
    ParticleSpecies(
      const int&, const std::string&, const float&, const float&, const std::size_t&);

    /**
     * @brief Copy constructor for the particle species container.
     *
     * @overload
     * @param spec The particle species to copy from.
     */
    ParticleSpecies(const ParticleSpecies&) = default;

    /**
     * @brief Destructor for the particle species container.
     */
    ~ParticleSpecies()                      = default;

    /**
     * @brief Get the species label.
     *
     * @return The species label as a std::string.
     */
    [[nodiscard]] auto label() const -> std::string {
      return m_label;
    }

    /**
     * @brief Get the species mass.
     *
     * @return The species mass as a float.
     */
    [[nodiscard]] auto mass() const -> float {
      return m_mass;
    }

    /**
     * @brief Get the species charge.
     *
     * @return The species charge as a float.
     */
    [[nodiscard]] auto charge() const -> float {
      return m_charge;
    }

    /**
     * @brief Get the max number of allocated particles.
     *
     * @return The maximum number of allocated particles as a std::size_t.
     */
    [[nodiscard]] auto maxnpart() const -> std::size_t {
      return m_maxnpart;
    }

    /**
     * @brief Get the pusher assigned for the species.
     *
     * @return The pusher assigned for the species as a ParticlePusher.
     */
    [[nodiscard]] auto pusher() const -> ParticlePusher {
      return m_pusher;
    }

    /**
     * @brief Get the species index.
     *
     * @return The index of the species in the meshblock::particles vector as an int.
     */
    [[nodiscard]] auto index() const -> int {
      return m_index;
    }
  };
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
    array_t<int*>           i1, i2, i3;
    // Displacement of a particle within the cell.
    array_t<float*>         dx1, dx2, dx3;
    // Three spatial components of the covariant 4-velocity (physical units).
    array_t<real_t*>        ux1, ux2, ux3;
    // Particle weights.
    array_t<float*>         weight;
    // Additional variables (specific to different cases).
    // previous coordinates (GR specific)
    array_t<real_t*>        i1_prev, i2_prev, i3_prev;
    array_t<real_t*>        dx1_prev, dx2_prev, dx3_prev;
    // phi coordinate (for axisymmetry)
    array_t<real_t*>        phi;
    // Array to track whether a particle is dead or not.
    array_t<bool*>          is_dead;

    // Host mirrors for cell indices.
    array_mirror_t<int*>    i1_h, i2_h, i3_h;
    // Host mirrors for displacement.
    array_mirror_t<float*>  dx1_h, dx2_h, dx3_h;
    // Host mirrors for the 4-velocities.
    array_mirror_t<real_t*> ux1_h, ux2_h, ux3_h;
    // Host mirrors for the particle weights.
    array_mirror_t<float*>  weight_h;
    // host mirrors for previous coordinates
    array_mirror_t<real_t*> i1_prev_h, i2_prev_h, i3_prev_h;
    array_mirror_t<real_t*> dx1_prev_h, dx2_prev_h, dx3_prev_h;
    // host mirrors for phi
    array_mirror_t<real_t*> phi_h;
    // host mirrors for is_dead
    array_mirror_t<bool*>   is_dead_h;

    /**
     * @brief Constructor for the particle container.
     * @param index The index of the species in the meshblock::particles vector (index + 1).
     * @param label The label for the species.
     * @param m The mass of the species.
     * @param ch The charge of the species.
     * @param maxnpart The maximum number of allocated particles for the species.
     */
    Particles(const int&         index,
              const std::string& label,
              const float&       m,
              const float&       ch,
              const std::size_t& maxnpart);

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
     * @brief Synchronize data from device to host.
     * Synchronize all the arrays that have host mirrors.
     */
    void SynchronizeHostDevice();

    /**
     * @brief Count the number of living particles
     * @return The number of living particles as a std::size_t.
     */
    auto CountLiving() const -> std::size_t;

    /**
     * @brief Reshuffle dead particles to the end of all arrays.
     */
    void ReshuffleDead();
  };

}    // namespace ntt

#endif
