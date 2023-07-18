#ifndef FRAMEWORK_SPECIES_H
#define FRAMEWORK_SPECIES_H

#include "wrapper.h"

#include <string>

namespace ntt {
  /**
   * @brief Container for the information about the particle species.
   */
  class ParticleSpecies {
  protected:
    // Species index.
    const int      m_index;
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
}    // namespace ntt

#endif    // FRAMEWORK_SPECIES_H