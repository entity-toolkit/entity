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
    const int   m_index;
    // Species label.
    std::string m_label;
    // Species mass in units of m0.
    float       m_mass;
    // Species charge in units of q0.
    float       m_charge;
    // Max number of allocated particles for the species.
    std::size_t m_maxnpart;

    unsigned short m_npld;

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
                    const ParticlePusher& pusher,
                    const unsigned short& npld = 0);

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
    ~ParticleSpecies() = default;

    [[nodiscard]]
    auto label() const -> std::string {
      return m_label;
    }

    [[nodiscard]]
    auto mass() const -> float {
      return m_mass;
    }

    [[nodiscard]]
    auto charge() const -> float {
      return m_charge;
    }

    [[nodiscard]]
    auto maxnpart() const -> std::size_t {
      return m_maxnpart;
    }

    [[nodiscard]]
    auto pusher() const -> ParticlePusher {
      return m_pusher;
    }

    [[nodiscard]]
    auto index() const -> int {
      return m_index;
    }

    [[nodiscard]]
    auto npld() const -> unsigned short {
      return m_npld;
    }
  };
} // namespace ntt

#endif // FRAMEWORK_SPECIES_H