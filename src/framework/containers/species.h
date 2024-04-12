/**
 * @file framework/containers/species.h
 * @brief Container for particle species information
 * @implements
 *   - ntt::ParticleSpecies
 * @depends:
 *   - enums.h
 * @namespaces:
 *   - ntt::
 * @note Particles class inherits from this one
 */

#ifndef FRAMEWORK_CONTAINERS_SPECIES_H
#define FRAMEWORK_CONTAINERS_SPECIES_H

#include "enums.h"

#include <string>

namespace ntt {

  class ParticleSpecies {
  protected:
    // Species index.
    const int         m_index;
    // Species label.
    const std::string m_label;
    // Species mass in units of m0.
    const float       m_mass;
    // Species charge in units of q0.
    const float       m_charge;
    // Max number of allocated particles for the species.
    std::size_t       m_maxnpart;

    // Pusher assigned for the species.
    const PrtlPusher m_pusher;

    // Cooling drag mechanism assigned for the species.
    const Cooling m_cooling;

    const unsigned short m_npld;

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
                    const PrtlPusher&     pusher,
                    const Cooling&        cooling,
                    const unsigned short& npld = 0) :
      m_index { index },
      m_label { std::move(label) },
      m_mass { m },
      m_charge { ch },
      m_maxnpart { maxnpart },
      m_pusher { pusher },
      m_cooling { cooling },
      m_npld { npld } {}

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
    auto pusher() const -> PrtlPusher {
      return m_pusher;
    }

    [[nodiscard]]
    auto cooling() const -> Cooling {
      return m_cooling;
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

#endif // FRAMEWORK_CONTAINERS_SPECIES_H