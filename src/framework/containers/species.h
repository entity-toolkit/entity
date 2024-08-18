/**
 * @file framework/containers/species.h
 * @brief Container for particle species information
 * @implements
 *   - ntt::ParticleSpecies
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
    // Species index
    const unsigned short m_index;
    // Species label
    const std::string    m_label;
    // Species mass in units of m0
    const float          m_mass;
    // Species charge in units of q0
    const float          m_charge;
    // Max number of allocated particles for the species
    std::size_t          m_maxnpart;

    // Pusher assigned for the species
    const PrtlPusher m_pusher;

    // Use byrid gca pusher for the species
    const bool m_use_gca;

    // Cooling drag mechanism assigned for the species
    const Cooling m_cooling;

    // Number of payloads for the species
    const unsigned short m_npld;

  public:
    ParticleSpecies()
      : m_index { 0 }
      , m_label { "" }
      , m_mass { 0.0 }
      , m_charge { 0.0 }
      , m_maxnpart { 0 }
      , m_pusher { PrtlPusher::INVALID }
      , m_use_gca { false }
      , m_cooling { Cooling::INVALID }
      , m_npld { 0 } {}

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
    ParticleSpecies(unsigned short     index,
                    const std::string& label,
                    float              m,
                    float              ch,
                    std::size_t        maxnpart,
                    const PrtlPusher&  pusher,
                    bool               use_gca,
                    const Cooling&     cooling,
                    unsigned short     npld = 0)
      : m_index { index }
      , m_label { std::move(label) }
      , m_mass { m }
      , m_charge { ch }
      , m_maxnpart { maxnpart }
      , m_pusher { pusher }
      , m_use_gca { use_gca }
      , m_cooling { cooling }
      , m_npld { npld } {}

    ParticleSpecies(const ParticleSpecies&) = default;

    /**
     * @brief Destructor for the particle species container.
     */
    ~ParticleSpecies() = default;

    [[nodiscard]]
    auto index() const -> unsigned short {
      return m_index;
    }

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
    auto use_gca() const -> bool {
      return m_use_gca;
    }

    [[nodiscard]]
    auto cooling() const -> Cooling {
      return m_cooling;
    }

    [[nodiscard]]
    auto npld() const -> unsigned short {
      return m_npld;
    }
  };
} // namespace ntt

#endif // FRAMEWORK_CONTAINERS_SPECIES_H
