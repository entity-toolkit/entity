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
    const spidx_t     m_index;
    // Species label
    const std::string m_label;
    // Species mass in units of m0
    const float       m_mass;
    // Species charge in units of q0
    const float       m_charge;
    // Max number of allocated particles for the species
    npart_t           m_maxnpart;

    // Pusher assigned for the species
    const PrtlPusher m_pusher;

    // Use particle tracking for the species
    const bool m_use_tracking;

    // Use byrid gca pusher for the species
    const bool m_use_gca;

    // Cooling drag mechanism assigned for the species
    const Cooling m_cooling;

    // Number of payloads for the species
    const unsigned short m_npld_r;
    const unsigned short m_npld_i;

  public:
    ParticleSpecies()
      : m_index { 0u }
      , m_label { "" }
      , m_mass { 0.0 }
      , m_charge { 0.0 }
      , m_maxnpart { 0 }
      , m_pusher { PrtlPusher::INVALID }
      , m_use_tracking { false }
      , m_use_gca { false }
      , m_cooling { Cooling::INVALID }
      , m_npld_r { 0 }
      , m_npld_i { 0 } {}

    /**
     * @brief Constructor for the particle species container.
     *
     * @param index The index of the species in the meshblock::particles vector (index + 1).
     * @param label The label for the species.
     * @param m The mass of the species.
     * @param ch The charge of the species.
     * @param maxnpart The maximum number of allocated particles for the species.
     * @param pusher The pusher assigned for the species.
     * @param use_tracking Use particle tracking for the species.
     * @param use_gca Use hybrid GCA pusher for the species.
     * @param cooling The cooling mechanism assigned for the species.
     * @param npld_r The number of real-valued payloads for the species
     * @param npld_i The number of integer-valued payloads for the species
     */
    ParticleSpecies(spidx_t            index,
                    const std::string& label,
                    float              m,
                    float              ch,
                    npart_t            maxnpart,
                    const PrtlPusher&  pusher,
                    bool               use_tracking,
                    bool               use_gca,
                    const Cooling&     cooling,
                    unsigned short     npld_r = 0,
                    unsigned short     npld_i = 0)
      : m_index { index }
      , m_label { std::move(label) }
      , m_mass { m }
      , m_charge { ch }
      , m_maxnpart { maxnpart }
      , m_pusher { pusher }
      , m_use_tracking { use_tracking }
      , m_use_gca { use_gca }
      , m_cooling { cooling }
      , m_npld_r { npld_r }
      , m_npld_i { npld_i } {
      if (use_tracking) {
#if !defined(MPI_ENABLED)
        raise::ErrorIf(m_npld_i < 1,
                       "npld_i must be at least 1 when tracking is enabled",
                       HERE);
#else
        raise::ErrorIf(
          m_npld_i < 2,
          "npld_i must be at least 2 when tracking is enabled with MPI",
          HERE);
#endif
      }
    }

    ParticleSpecies(const ParticleSpecies&) = default;

    /**
     * @brief Destructor for the particle species container.
     */
    ~ParticleSpecies() = default;

    [[nodiscard]]
    auto index() const -> spidx_t {
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
    auto maxnpart() const -> npart_t {
      return m_maxnpart;
    }

    [[nodiscard]]
    auto pusher() const -> PrtlPusher {
      return m_pusher;
    }

    [[nodiscard]]
    auto use_tracking() const -> bool {
      return m_use_tracking;
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
    auto npld_r() const -> unsigned short {
      return m_npld_r;
    }

    [[nodiscard]]
    auto npld_i() const -> unsigned short {
      return m_npld_i;
    }
  };
} // namespace ntt

#endif // FRAMEWORK_CONTAINERS_SPECIES_H
