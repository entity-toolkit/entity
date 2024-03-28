/**
 * @file enums.h
 * @brief Global enum variables describing the simulation
 * @implements
 *   - ntt::em, ntt::cur
 *   - ntt::ParticleTag: dead, alive
 *   - ntt::Coord: Cart, Sph, Qsph
 *   - ntt::Metric: Minkowski, Spherical, QSpherical, Kerr_Schild, QKerr_Schild,
 * Kerr_Schild_0
 *   - ntt::SimulationEngine: SRPIC, GRPIC
 *   - ntt::ParticleBC: periodic, absorb, atmosphere, custom, reflect, horizon, axis, send
 *   - ntt::FieldsBC: periodic, absorb, atmosphere, custom, horizon, axis
 *   - ntt::CommBC: physical, comm
 *   - ntt::ParticlePusher: boris, vay, boris,gca, vay,gca, photon, none
 *   - ntt::Cooling: synchrotron, none
 * @depends:
 *   - utils/error.h
 *   - utils/formatting.h
 *   - utils/log.h
 * @namespaces:
 *   - ntt::
 * @note Enums of the same type can be compared with each other and with strings
 * @note Comparison with strings is case-insensitive
 * @note    example: "srpIC" == SimulationEngine::SRPIC
 * @note To convert an enum to a string, use its stringize() method
 * @note    example: SimulationEngine::SRPIC.stringize() == "srpic"
 * @note To check if a string is a valid option, use the contains() function
 * @note    example: ParticlePusher::contains("Vay") == true
 * @note To get the proper enum instance from a string, use the pick() function
 * @note    example: ParticlePusher::pick("Vay") == ParticlePusher::VAY
 * @note To get the total number of enum instances, use the total variable
 * @note    example: Cooling::total == 2
 * @note To iterate over all enum instances, use the variants array
 * @note    example: for (const auto& cooling : Cooling::variants) { ... }
 */

#ifndef GLOBAL_ENUMS_H
#define GLOBAL_ENUMS_H

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include <algorithm>
#include <string>

#include <string_view>

namespace ntt {

  enum em {
    ex1 = 0,
    ex2 = 1,
    ex3 = 2,
    dx1 = 0,
    dx2 = 1,
    dx3 = 2,
    bx1 = 3,
    bx2 = 4,
    bx3 = 5,
    hx1 = 3,
    hx2 = 4,
    hx3 = 5
  };

  enum cur {
    jx1 = 0,
    jx2 = 1,
    jx3 = 2
  };

  enum ParticleTag : short {
    dead = 0,
    alive
  };

  namespace enums_hidden {
    /**
     * @brief Enum class template
     */
    template <class T>
    struct EnumClass_t {
      explicit EnumClass_t(const std::string& v) : m_variant { v } {}

      auto stringize() const -> std::string {
        return fmt::toLower(m_variant.data());
      }

      auto operator==(const std::string& other) const -> bool {
        return stringize() == fmt::toLower(other);
      }

      auto operator!=(const std::string& other) const -> bool {
        return stringize() != fmt::toLower(other);
      }

      auto operator==(const EnumClass_t& other) const -> bool {
        return stringize() == other.stringize();
      }

      auto operator!=(const EnumClass_t& other) const -> bool {
        return stringize() != other.stringize();
      }

    protected:
      const std::string m_variant;
    };

    template <typename T>
    auto baseContains(T* const           arr,
                      const std::size_t  n,
                      const std::string& elem) -> bool {
      for (std::size_t i = 0; i < n; ++i) {
        if (arr[i] == elem) {
          return true;
        }
      }
      return false;
    }

    template <typename T>
    auto basePick(const T            arr[],
                  const std::size_t  n,
                  const std::string& elem,
                  const T&           invalid) -> T {
      for (std::size_t i = 0; i < n; ++i) {
        if (arr[i] == elem) {
          return arr[i];
        }
      }
      raise::Error("Invalid enum value: " + elem, HERE);
      return invalid;
    }
  } // namespace enums_hidden

  namespace Coord {
    struct Coord_t : enums_hidden::EnumClass_t<Coord_t> {
      using enums_hidden::EnumClass_t<Coord_t>::EnumClass_t;
    };

    using type = Coord_t;
  } // namespace Coord

  namespace Metric {
    struct Metric_t : enums_hidden::EnumClass_t<Metric_t> {
      using enums_hidden::EnumClass_t<Metric_t>::EnumClass_t;
    };

    using type = Metric_t;
  } // namespace Metric

  namespace SimulationEngine {
    struct SimulationEngine_t : enums_hidden::EnumClass_t<SimulationEngine_t> {
      using enums_hidden::EnumClass_t<SimulationEngine_t>::EnumClass_t;
    };

    using type = SimulationEngine_t;
  } // namespace SimulationEngine

  namespace ParticleBC {
    struct ParticleBC_t : enums_hidden::EnumClass_t<ParticleBC_t> {
      using enums_hidden::EnumClass_t<ParticleBC_t>::EnumClass_t;
    };

    using type = ParticleBC_t;
  } // namespace ParticleBC

  namespace FieldsBC {
    struct FieldsBC_t : enums_hidden::EnumClass_t<FieldsBC_t> {
      using enums_hidden::EnumClass_t<FieldsBC_t>::EnumClass_t;
    };

    using type = FieldsBC_t;
  } // namespace FieldsBC

  namespace CommBC {
    struct CommBC_t : enums_hidden::EnumClass_t<CommBC_t> {
      using enums_hidden::EnumClass_t<CommBC_t>::EnumClass_t;
    };

    using type = CommBC_t;
  } // namespace CommBC

  namespace ParticlePusher {
    struct ParticlePusher_t : enums_hidden::EnumClass_t<ParticlePusher_t> {
      using enums_hidden::EnumClass_t<ParticlePusher_t>::EnumClass_t;
    };

    using type = ParticlePusher_t;
  } // namespace ParticlePusher

  namespace Cooling {
    struct Cooling_t : enums_hidden::EnumClass_t<Cooling_t> {
      using enums_hidden::EnumClass_t<Cooling_t>::EnumClass_t;
    };

    using type = Cooling_t;
  } // namespace Cooling

  namespace Coord {
    const type INVALID { "invalid" };
    const type CART { "cart" };
    const type SPH { "sph" };
    const type QSPH { "qsph" };
    const type variants[] = { CART, SPH, QSPH };
  } // namespace Coord

  namespace Metric {
    const type INVALID { "invalid" };
    const type MINKOWSKI { "minkowski" };
    const type SPHERICAL { "spherical" };
    const type QSPHERICAL { "qspherical" };
    const type KERR_SCHILD { "kerr_schild" };
    const type QKERR_SCHILD { "qkerr_schild" };
    const type KERR_SCHILD_0 { "kerr_schild_0" };
    const type variants[] = { MINKOWSKI,   SPHERICAL,    QSPHERICAL,
                              KERR_SCHILD, QKERR_SCHILD, KERR_SCHILD_0 };
  } // namespace Metric

  namespace SimulationEngine {
    const type INVALID { "invalid" };
    const type SRPIC { "SRPIC" };
    const type GRPIC { "GRPIC" };
    const type variants[] = { SRPIC, GRPIC };
  } // namespace SimulationEngine

  namespace ParticleBC {
    const type INVALID { "invalid" };
    const type PERIODIC { "periodic" };
    const type ABSORB { "absorb" };
    const type ATMOSPHERE { "atmosphere" };
    const type CUSTOM { "custom" };
    const type REFLECT { "reflect" };
    const type HORIZON { "horizon" };
    const type AXIS { "axis" };
    const type SEND { "send" };
    const type variants[] = { PERIODIC, ABSORB,  ATMOSPHERE, CUSTOM,
                              REFLECT,  HORIZON, AXIS,       SEND };
  } // namespace ParticleBC

  namespace FieldsBC {
    const type INVALID { "invalid" };
    const type PERIODIC { "periodic" };
    const type ABSORB { "absorb" };
    const type ATMOSPHERE { "atmosphere" };
    const type CUSTOM { "custom" };
    const type HORIZON { "horizon" };
    const type AXIS { "axis" };
    const type variants[] = { PERIODIC, ABSORB,  ATMOSPHERE,
                              CUSTOM,   HORIZON, AXIS };
  } // namespace FieldsBC

  namespace CommBC {
    const type INVALID { "invalid" };
    const type PHYSICAL { "physical" };
    const type COMM { "comm" };
    const type variants[] = { PHYSICAL, COMM };
  } // namespace CommBC

  namespace ParticlePusher {
    const type INVALID { "invalid" };
    const type BORIS { "boris" };
    const type VAY { "vay" };
    const type BORIS_GCA { "boris,gca" };
    const type VAY_GCA { "vay,gca" };
    const type PHOTON { "photon" };
    const type NONE { "none" };
    const type variants[] = { BORIS, VAY, BORIS_GCA, VAY_GCA, PHOTON, NONE };
  } // namespace ParticlePusher

  namespace Cooling {
    const type INVALID { "invalid" };
    const type SYNCHROTRON { "synchrotron" };
    const type NONE { "none" };
    const type variants[] = { SYNCHROTRON, NONE };
  } // namespace Cooling

  namespace Coord {
    const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

    auto pick = [](const std::string& key) -> type {
      return enums_hidden::basePick(variants, total, key, INVALID);
    };

    auto contains = [](const std::string& key) -> bool {
      return enums_hidden::baseContains(variants, total, key);
    };
  } // namespace Coord

  namespace Metric {
    const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

    auto pick = [](const std::string& key) -> type {
      return enums_hidden::basePick(variants, total, key, INVALID);
    };

    auto contains = [](const std::string& key) -> bool {
      return enums_hidden::baseContains(variants, total, key);
    };
  } // namespace Metric

  namespace SimulationEngine {
    const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

    auto pick = [](const std::string& key) -> type {
      return enums_hidden::basePick(variants, total, key, INVALID);
    };

    auto contains = [](const std::string& key) -> bool {
      return enums_hidden::baseContains(variants, total, key);
    };
  } // namespace SimulationEngine

  namespace ParticleBC {
    const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

    auto pick = [](const std::string& key) -> type {
      return enums_hidden::basePick(variants, total, key, INVALID);
    };

    auto contains = [](const std::string& key) -> bool {
      return enums_hidden::baseContains(variants, total, key);
    };
  } // namespace ParticleBC

  namespace FieldsBC {
    const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

    auto pick = [](const std::string& key) -> type {
      return enums_hidden::basePick(variants, total, key, INVALID);
    };

    auto contains = [](const std::string& key) -> bool {
      return enums_hidden::baseContains(variants, total, key);
    };
  } // namespace FieldsBC

  namespace CommBC {
    const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

    auto pick = [](const std::string& key) -> type {
      return enums_hidden::basePick(variants, total, key, INVALID);
    };

    auto contains = [](const std::string& key) -> bool {
      return enums_hidden::baseContains(variants, total, key);
    };
  } // namespace CommBC

  namespace ParticlePusher {
    const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

    auto pick = [](const std::string& key) -> type {
      return enums_hidden::basePick(variants, total, key, INVALID);
    };

    auto contains = [](const std::string& key) -> bool {
      return enums_hidden::baseContains(variants, total, key);
    };
  } // namespace ParticlePusher

  namespace Cooling {
    const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

    auto pick = [](const std::string& key) -> type {
      return enums_hidden::basePick(variants, total, key, INVALID);
    };

    auto contains = [](const std::string& key) -> bool {
      return enums_hidden::baseContains(variants, total, key);
    };
  } // namespace Cooling

} // namespace ntt

#endif // GLOBAL_ENUMS_H
