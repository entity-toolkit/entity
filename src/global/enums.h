/**
 * @file enums.h
 * @brief Global enum variables describing the simulation
 * @implements
 *   - enum ntt::em
 *   - enum ntt::cur
 *   - enum ntt::ParticleTag       // dead, alive
 *   - enum ntt::Coord             // Cart, Sph, Qsph
 *   - enum ntt::Metric            // Minkowski, Spherical, QSpherical,
 *                                Kerr_Schild, QKerr_Schild, Kerr_Schild_0
 *   - enum ntt::SimulationEngine  // SRPIC, GRPIC
 *   - enum ntt::ParticleBC        // periodic, absorb, atmosphere, custom,
 *                                reflect, horizon, axis, send
 *   - enum ntt::FieldsBC          // periodic, absorb, atmosphere, custom, horizon, axis
 *   - enum ntt::CommBC            // physical, comm
 *   - enum ntt::ParticlePusher    // boris, vay, boris,gca, vay,gca, photon, none
 *   - enum ntt::Cooling           // synchrotron, none
 * @depends:
 *   - utils/error.h
 *   - utils/formatting.h
 *   - utils/log.h
 * @namespaces:
 *   - ntt::
 * @note Enums of the same type can be compared with each other and with strings
 * @note
 * Comparison with strings is case-sensitive(!) since the latter are defined as
 * string literals example: "srpIC" != SimulationEngine::SRPIC
 * @note
 * To convert an enum to a string, use its std::string()
 * example: std::string(SimulationEngine::SRPIC) == "srpic"
 * @note
 * To check if a string is a valid option, use the contains() function
 * example: ParticlePusher::contains("vay") == true
 * @note
 * To get the proper enum instance from a string, use the pick() function
 * example: ParticlePusher::pick("vay") == ParticlePusher::VAY
 * @note
 * To get the total number of enum instances, use the total variable
 * example: Cooling::total == 2
 * @note
 * To iterate over all enum instances, use the variants array
 * example: for (const auto& cooling : Cooling::variants) { ... }
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
    template <typename T>
    auto baseContains(T* const arr, const std::size_t n, const std::string& elem)
      -> bool {
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
    using type = std::string_view;
    constexpr type INVALID { "invalid" };
    constexpr type CART { "cart" };
    constexpr type SPH { "sph" };
    constexpr type QSPH { "qsph" };
    constexpr type variants[] = { CART, SPH, QSPH };
  } // namespace Coord

  namespace Metric {
    using type = std::string_view;
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
    using type = std::string_view;
    const type INVALID { "invalid" };
    const type SRPIC { "srpic" };
    const type GRPIC { "grpic" };
    const type variants[] = { SRPIC, GRPIC };
  } // namespace SimulationEngine

  namespace ParticleBC {
    using type = std::string_view;
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
    using type = std::string_view;
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
    using type = std::string_view;
    const type INVALID { "invalid" };
    const type PHYSICAL { "physical" };
    const type COMM { "comm" };
    const type variants[] = { PHYSICAL, COMM };
  } // namespace CommBC

  namespace ParticlePusher {
    using type = std::string_view;
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
    using type = std::string_view;
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
