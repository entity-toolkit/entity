/**
 * @file enums.h
 * @brief Global enum variables describing the simulation
 * @implements
 *   - enum ntt::em
 *   - enum ntt::cur
 *   - enum ntt::ParticleTag       // dead, alive
 *   - enum ntt::Coord             // Cart, Sph, Qsph
 *   - enum ntt::Metric            // Minkowski, Spherical, QSpherical,
 *                                    Kerr_Schild, QKerr_Schild, Kerr_Schild_0
 *   - enum ntt::SimEngine         // SRPIC, GRPIC
 *   - enum ntt::PrtlBC            // periodic, absorb, atmosphere, custom,
 *                                    reflect, horizon, axis, send
 *   - enum ntt::FldsBC            // periodic, absorb, atmosphere, custom, horizon, axis
 *   - enum ntt::CommBC            // physical, comm
 *   - enum ntt::PrtlPusher        // boris, vay, boris,gca, vay,gca, photon, none
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
 * string literals example: "srpIC" != SimEngine::SRPIC
 * @note
 * To convert an enum to a string, use its std::string()
 * example: SimEngine(SimEngine::SRPIC).to_string() [return "srpic"]
 * @note
 * To check if a string is a valid option, use the contains() function
 * example: PrtlPusher::contains("vay") == true
 * @note
 * To get the proper enum instance from a string, use the pick() function
 * example: PrtlPusher::pick("vay") [returns PrtlPusher(PrtlPusher::VAY)]
 * @note
 * To get the total number of enum instances, use the total variable
 * example: Cooling::total == 2
 * @note
 * To iterate over all enum instances, use the variants array
 * example: for (Cooling c : Cooling::variants) { ... }
 */

#ifndef GLOBAL_ENUMS_H
#define GLOBAL_ENUMS_H

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include <algorithm>
#include <cstring>
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
    constexpr auto basePick(const enum T::type arr[],
                            const char* const* arr_c,
                            const std::size_t  n,
                            const char*        elem) -> T {
      for (std::size_t i = 0; i < n; ++i) {
        if (strcmp(arr_c[i], elem) == 0) {
          return (T)(arr[i]);
        }
      }
      raise::Error("Invalid enum value: " + std::string(elem), HERE);
      return T::INVALID;
    }

    template <typename T>
    constexpr auto baseContains(const char* const* arr_c,
                                const std::size_t  n,
                                const char*        elem) -> bool {
      for (std::size_t i = 0; i < n; ++i) {
        if (strcmp(arr_c[i], elem) == 0) {
          return true;
        }
      }
      return false;
    }

    template <class T>
    class BaseEnum {
    public:
      constexpr bool operator==(BaseEnum<T> other) const {
        return value == other.value;
      }

      constexpr bool operator!=(BaseEnum<T> other) const {
        return value != other.value;
      }

      constexpr bool operator==(uint8_t other) const {
        return value == other;
      }

      constexpr bool operator!=(uint8_t other) const {
        return value != other;
      }

      constexpr bool operator==(const char* other) const {
        return strcmp(T::lookup[value - 1], other) == 0;
      }

      constexpr bool operator!=(const char* other) const {
        return strcmp(T::lookup[value - 1], other) != 0;
      }

      static constexpr auto pick(const char* c) -> T {
        return basePick<T>(T::variants, T::lookup, T::total, c);
      }

      static constexpr auto contains(const char* c) -> bool {
        return baseContains<T>(T::lookup, T::total, c);
      }

      constexpr auto to_string() const -> const char* {
        return T::lookup[value - 1];
      }

      BaseEnum() = default;

      constexpr BaseEnum(uint8_t v) : value(v) {}

    protected:
      const uint8_t value;
    };
  } // namespace enums_hidden

  class Coord : public enums_hidden::BaseEnum<Coord> {
  public:
    enum type : uint8_t {
      INVALID = 0,
      Cart    = 1,
      Sph     = 2,
      Qsph    = 3,
    };

    constexpr Coord(uint8_t c) : enums_hidden::BaseEnum<Coord> { c } {}

    static constexpr type        variants[] = { Cart, Sph, Qsph };
    static constexpr const char* lookup[]   = { "cart", "sph", "qsph" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  class Metric : public enums_hidden::BaseEnum<Metric> {
  public:
    enum type : uint8_t {
      INVALID       = 0,
      Minkowski     = 1,
      Spherical     = 2,
      QSpherical    = 3,
      Kerr_Schild   = 4,
      QKerr_Schild  = 5,
      Kerr_Schild_0 = 6,
    };

    constexpr Metric(uint8_t c) : enums_hidden::BaseEnum<Metric> { c } {}

    static constexpr type        variants[] = { Minkowski,    Spherical,
                                                QSpherical,   Kerr_Schild,
                                                QKerr_Schild, Kerr_Schild_0 };
    static constexpr const char* lookup[]   = { "minkowski",    "spherical",
                                                "qspherical",   "kerr_schild",
                                                "qkerr_schild", "kerr_schild_0" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  class SimEngine : public enums_hidden::BaseEnum<SimEngine> {
  public:
    enum type : uint8_t {
      INVALID = 0,
      SRPIC   = 1,
      GRPIC   = 2,
    };

    constexpr SimEngine(uint8_t c) : enums_hidden::BaseEnum<SimEngine> { c } {}

    static constexpr type        variants[] = { SRPIC, GRPIC };
    static constexpr const char* lookup[]   = { "srpic", "grpic" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  class PrtlBC : public enums_hidden::BaseEnum<PrtlBC> {
  public:
    enum type : uint8_t {
      INVALID    = 0,
      PERIODIC   = 1,
      ABSORB     = 2,
      ATMOSPHERE = 3,
      CUSTOM     = 4,
      REFLECT    = 5,
      HORIZON    = 6,
      AXIS       = 7,
      SEND       = 8,
    };

    constexpr PrtlBC(uint8_t c) : enums_hidden::BaseEnum<PrtlBC> { c } {}

    static constexpr type variants[] = { PERIODIC, ABSORB,  ATMOSPHERE, CUSTOM,
                                         REFLECT,  HORIZON, AXIS,       SEND };
    static constexpr const char* lookup[] = { "periodic",   "absorb",
                                              "atmosphere", "custom",
                                              "reflect",    "horizon",
                                              "axis",       "send" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  class FldsBC : public enums_hidden::BaseEnum<FldsBC> {
  public:
    enum type : uint8_t {
      INVALID    = 0,
      PERIODIC   = 1,
      ABSORB     = 2,
      ATMOSPHERE = 3,
      CUSTOM     = 4,
      HORIZON    = 5,
      AXIS       = 6,
    };

    constexpr FldsBC(uint8_t c) : enums_hidden::BaseEnum<FldsBC> { c } {}

    static constexpr type        variants[] = { PERIODIC, ABSORB,  ATMOSPHERE,
                                                CUSTOM,   HORIZON, AXIS };
    static constexpr const char* lookup[]   = { "periodic",   "absorb",
                                                "atmosphere", "custom",
                                                "horizon",    "axis" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  class CommBC : public enums_hidden::BaseEnum<CommBC> {
  public:
    enum type : uint8_t {
      INVALID  = 0,
      PHYSICAL = 1,
      COMM     = 2,
    };

    constexpr CommBC(uint8_t c) : enums_hidden::BaseEnum<CommBC> { c } {}

    static constexpr type        variants[] = { PHYSICAL, COMM };
    static constexpr const char* lookup[]   = { "physical", "comm" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  class PrtlPusher : public enums_hidden::BaseEnum<PrtlPusher> {
  public:
    enum type : uint8_t {
      INVALID   = 0,
      BORIS     = 1,
      VAY       = 2,
      BORIS_GCA = 3,
      VAY_GCA   = 4,
      PHOTON    = 5,
      NONE      = 6,
    };

    constexpr PrtlPusher(uint8_t c) :
      enums_hidden::BaseEnum<PrtlPusher> { c } {}

    static constexpr type        variants[] = { BORIS,   VAY,    BORIS_GCA,
                                                VAY_GCA, PHOTON, NONE };
    static constexpr const char* lookup[] = { "boris",   "vay",    "boris,gca",
                                              "vay,gca", "photon", "none" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  class Cooling : public enums_hidden::BaseEnum<Cooling> {
  public:
    enum type : uint8_t {
      INVALID     = 0,
      SYNCHROTRON = 1,
      NONE        = 2,
    };

    constexpr Cooling(uint8_t c) : enums_hidden::BaseEnum<Cooling> { c } {}

    static constexpr type        variants[] = { SYNCHROTRON, NONE };
    static constexpr const char* lookup[]   = { "synchrotron", "none" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  // namespace Coord {
  //   using type = std::string_view;
  //   constexpr type INVALID { "invalid" };
  //   constexpr type CART { "cart" };
  //   constexpr type SPH { "sph" };
  //   constexpr type QSPH { "qsph" };
  //   constexpr type variants[] = { CART, SPH, QSPH };
  // } // namespace Coord

  // namespace Metric {
  //   using type = std::string_view;
  //   const type INVALID { "invalid" };
  //   const type MINKOWSKI { "minkowski" };
  //   const type SPHERICAL { "spherical" };
  //   const type QSPHERICAL { "qspherical" };
  //   const type KERR_SCHILD { "kerr_schild" };
  //   const type QKERR_SCHILD { "qkerr_schild" };
  //   const type KERR_SCHILD_0 { "kerr_schild_0" };
  //   const type variants[] = { MINKOWSKI,   SPHERICAL,    QSPHERICAL,
  //                             KERR_SCHILD, QKERR_SCHILD, KERR_SCHILD_0 };
  // } // namespace Metric

  // // namespace SimEngine {
  // //   // using type = std::string_view;

  // //   // struct INVALID {
  // //   //   static constexpr operator()() const -> std::string_view {
  // //   //     return "invalid";
  // //   //   }
  // //   //   INVALID() = delete;
  // //   //   ~INVALID() = delete;
  // //   // };

  // //   // enum : std::string {
  // //   //

  // namespace Coord {
  //   using type = std::string_view;
  //   constexpr type INVALID { "invalid" };
  //   constexpr type CART { "cart" };
  //   constexpr type SPH { "sph" };
  //   constexpr type QSPH { "qsph" };
  //   constexpr type variants[] = { CART, SPH, QSPH };
  // } // namespace Coord

  // namespace Metric {
  //   using type = std::string_view;
  //   const type INVALID { "invalid" };
  //   const type MINKOWSKI { "minkowski" };
  //   const type SPHERICAL { "spherical" };
  //   const type QSPHERICAL { "qspherical" };
  //   const type KERR_SCHILD { "kerr_schild" };
  //   const type QKERR_SCHILD { "qkerr_schild" };
  //   const type KERR_SCHILD_0 { "kerr_schild_0" };
  //   const type variants[] = { MINKOWSKI,   SPHERICAL,    QSPHERICAL,
  //                             KERR_SCHILD, QKERR_SCHILD, KERR_SCHILD_0 };
  // } // namespace Metric

  // // namespace SimEngine {
  // //   // using type = std::string_view;

  // //   // struct INVALID {
  // //   //   static constexpr operator()() const -> std::string_view {
  // //   //     return "invalid";
  // //   //   }
  // //   //   INVALID() = delete;
  // //   //   ~INVALID() = delete;
  // //   // };

  // //   // enum : std::string {
  // //   //   INVALID = "invalid",
  // //   //   SRPIC   = "srpic",
  // //   //   GRPIC   = "grpic",
  // //   // };

  // //   // const type INVALID { "invalid" };
  // //   // const type SRPIC { "srpic" };
  // //   // const type GRPIC { "grpic" };
  // //   const type variants[] = { SRPIC, GRPIC };
  // // } // namespace SimEngine

  // namespace PrtlBC {
  //   using type = std::string_view;
  //   const type INVALID { "invalid" };
  //   const type PERIODIC { "periodic" };
  //   const type ABSORB { "absorb" };
  //   const type ATMOSPHERE { "atmosphere" };
  //   const type CUSTOM { "custom" };
  //   const type REFLECT { "reflect" };
  //   const type HORIZON { "horizon" };
  //   const type AXIS { "axis" };
  //   const type SEND { "send" };
  //   const type variants[] = { PERIODIC, ABSORB,  ATMOSPHERE, CUSTOM,
  //                             REFLECT,  HORIZON, AXIS,       SEND };
  // } // namespace PrtlBC

  // namespace FldsBC {
  //   using type = std::string_view;
  //   const type INVALID { "invalid" };
  //   const type PERIODIC { "periodic" };
  //   const type ABSORB { "absorb" };
  //   const type ATMOSPHERE { "atmosphere" };
  //   const type CUSTOM { "custom" };
  //   const type HORIZON { "horizon" };
  //   const type AXIS { "axis" };
  //   const type variants[] = { PERIODIC, ABSORB,  ATMOSPHERE,
  //                             CUSTOM,   HORIZON, AXIS };
  // } // namespace FldsBC

  // namespace CommBC {
  //   using type = std::string_view;
  //   const type INVALID { "invalid" };
  //   const type PHYSICAL { "physical" };
  //   const type COMM { "comm" };
  //   const type variants[] = { PHYSICAL, COMM };
  // } // namespace CommBC

  // namespace PrtlPusher {
  //   using type = std::string_view;
  //   const type INVALID { "invalid" };
  //   const type BORIS { "boris" };
  //   const type VAY { "vay" };
  //   const type BORIS_GCA { "boris,gca" };
  //   const type VAY_GCA { "vay,gca" };
  //   const type PHOTON { "photon" };
  //   const type NONE { "none" };
  //   const type variants[] = { BORIS, VAY, BORIS_GCA, VAY_GCA, PHOTON, NONE };
  // } // namespace PrtlPusher

  // namespace Cooling {
  //   using type = std::string_view;
  //   const type INVALID { "invalid" };
  //   const type SYNCHROTRON { "synchrotron" };
  //   const type NONE { "none" };
  //   const type variants[] = { SYNCHROTRON, NONE };
  // } // namespace Cooling

  // namespace Coord {
  //   const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

  //   auto pick = [](const std::string& key) -> type {
  //     return enums_hidden::basePick(variants, total, key, INVALID);
  //   };

  //   auto contains = [](const std::string& key) -> bool {
  //     return enums_hidden::baseContains(variants, total, key);
  //   };
  // } // namespace Coord

  // namespace Metric {
  //   const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

  //   auto pick = [](const std::string& key) -> type {
  //     return enums_hidden::basePick(variants, total, key, INVALID);
  //   };

  //   auto contains = [](const std::string& key) -> bool {
  //     return enums_hidden::baseContains(variants, total, key);
  //   };
  // } // namespace Metric

  // namespace SimEngine {
  //   const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

  //   auto pick = [](const std::string& key) -> type {
  //     return enums_hidden::basePick(variants, total, key, INVALID);
  //   };

  //   auto contains = [](const std::string& key) -> bool {
  //     return enums_hidden::baseContains(variants, total, key);
  //   };
  // } // namespace SimEngine

  // namespace PrtlBC {
  //   const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

  //   auto pick = [](const std::string& key) -> type {
  //     return enums_hidden::basePick(variants, total, key, INVALID);
  //   };

  //   auto contains = [](const std::string& key) -> bool {
  //     return enums_hidden::baseContains(variants, total, key);
  //   };
  // } // namespace PrtlBC

  // namespace FldsBC {
  //   const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

  //   auto pick = [](const std::string& key) -> type {
  //     return enums_hidden::basePick(variants, total, key, INVALID);
  //   };

  //   auto contains = [](const std::string& key) -> bool {
  //     return enums_hidden::baseContains(variants, total, key);
  //   };
  // } // namespace FldsBC

  // namespace CommBC {
  //   const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

  //   auto pick = [](const std::string& key) -> type {
  //     return enums_hidden::basePick(variants, total, key, INVALID);
  //   };

  //   auto contains = [](const std::string& key) -> bool {
  //     return enums_hidden::baseContains(variants, total, key);
  //   };
  // } // namespace CommBC

  // namespace PrtlPusher {
  //   const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

  //   auto pick = [](const std::string& key) -> type {
  //     return enums_hidden::basePick(variants, total, key, INVALID);
  //   };

  //   auto contains = [](const std::string& key) -> bool {
  //     return enums_hidden::baseContains(variants, total, key);
  //   };
  // } // namespace PrtlPusher

  // namespace Cooling {
  //   const std::size_t total { sizeof(variants) / sizeof(variants[0]) };

  //   auto pick = [](const std::string& key) -> type {
  //     return enums_hidden::basePick(variants, total, key, INVALID);
  //   };

  //   auto contains = [](const std::string& key) -> bool {
  //     return enums_hidden::baseContains(variants, total, key);
  //   };
  // } // namespace Cooling

} // namespace ntt

#endif // GLOBAL_ENUMS_H
