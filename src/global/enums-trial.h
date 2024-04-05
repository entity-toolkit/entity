// /**
//  * @file enums.h
//  * @brief Global enum variables describing the simulation
//  * @implements
//  *   - enum ntt::em
//  *   - enum ntt::cur
//  *   - enum ntt::ParticleTag       // dead, alive
//  *   - enum ntt::Coord             // Cart, Sph, Qsph
//  *   - enum ntt::Metric            // Minkowski, Spherical, QSpherical,
//  *                                Kerr_Schild, QKerr_Schild, Kerr_Schild_0
//  *   - enum ntt::SimEngine  // SRPIC, GRPIC
//  *   - enum ntt::PrtlBC        // periodic, absorb, atmosphere, custom,
//  *                                reflect, horizon, axis, send
//  *   - enum ntt::FldsBC          // periodic, absorb, atmosphere, custom, horizon, axis
//  *   - enum ntt::CommBC            // physical, comm
//  *   - enum ntt::PrtlPusher    // boris, vay, boris,gca, vay,gca, photon, none
//  *   - enum ntt::Cooling           // synchrotron, none
//  * @depends:
//  *   - utils/error.h
//  *   - utils/formatting.h
//  *   - utils/log.h
//  * @namespaces:
//  *   - ntt::
//  * @note Enums of the same type can be compared with each other and with strings
//  * @note
//  * Comparison with strings is case-sensitive(!) since the latter are defined as
//  * string literals example: "srpIC" != SimEngine::SRPIC
//  * @note
//  * To convert an enum to a string, use its std::string()
//  * example: std::string(SimEngine::SRPIC) == "srpic"
//  * @note
//  * To check if a string is a valid option, use the contains() function
//  * example: PrtlPusher::contains("vay") == true
//  * @note
//  * To get the proper enum instance from a string, use the pick() function
//  * example: PrtlPusher::pick("vay") == PrtlPusher::VAY
//  * @note
//  * To get the total number of enum instances, use the total variable
//  * example: Cooling::total == 2
//  * @note
//  * To iterate over all enum instances, use the variants array
//  * example: for (const auto& cooling : Cooling::variants) { ... }
//  */

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
    class Enum {
    public:
      class Iterator {
      public:
        Iterator(int value) : m_value(value) {}

        T operator*(void) const {
          return (T)m_value;
        }

        void operator++(void) {
          ++m_value;
        }

        bool operator!=(Iterator rhs) {
          return m_value != rhs.m_value;
        }

      private:
        int m_value;
      };
    };

    template <typename T>
    typename Enum<T>::Iterator begin(Enum<T>) {
      return typename Enum<T>::Iterator((int)T::__first__);
    }

    template <typename T>
    typename Enum<T>::Iterator end(Enum<T>) {
      return typename Enum<T>::Iterator(((int)T::__last__) + 1);
    }
  } // namespace enums_hidden

  // enum class Coord {
  //   INVALID,
  //   Cart,
  //   Sph,
  //   Qsph,
  //   __first__ = Cart,
  //   __last__  = Qsph,
  // };
  class Coord {
  public:
    enum Value {
      INVALID,
      Cart,
      Sph,
      Qsph,
    };

    static constexpr Value variants[] = { Cart, Sph, Qsph };

    Coord() = default;

    constexpr Coord(Value c) : value(c) {}

    constexpr bool operator==(Coord other) const {
      return value == other.value;
    }

    constexpr bool operator!=(Coord other) const {
      return value != other.value;
    }

    // overload to_string (constexpr)
    constexpr auto to_string() const -> const char* {
      return enums_hidden::coord_lookup[value];
    }

    // overload <<
    friend std::ostream& operator<<(std::ostream& os, const Coord& c) {
      os << enums_hidden::coord_lookup[c.value];
      return os;
    }

  private:
    Value value;
  };

  namespace enums_hidden {

    static constexpr const char* coord_lookup[] = {
      "cart",
      "sph",
      "qsph",
    };
  } // namespace enums_hidden

  enum class SimEngine {
    INVALID,
    SRPIC,
    GRPIC,
    __first__ = SRPIC,
    __last__  = GRPIC,
  };

  namespace enums_hidden {
    static constexpr const char* simulation_engine_lookup[] = {
      "srpic",
      "grpic",
    };
  } // namespace enums_hidden

  enum class PrtlBC {
    INVALID,
    Periodic,
    Absorb,
    Atmosphere,
    Custom,
    Reflect,
    Horizon,
    Axis,
    Comm,
    __first__ = Periodic,
    __last__  = Comm,
  };

  namespace enums_hidden {
    static constexpr const char* particle_bc_lookup[] = {
      "periodic", "absorb",  "atmosphere", "custom",
      "reflect",  "horizon", "axis",       "comm",
    };
  } // namespace enums_hidden

  enum class FldsBC {
    INVALID,
    Periodic,
    Absorb,
    Atmosphere,
    Custom,
    Horizon,
    Axis,
    Comm,
    __first__ = Periodic,
    __last__  = Comm,
  };

  namespace enums_hidden {
    static constexpr const char* fields_bc_lookup[] = {
      "periodic", "absorb", "atmosphere", "custom", "horizon", "axis", "comm",
    };
  } // namespace enums_hidden

  enum class PrtlPusher {
    INVALID,
    Boris,
    Vay,
    Boris_GCA,
    Vay_GCA,
    Photon,
    None,
    __first__ = Boris,
    __last__  = None,
  };

  namespace enums_hidden {
    static constexpr const char* particle_pusher_lookup[] = {
      "boris", "vay", "boris,gca", "vay,gca", "photon", "none",
    };
  } // namespace enums_hidden

  enum class Cooling {
    INVALID,
    Synchrotron,
    None,
    __first__ = Synchrotron,
    __last__  = None,
  };

  namespace enums_hidden {
    static constexpr const char* cooling_lookup[] = {
      "synchrotron",
      "none",
    };
  } // namespace enums_hidden

  auto to_string(SimEngine e) -> const char* {
    return enums_hidden::simulation_engine_lookup[static_cast<uint32_t>(e)];
  }

  auto to_string(Coord c) -> const char* {
    return enums_hidden::coord_lookup[static_cast<uint32_t>(c)];
  }

  auto to_string(PrtlBC bc) -> const char* {
    return enums_hidden::particle_bc_lookup[static_cast<uint32_t>(bc)];
  }

  auto to_string(FldsBC bc) -> const char* {
    return enums_hidden::fields_bc_lookup[static_cast<uint32_t>(bc)];
  }

  auto to_string(PrtlPusher pp) -> const char* {
    return enums_hidden::particle_pusher_lookup[static_cast<uint32_t>(pp)];
  }

  auto to_string(Cooling c) -> const char* {
    return enums_hidden::cooling_lookup[static_cast<uint32_t>(c)];
  }

  namespace enums_hidden {
    template <typename T>
    auto pick(const std::string& key, const char* const* lookup) -> T {
      for (int i = 0; i <= (int)T::__last__; ++i) {
        if (fmt::toLower(key) == fmt::toLower(lookup[i])) {
          return (T)i;
        }
      }
      raise::Error("Invalid enum value: " + key, HERE);
      throw;
    }
  } // namespace enums_hidden

  auto pick(const std::string& key) -> SimEngine {
    return enums_hidden::pick(key, enums_hidden::simulation_engine_lookup);
  }

  auto pick(const std::string& key) -> Coord {
    return enums_hidden::pick(key, enums_hidden::coord_lookup);
  }

  auto pick(const std::string& key) -> PrtlBC {
    return enums_hidden::pick(key, enums_hidden::particle_bc_lookup);
  }

  auto pick(const std::string& key) -> FldsBC {
    return enums_hidden::pick(key, enums_hidden::fields_bc_lookup);
  }

  auto pick(const std::string& key) -> PrtlPusher {
    return enums_hidden::pick(key, enums_hidden::particle_pusher_lookup);
  }

  auto pick(const std::string& key) -> Cooling {
    return enums_hidden::pick(key, enums_hidden::cooling_lookup);
  }

  namespace enums_hidden {
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
