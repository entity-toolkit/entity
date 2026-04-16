/**
 * @file enums.h
 * @brief Special enum variables describing the simulation
 * @implements
 *   - enum ntt::Coord             // Cart, Sph, Qsph
 *   - enum ntt::Metric            // Minkowski, Spherical, QSpherical,
 *                                    Kerr_Schild, QKerr_Schild, Kerr_Schild_0
 *   - enum ntt::SimEngine         // SRPIC, GRPIC
 *   - enum ntt::PrtlBC            // periodic, absorb, atmosphere, custom,
 *                                    reflect, horizon, axis, sync
 *   - enum ntt::FldsBC            // periodic, match, fixed, atmosphere,
 *                                    custom, horizon, axis, conductor, sync
 *   - enum ntt::FldsID            // e, dive, d, divd, b, h, j,
 *                                    a, t, rho, charge, n, nppc, v, custom
 *   - enum ntt::StatsID           // b^2, e^2, exb, j.e, t, rho,
 *                                    charge, n, npart
 *
 *   - enum ntt::ParticlePusher    // photon, boris, vay, gca, none
 *   - enum ntt::RadiativeDrag     // compton, synchrotron, none
 *   - enum ntt::EmissionType      // none, synchrotron, inversecompton, custom
 *
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
 * example: PrtlBC::contains("periodic") == true
 * @note
 * To get the proper enum instance from a string, use the pick() function
 * example: PrtlBC::pick("periodic") [returns PrtlBC::PERIODIC]
 * @note
 * To get the total number of enum instances, use the total variable
 * example: SimEngine::total == 2
 * @note
 * To iterate over all enum instances, use the variants array
 * example: for (auto s : SimEngine::variants) { ... }
 */

#ifndef GLOBAL_ENUMS_H
#define GLOBAL_ENUMS_H

#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"

#include <cstring>
#include <string>

namespace ntt {

  namespace enums_hidden {
    template <typename Derived>
    class EnumBase {
      constexpr auto idx() const -> std::size_t {
        return static_cast<std::size_t>(d().val) - 1; // assumes 1-indexed
      }

      constexpr auto d() const -> const Derived& {
        return static_cast<const Derived&>(*this);
      }

    public:
      constexpr bool operator==(Derived o) const noexcept {
        return d().val == o.val;
      }

      constexpr bool operator!=(Derived o) const noexcept {
        return d().val != o.val;
      }

      constexpr bool operator==(const char* s) const noexcept {
        return std::strcmp(Derived::lookup[idx()], s) == 0;
      }

      constexpr bool operator!=(const char* s) const noexcept {
        return std::strcmp(Derived::lookup[idx()], s) != 0;
      }

      constexpr auto to_string() const -> const char* {
        return Derived::lookup[idx()];
      }

      static auto pick(const char* s) -> Derived {
        for (auto i { 0 }; i < Derived::total; ++i) {
          if (std::strcmp(Derived::lookup[i], s) == 0) {
            return Derived { Derived::variants[i] };
          }
        }
        raise::Error(fmt::format("Invalid %s: %s", Derived::label, s), HERE);
        return Derived { Derived::variants[0] };
      }

      static auto contains(const char* s) -> bool {
        for (auto i { 0u }; i < Derived::total; ++i) {
          if (std::strcmp(Derived::lookup[i], s) == 0) {
            return true;
          }
        }
        return false;
      }
    };

  } // namespace enums_hidden

  struct Coord : public enums_hidden::EnumBase<Coord> {
    enum class type : uint8_t {
      INVALID   = 0,
      Cartesian = 1,
      Spherical,
      Qspherical
    };
    using enum type;
    type val;

    constexpr Coord(type v) noexcept : val { v } {}

    static constexpr const char* label = "coord";
    static constexpr type variants[]   = { Cartesian, Spherical, Qspherical };
    static constexpr const char* lookup[] = { "cart", "sph", "qsph" };
    static constexpr std::size_t total    = std::size(variants);
  };

  struct Metric : public enums_hidden::EnumBase<Metric> {
    enum class type : uint8_t {
      INVALID       = 0,
      Minkowski     = 1,
      Spherical     = 2,
      QSpherical    = 3,
      Kerr_Schild   = 4,
      QKerr_Schild  = 5,
      Kerr_Schild_0 = 6,
    };
    using enum type;
    type val;

    constexpr Metric(type v) noexcept : val { v } {}

    static constexpr const char* label      = "metric";
    static constexpr type        variants[] = { Minkowski,    Spherical,
                                                QSpherical,   Kerr_Schild,
                                                QKerr_Schild, Kerr_Schild_0 };
    static constexpr const char* lookup[]   = { "minkowski",    "spherical",
                                                "qspherical",   "kerr_schild",
                                                "qkerr_schild", "kerr_schild_0" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  struct SimEngine : public enums_hidden::EnumBase<SimEngine> {
    enum class type : uint8_t {
      INVALID = 0,
      SRPIC   = 1,
      GRPIC   = 2,
    };
    using enum type;
    type val;

    constexpr SimEngine(type v) noexcept : val { v } {}

    static constexpr const char* label      = "sim_engine";
    static constexpr type        variants[] = { SRPIC, GRPIC };
    static constexpr const char* lookup[]   = { "srpic", "grpic" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  struct PrtlBC : public enums_hidden::EnumBase<PrtlBC> {
    enum class type : uint8_t {
      INVALID    = 0,
      PERIODIC   = 1,
      ABSORB     = 2,
      ATMOSPHERE = 3,
      CUSTOM     = 4,
      REFLECT    = 5,
      HORIZON    = 6,
      AXIS       = 7,
      SYNC       = 8,
    };
    using enum type;
    type val;

    constexpr PrtlBC(type v) noexcept : val { v } {}

    static constexpr const char* label = "prtl_bc";
    static constexpr type variants[] = { PERIODIC, ABSORB,  ATMOSPHERE, CUSTOM,
                                         REFLECT,  HORIZON, AXIS,       SYNC };
    static constexpr const char* lookup[] = { "periodic",   "absorb",
                                              "atmosphere", "custom",
                                              "reflect",    "horizon",
                                              "axis",       "sync" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  struct FldsBC : public enums_hidden::EnumBase<FldsBC> {
    enum class type : uint8_t {
      INVALID    = 0,
      PERIODIC   = 1,
      MATCH      = 2,
      FIXED      = 3,
      ATMOSPHERE = 4,
      CUSTOM     = 5,
      HORIZON    = 6,
      AXIS       = 7,
      CONDUCTOR  = 8,
      SYNC       = 9 // <- SYNC means synchronization with other domains
    };
    using enum type;
    type val;

    constexpr FldsBC(type v) noexcept : val { v } {}

    static constexpr const char* label      = "flds_bc";
    static constexpr type        variants[] = {
      PERIODIC, MATCH, FIXED,     ATMOSPHERE, CUSTOM,
      HORIZON,  AXIS,  CONDUCTOR, SYNC,
    };
    static constexpr const char* lookup[] = {
      "periodic", "match", "fixed",     "atmosphere", "custom",
      "horizon",  "axis",  "conductor", "sync"
    };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  struct FldsID : public enums_hidden::EnumBase<FldsID> {
    enum class type : uint8_t {
      INVALID = 0,
      E       = 1,
      divE    = 2,
      D       = 3,
      divD    = 4,
      B       = 5,
      H       = 6,
      J       = 7,
      A       = 8,
      T       = 9,
      Rho     = 10,
      Charge  = 11,
      N       = 12,
      Nppc    = 13,
      V       = 14,
      Custom  = 15,
    };
    using enum type;
    type val;

    constexpr FldsID(type v) noexcept : val { v } {}

    static constexpr const char* label      = "out_flds";
    static constexpr type        variants[] = { E,      divE, D,    divD, B,
                                                H,      J,    A,    T,    Rho,
                                                Charge, N,    Nppc, V,    Custom };
    static constexpr const char* lookup[] = { "e",    "dive", "d",      "divd",
                                              "b",    "h",    "j",      "a",
                                              "t",    "rho",  "charge", "n",
                                              "nppc", "v",    "custom" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  struct StatsID : public enums_hidden::EnumBase<StatsID> {
    enum class type : uint8_t {
      INVALID = 0,
      B2      = 1,
      E2      = 2,
      ExB     = 3,
      JdotE   = 4,
      T       = 5,
      Rho     = 6,
      Charge  = 7,
      N       = 8,
      Npart   = 9,
      Custom  = 10,
    };
    using enum type;
    type val;

    constexpr StatsID(type v) noexcept : val { v } {}

    static constexpr const char* label      = "out_stats";
    static constexpr type        variants[] = { B2,  E2,     ExB, JdotE, T,
                                                Rho, Charge, N,   Npart, Custom };
    static constexpr const char* lookup[] = { "b^2",   "e^2",   "exb",    "j.e",
                                              "t",     "rho",   "charge", "n",
                                              "npart", "custom" };
    static constexpr std::size_t total = sizeof(variants) / sizeof(variants[0]);
  };

  namespace ParticlePusher {
    enum ParticlePusherFlags_ {
      NONE   = 0,
      PHOTON = 1 << 0,
      BORIS  = 1 << 1,
      VAY    = 1 << 2,
      GCA    = 1 << 3,
    };

    inline auto to_string(int flags) -> std::string {
      if (flags == NONE) {
        return "none";
      } else {
        std::string result = "";
        if (flags & PHOTON) {
          result += "photon";
        } else if (flags & BORIS) {
          result += "boris";
        } else if (flags & VAY) {
          result += "vay";
        }
        if (flags & GCA) {
          if (!result.empty()) {
            result += ",";
          }
          result += "gca";
        }
        return result;
      }
    }
  } // namespace ParticlePusher

  typedef int ParticlePusherFlags;

  namespace RadiativeDrag {
    enum RadiativeDragFlags_ {
      NONE        = 0,
      SYNCHROTRON = 1 << 0,
      COMPTON     = 1 << 1,
    };

    inline auto to_string(int flags) -> std::string {
      if (flags == NONE) {
        return "none";
      } else {
        std::string result = "";
        if (flags & SYNCHROTRON) {
          result += "synchrotron";
        }
        if (flags & COMPTON) {
          if (!result.empty()) {
            result += ",";
          }
          result += "compton";
        }
        return result;
      }
    }
  } // namespace RadiativeDrag

  typedef int RadiativeDragFlags;

  namespace EmissionType {
    enum EmissionTypeFlag_ {
      NONE        = 0,
      SYNCHROTRON = 1,
      COMPTON     = 2,
      CUSTOM      = 3,
    };

    inline auto to_string(int flags) -> std::string {
      switch (flags) {
        case NONE:
          return "none";
        case SYNCHROTRON:
          return "synchrotron";
        case COMPTON:
          return "compton";
        case CUSTOM:
          return "custom";
        default:
          return "unknown";
      }
    }
  } // namespace EmissionType

  typedef int EmissionTypeFlag;

} // namespace ntt

#endif // GLOBAL_ENUMS_H
