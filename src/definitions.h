#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <plog/Log.h>

#include <cstdint>
#include <iomanip>
#include <string>
#include <vector>

#include <string_view>

/* -------------------------------------------------------------------------- */
/*                                   Macros                                   */
/* -------------------------------------------------------------------------- */

// Simple math expressions (for real-type arguments)
#ifdef SINGLE_PRECISION
inline constexpr float ONE    = 1.0f;
inline constexpr float TWO    = 2.0f;
inline constexpr float ZERO   = 0.0f;
inline constexpr float HALF   = 0.5f;
inline constexpr float INV_2  = 0.5f;
inline constexpr float INV_4  = 0.25f;
inline constexpr float INV_8  = 0.125f;
inline constexpr float INV_16 = 0.0625f;
inline constexpr float INV_32 = 0.03125f;
inline constexpr float INV_64 = 0.015625f;
#else
inline constexpr double ONE    = 1.0;
inline constexpr double TWO    = 2.0;
inline constexpr double ZERO   = 0.0;
inline constexpr double HALF   = 0.5;
inline constexpr double INV_2  = 0.5;
inline constexpr double INV_4  = 0.25;
inline constexpr double INV_8  = 0.125;
inline constexpr double INV_16 = 0.0625;
inline constexpr double INV_32 = 0.03125;
inline constexpr double INV_64 = 0.015625;
#endif

#define IMIN(a, b)   ((a) < (b) ? (a) : (b))
#define IMAX(a, b)   ((a) > (b) ? (a) : (b))
#define SIGNd(x)     (((x) < 0.0) ? -1.0 : 1.0)
#define SIGNf(x)     (((x) < 0.0f) ? -1.0f : 1.0f)
#define SIGN(x)      (((x) < ZERO) ? -ONE : ONE)
#define ABS(x)       (((x) < ZERO) ? -(x) : (x))
#define HEAVISIDE(x) (((x) <= ZERO) ? ZERO : ONE)
#define SQR(x)       ((x) * (x))
#define CUBE(x)      ((x) * (x) * (x))

/* -------------------------------------------------------------------------- */
/*                               Math constants                               */
/* -------------------------------------------------------------------------- */

namespace ntt {
  namespace constant {
    inline constexpr std::uint64_t RandomSeed = 0x123456789abcdef0;
    inline constexpr double        PI         = 3.14159265358979323846;
    inline constexpr double        INV_PI     = 0.31830988618379067154;
    inline constexpr double        PI_SQR     = 9.86960440108935861882;
    inline constexpr double        INV_PI_SQR = 0.10132118364233777144;
    inline constexpr double        TWO_PI     = 6.28318530717958647692;
    inline constexpr double        E          = 2.71828182845904523536;
    inline constexpr double        SQRT2      = 1.41421356237309504880;
    inline constexpr double        INV_SQRT2  = 0.70710678118654752440;
    inline constexpr double        SQRT3      = 1.73205080756887729352;
  }    // namespace constant
}    // namespace ntt

/* -------------------------------------------------------------------------- */
/*                           Enums and type aliases                           */
/* -------------------------------------------------------------------------- */

namespace ntt {
  enum { LogFile = 1, InfoFile };

  // Defining specific code configurations as enum classes
  enum class Dimension { ONE_D = 1, TWO_D = 2, THREE_D = 3 };

  enum class SimulationEngine { UNDEFINED, SANDBOX, PIC, GRPIC };
  enum class BoundaryCondition { UNDEFINED, PERIODIC, ABSORB, USER, OPEN, COMM, AXIS };
  enum class ParticlePusher { UNDEFINED, NONE, BORIS, VAY, PHOTON };

  inline constexpr auto Dim1          = Dimension::ONE_D;
  inline constexpr auto Dim2          = Dimension::TWO_D;
  inline constexpr auto Dim3          = Dimension::THREE_D;
  inline constexpr auto SANDBOXEngine = SimulationEngine::SANDBOX;
  inline constexpr auto PICEngine     = SimulationEngine::PIC;
  inline constexpr auto GRPICEngine   = SimulationEngine::GRPIC;

  // ND list alias
  template <typename T, Dimension D>
  using tuple_t = T[static_cast<short>(D)];

  // list alias of size N
  template <typename T, int N>
  using list_t = T[N];

  // ND coordinate alias
  template <Dimension D>
  using coord_t = tuple_t<real_t, D>;

  // ND vector alias
  template <Dimension D>
  using vec_t   = tuple_t<real_t, D>;

  using index_t = const std::size_t;
}    // namespace ntt

/* -------------------------------------------------------------------------- */
/*                                  Defaults                                  */
/* -------------------------------------------------------------------------- */

namespace ntt {
  namespace options {
    const std::vector<std::string> pushers = { "Boris", "Photon", "None" };
    const std::vector<std::string> boundaries
      = { "PERIODIC", "ABSORB", "USER", "OPEN", "AXIS" };
    const std::vector<std::string> outputs = { "disabled", "HDF5" };
  }    // namespace options

  namespace defaults {
    constexpr std::string_view input_filename    = "input";
    constexpr std::string_view output_path       = "output";

    const std::string          title             = "EntitySimulation";
    const int                  n_species         = 0;
    const std::string          em_pusher         = "Boris";
    const std::string          ph_pusher         = "photon";
    const std::string          metric            = "minkowski";

    const real_t               runtime           = 1e10;
    const real_t               correction        = 1.0;
    const real_t               cfl               = 0.95;

    const unsigned short       current_filters   = 0;

    const bool                 use_weights       = false;

    const int                  shuffle_interval  = 0;
    const double               max_dead_frac     = 0.0;

    const std::string          output_format     = options::outputs[0];
    const int                  output_interval   = 1;
    const int                  output_mom_smooth = 1;
  }    // namespace defaults

  // Field IDs used for io
  enum class FieldID {
    E,      // Electric fields
    B,      // Magnetic fields
    J,      // Current density
    T,      // Particle distribution moments
    Rho,    // Particle density
    N,      // Particle number density
    Nppc    // Raw number of particles per each cell
  };
}    // namespace ntt

/* -------------------------------------------------------------------------- */
/*                                Log formatter                               */
/* -------------------------------------------------------------------------- */

namespace plog {
  class Nt2ConsoleFormatter {
  public:
    static auto header() -> util::nstring {
      return util::nstring();
    }
    static auto format(const Record& record) -> util::nstring {
      util::nostringstream ss;
      if (record.getSeverity() == plog::debug
          && plog::get()->getMaxSeverity() == plog::verbose) {
        ss << PLOG_NSTR("\n") << record.getFunc() << PLOG_NSTR(" @ ") << record.getLine()
           << PLOG_NSTR("\n");
      }
      ss << std::setw(9) << std::left << severityToString(record.getSeverity())
         << PLOG_NSTR(": ");
      ss << record.getMessage() << PLOG_NSTR("\n");
      return ss.str();
    }
  };

  class Nt2InfoFormatter {
  public:
    static auto header() -> util::nstring {
      return util::nstring();
    }
    static auto format(const Record& record) -> util::nstring {
      util::nostringstream ss;
      ss << record.getMessage() << PLOG_NSTR("\n");
      return ss.str();
    }
  };
}    // namespace plog

#endif
