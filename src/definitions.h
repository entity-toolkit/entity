#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <plog/Log.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <string_view>
#include <type_traits>

/* -------------------------------------------------------------------------- */
/*                                   Macros                                   */
/* -------------------------------------------------------------------------- */

// Simple math expressions (for real-type arguments)
#ifdef SINGLE_PRECISION
inline constexpr float ONE    = 1.0f;
inline constexpr float TWO    = 2.0f;
inline constexpr float THREE  = 3.0f;
inline constexpr float FOUR   = 4.0f;
inline constexpr float FIVE   = 5.0f;
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
inline constexpr double THREE  = 3.0;
inline constexpr double FOUR   = 4.0;
inline constexpr double FIVE   = 5.0;
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

#define DOT(ax1, ax2, ax3, bx1, bx2, bx3)                                      \
  ((ax1) * (bx1) + (ax2) * (bx2) + (ax3) * (bx3))
#define NORM_SQR(ax1, ax2, ax3)                (DOT((ax1), (ax2), (ax3), (ax1), (ax2), (ax3)))
#define NORM(ax1, ax2, ax3)                    (math::sqrt(NORM_SQR((ax1), (ax2), (ax3))))
#define CROSS_x1(ax1, ax2, ax3, bx1, bx2, bx3) ((ax2) * (bx3) - (ax3) * (bx2))
#define CROSS_x2(ax1, ax2, ax3, bx1, bx2, bx3) ((ax3) * (bx1) - (ax1) * (bx3))
#define CROSS_x3(ax1, ax2, ax3, bx1, bx2, bx3) ((ax1) * (bx2) - (ax2) * (bx1))

/* -------------------------------------------------------------------------- */
/*                               Math constants                               */
/* -------------------------------------------------------------------------- */

namespace ntt {
  namespace constant {
    inline constexpr std::uint64_t RandomSeed = 0x123456789abcdef0;
    inline constexpr double        HALF_PI    = 1.57079632679489661923;
    inline constexpr double        PI         = 3.14159265358979323846;
    inline constexpr double        INV_PI     = 0.31830988618379067154;
    inline constexpr double        PI_SQR     = 9.86960440108935861882;
    inline constexpr double        INV_PI_SQR = 0.10132118364233777144;
    inline constexpr double        TWO_PI     = 6.28318530717958647692;
    inline constexpr double        E          = 2.71828182845904523536;
    inline constexpr double        SQRT2      = 1.41421356237309504880;
    inline constexpr double        INV_SQRT2  = 0.70710678118654752440;
    inline constexpr double        SQRT3      = 1.73205080756887729352;
  } // namespace constant
} // namespace ntt

/* -------------------------------------------------------------------------- */
/*                           Enums and type aliases                           */
/* -------------------------------------------------------------------------- */

namespace ntt {
  template <typename C, C beginVal, C endVal>
  class Iterator {
    typedef typename std::underlying_type<C>::type val_t;
    int                                            val;

  public:
    Iterator(const C& f) : val(static_cast<val_t>(f)) {}

    Iterator() : val(static_cast<val_t>(beginVal)) {}

    Iterator operator++() {
      ++val;
      return *this;
    }

    C operator*() {
      return static_cast<C>(val);
    }

    Iterator begin() {
      return *this;
    }

    Iterator end() {
      static const Iterator endIter = ++Iterator(endVal); // cache it
      return endIter;
    }

    bool operator!=(const Iterator& i) {
      return val != i.val;
    }
  };

  enum {
    LogFile = 1,
    InfoFile
  };

  // Defining specific code configurations as enum classes
  enum class Dimension {
    ONE_D   = 1,
    TWO_D   = 2,
    THREE_D = 3
  };

  template <Dimension D>
  struct DimensionTag {};

#if defined(MINKOWSKI_METRIC) || defined(GRPIC_ENGINE)
  #define FullD D
#else
  #define FullD Dim3
#endif

  enum class SimulationEngine {
    UNDEFINED,
    SANDBOX,
    PIC,
    GRPIC
  };
  enum class BoundaryCondition {
    UNDEFINED,
    PERIODIC,
    ABSORB,
    CUSTOM,
    OPEN,
    COMM,
    AXIS
  };
  enum class ParticlePusher {
    UNDEFINED,
    NONE,
    BORIS,
    VAY,
    BORIS_GCA,
    VAY_GCA,
    PHOTON
  };
  using PusherIterator =
    Iterator<ParticlePusher, ParticlePusher::UNDEFINED, ParticlePusher::PHOTON>;

  inline constexpr auto Dim1          = Dimension::ONE_D;
  inline constexpr auto Dim2          = Dimension::TWO_D;
  inline constexpr auto Dim3          = Dimension::THREE_D;
  inline constexpr auto SANDBOXEngine = SimulationEngine::SANDBOX;
  inline constexpr auto PICEngine     = SimulationEngine::PIC;
  inline constexpr auto GRPICEngine   = SimulationEngine::GRPIC;

  inline auto stringizeSimulationEngine(const SimulationEngine& sim)
    -> std::string {
    switch (sim) {
      case SANDBOXEngine:
        return "Sandbox";
      case PICEngine:
        return "PIC";
      case GRPICEngine:
        return "GRPIC";
      default:
        return "N/A";
    }
  }

  inline auto stringizeBoundaryCondition(const BoundaryCondition& bc)
    -> std::string {
    switch (bc) {
      case BoundaryCondition::PERIODIC:
        return "Periodic";
      case BoundaryCondition::ABSORB:
        return "Absorb";
      case BoundaryCondition::OPEN:
        return "Open";
      case BoundaryCondition::CUSTOM:
        return "Custom";
      case BoundaryCondition::AXIS:
        return "Axis";
      case BoundaryCondition::COMM:
        return "Comm";
      default:
        return "N/A";
    }
  }

  inline auto stringizeParticlePusher(const ParticlePusher& pusher) -> std::string {
    switch (pusher) {
      case ParticlePusher::BORIS:
        return "Boris";
      case ParticlePusher::VAY:
        return "Vay";
      case ParticlePusher::BORIS_GCA:
        return "Boris,GCA";
      case ParticlePusher::VAY_GCA:
        return "Vay,GCA";
      case ParticlePusher::PHOTON:
        return "Photon";
      case ParticlePusher::NONE:
        return "None";
      default:
        return "N/A";
    }
  }

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
  using vec_t = tuple_t<real_t, D>;

  using index_t = const std::size_t;

  using range_tuple_t = std::pair<std::size_t, std::size_t>;

  // Field IDs used for io
  enum class FieldID {
    E,      // Electric fields
    divE,   // Divergence of electric fields
    D,      // Electric fields (GR)
    divD,   // Divergence of electric fields (GR)
    B,      // Magnetic fields
    H,      // Magnetic fields (GR)
    J,      // Current density
    A,      // Vector potential
    T,      // Particle distribution moments
    Rho,    // Particle mass density
    Charge, // Charge density
    N,      // Particle number density
    Nppc    // Raw number of particles per each cell
  };

  enum class PrtlID {
    X, // Position
    U, // 4-Velocity / 4-Momentum
    W  // Weight
  };
} // namespace ntt

/* -------------------------------------------------------------------------- */
/*                                  Defaults                                  */
/* -------------------------------------------------------------------------- */

namespace ntt {
  namespace options {
    const std::vector<std::string> pushers = { "Boris",   "Vay",       "Photon",
                                               "Vay,GCA", "Boris,GCA", "None" };
    const std::vector<std::string> boundaries = { "PERIODIC",
                                                  "ABSORB",
                                                  "CUSTOM",
                                                  "OPEN",
                                                  "AXIS" };
    const std::vector<std::string> outputs    = { "disabled", "HDF5", "BP5" };
  } // namespace options

  namespace defaults {
    constexpr std::string_view input_filename = "input";
    constexpr std::string_view output_path    = "output";

    const std::string title     = "EntitySimulation";
    const int         n_species = 0;
    const std::string em_pusher = "Boris";
    const std::string ph_pusher = "Photon";
    const std::string metric    = "minkowski";

    const real_t runtime    = 1e10;
    const real_t correction = 1.0;
    const real_t cfl        = 0.95;

    const unsigned short current_filters = 0;

    const bool use_weights = false;

    const std::string output_format      = options::outputs[0];
    const int         output_interval    = 1;
    const int         output_mom_smooth  = 1;
    const std::size_t output_prtl_stride = 100;

    const std::string log_level       = "info";
    const int         diag_interval   = 1;
    const bool        blocking_timers = false;
  } // namespace defaults
} // namespace ntt

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
      if (record.getSeverity() == plog::debug &&
          plog::get()->getMaxSeverity() == plog::verbose) {
        ss << PLOG_NSTR("\n") << record.getFunc() << PLOG_NSTR(" @ ")
           << record.getLine() << PLOG_NSTR("\n");
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
} // namespace plog

#if !defined(MPI_ENABLED)
template <typename Func, typename... Args>
void PrintOnce(Func func, Args&&... args) {
  func(std::forward<Args>(args)...);
}
#endif

#if defined(MPI_ENABLED)

  #include <mpi.h>

/* -------------------------------------------------------------------------- */
/*                                     MPI                                    */
/* -------------------------------------------------------------------------- */

template <typename Func, typename... Args>
void PrintOnce(Func func, Args&&... args) {
  int rank, root_rank { 0 };
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == root_rank) {
    func(std::forward<Args>(args)...);
  }
}

template <typename T>
[[nodiscard]]
constexpr MPI_Datatype mpi_get_type() noexcept {
  MPI_Datatype mpi_type = MPI_DATATYPE_NULL;

  if constexpr (std::is_same<T, char>::value) {
    mpi_type = MPI_CHAR;
  } else if constexpr (std::is_same<T, signed char>::value) {
    mpi_type = MPI_SIGNED_CHAR;
  } else if constexpr (std::is_same<T, unsigned char>::value) {
    mpi_type = MPI_UNSIGNED_CHAR;
  } else if constexpr (std::is_same<T, wchar_t>::value) {
    mpi_type = MPI_WCHAR;
  } else if constexpr (std::is_same<T, signed short>::value) {
    mpi_type = MPI_SHORT;
  } else if constexpr (std::is_same<T, unsigned short>::value) {
    mpi_type = MPI_UNSIGNED_SHORT;
  } else if constexpr (std::is_same<T, signed int>::value) {
    mpi_type = MPI_INT;
  } else if constexpr (std::is_same<T, unsigned int>::value) {
    mpi_type = MPI_UNSIGNED;
  } else if constexpr (std::is_same<T, signed long int>::value) {
    mpi_type = MPI_LONG;
  } else if constexpr (std::is_same<T, unsigned long int>::value) {
    mpi_type = MPI_UNSIGNED_LONG;
  } else if constexpr (std::is_same<T, signed long long int>::value) {
    mpi_type = MPI_LONG_LONG;
  } else if constexpr (std::is_same<T, unsigned long long int>::value) {
    mpi_type = MPI_UNSIGNED_LONG_LONG;
  } else if constexpr (std::is_same<T, float>::value) {
    mpi_type = MPI_FLOAT;
  } else if constexpr (std::is_same<T, double>::value) {
    mpi_type = MPI_DOUBLE;
  } else if constexpr (std::is_same<T, long double>::value) {
    mpi_type = MPI_LONG_DOUBLE;
  } else if constexpr (std::is_same<T, std::int8_t>::value) {
    mpi_type = MPI_INT8_T;
  } else if constexpr (std::is_same<T, std::int16_t>::value) {
    mpi_type = MPI_INT16_T;
  } else if constexpr (std::is_same<T, std::int32_t>::value) {
    mpi_type = MPI_INT32_T;
  } else if constexpr (std::is_same<T, std::int64_t>::value) {
    mpi_type = MPI_INT64_T;
  } else if constexpr (std::is_same<T, std::uint8_t>::value) {
    mpi_type = MPI_UINT8_T;
  } else if constexpr (std::is_same<T, std::uint16_t>::value) {
    mpi_type = MPI_UINT16_T;
  } else if constexpr (std::is_same<T, std::uint32_t>::value) {
    mpi_type = MPI_UINT32_T;
  } else if constexpr (std::is_same<T, std::uint64_t>::value) {
    mpi_type = MPI_UINT64_T;
  } else if constexpr (std::is_same<T, bool>::value) {
    mpi_type = MPI_C_BOOL;
  }

  assert(mpi_type != MPI_DATATYPE_NULL);
  return mpi_type;
}
#endif

#endif
