#ifndef CONFIG_H
#define CONFIG_H

#include <fmt/core.h>

#include <stdexcept>

#define STRINGIZE(x)        STRINGIZE_DETAIL(x)
#define STRINGIZE_DETAIL(x) #x
#define LINE_STRING         STRINGIZE(__LINE__)

#if defined(MPI_ENABLED)
  #include <mpi.h>

  #define NTTLog()                                                             \
    {                                                                          \
      int mpi_rank;                                                            \
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);                                \
      if (mpi_rank == 0) {                                                     \
        PLOGV_(ntt::LogFile);                                                  \
      }                                                                        \
    }

#else // not MPI_ENABLED
  #define NTTLog()                                                             \
    { PLOGV_(ntt::LogFile); }

#endif

#define NTTWarn(msg)                                                           \
  { PLOGW_(ntt::LogFile) << msg; }

#define NTTFatal()                                                             \
  { PLOGF_(ntt::LogFile); }

#define NTTHostError(msg)                                                      \
  {                                                                            \
    auto err = fmt::format("# ERROR: {}  : filename: {} : line: {}",           \
                           msg,                                                \
                           __FILE__,                                           \
                           LINE_STRING);                                       \
    NTTFatal();                                                                \
    throw std::runtime_error(err);                                             \
  }

#define NTTHostErrorIf(condition, msg)                                         \
  {                                                                            \
    if ((condition)) {                                                         \
      NTTHostError(msg);                                                       \
    }                                                                          \
  }

#if defined(GPU_ENABLED)
  #define NTTError(msg) ({})
#else // not GPU_ENABLED
  #define NTTError(msg) NTTHostError(msg)
#endif

// Defining precision-based constants and types
using prtldx_t = float;
#ifdef SINGLE_PRECISION
using real_t = float;
#else
using real_t = double;
#endif

inline constexpr unsigned int N_GHOSTS = 2;

#endif // CONFIG_H