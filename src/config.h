#ifndef CONFIG_H
#define CONFIG_H

#include <memory>
#include <stdexcept>
#include <string>

#define STRINGIZE(x)        STRINGIZE_DETAIL(x)
#define STRINGIZE_DETAIL(x) #x
#define LINE_STRING         STRINGIZE(__LINE__)

namespace fmt {
  template <typename... Args>
  auto format(const std::string& format, Args... args) -> std::string {
    auto size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
    if (size_s <= 0) {
      throw std::runtime_error("Error during formatting.");
    }
    auto                    size { static_cast<std::size_t>(size_s) };
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1);
  }
} // namespace fmt

#if defined(MPI_ENABLED)
  #include <mpi.h>

  #define NTTLog()                                                             \
    {                                                                          \
      ntt::WaitAndSynchronize(true);                                           \
      int mpi_rank;                                                            \
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);                                \
      if (mpi_rank == 0) {                                                     \
        PLOGV_(ntt::LogFile);                                                  \
      }                                                                        \
    }
  #define NTTLogPrint(msg)                                                     \
    {                                                                          \
      ntt::WaitAndSynchronize(true);                                           \
      int mpi_rank;                                                            \
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);                                \
      if (mpi_rank == 0) {                                                     \
        PLOGV_(ntt::LogFile) << msg;                                           \
      }                                                                        \
    }

#else // not MPI_ENABLED
  #define NTTLog()                                                             \
    { PLOGV_(ntt::LogFile); }
  #define NTTLogPrint(msg)                                                     \
    { PLOGV_(ntt::LogFile) << msg; }

#endif

#define NTTWarn(msg)                                                           \
  { PLOGW_(ntt::LogFile) << msg; }

#define NTTFatal()                                                             \
  { PLOGF_(ntt::LogFile); }

#define NTTHostError(msg)                                                      \
  {                                                                            \
    auto err = fmt::format("# ERROR: %s : filename: %s : line: %s",            \
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