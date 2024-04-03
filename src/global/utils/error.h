/**
 * @file utils/error.h
 * @brief Error handling utilities
 * @implements
 *   - raise::Warning -> void
 *   - raise::Error -> void
 *   - raise::Fatal -> void
 *   - raise::ErrorIf -> void
 *   - raise::KernelError<> -> void
 *   - raise::KernelNotImplementedError -> void
 * @depends:
 *   - arch/kokkos_aliases.h
 *   - utils/formatting.h
 *   - utils/log.h
 * @namespaces:
 *   - raise::
 * @macros:
 *   - MPI_ENABLED
 * !TODO:
 *   - migrate to Kokkos::printf (4.2)
 */

#ifndef GLOBAL_UTILS_ERROR_H
#define GLOBAL_UTILS_ERROR_H

#include "arch/kokkos_aliases.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include <Kokkos_Core.hpp>
#include <plog/Log.h>

#include <cstdio>
#include <string>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

namespace raise {

  inline void Warning(const std::string& msg,
                      const std::string& file,
                      const std::string& func,
                      int                line) {
    PLOGW_(ErrFile) << "Warning: " << file << " : " << func << " @ " << line;
#if defined(MPI_ENABLED)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PLOGW_(ErrFile) << ": rank : " << rank;
#endif
    PLOGW_(ErrFile) << ": message : " << msg;
  }

  inline void Error(const std::string& msg,
                    const std::string& file,
                    const std::string& func,
                    int                line) {
    PLOGE_(ErrFile) << "Error: " << file << " : " << func << " @ " << line;
#if defined(MPI_ENABLED)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PLOGE_(ErrFile) << ": rank : " << rank;
#endif
    PLOGE_(ErrFile) << ": message : " << msg;
#if defined(MPI_ENABLED)
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
#endif
    throw std::logic_error(msg.c_str());
  }

  inline void Fatal(const std::string& msg,
                    const std::string& file,
                    const std::string& func,
                    int                line) {
    PLOGF_(ErrFile) << "Fatal: " << file << " : " << func << " @ " << line;
#if defined(MPI_ENABLED)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PLOGF_(ErrFile) << ": rank : " << rank;
#endif
    PLOGF_(ErrFile) << ": message : " << msg;
#if defined(MPI_ENABLED)
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
#endif
    throw std::runtime_error(msg.c_str());
  }

  inline void ErrorIf(bool               condition,
                      const std::string& msg,
                      const std::string& file,
                      const std::string& func,
                      int                line) {
    if (condition) {
      Error(msg, file, func, line);
    }
  }

  inline void FatalIf(bool               condition,
                      const std::string& msg,
                      const std::string& file,
                      const std::string& func,
                      int                line) {
    if (condition) {
      Fatal(msg, file, func, line);
    }
  }

  template <typename... Args>
  Inline void KernelError(const char* file,
                          const char* func,
                          int         line,
                          const char* fmt,
                          Args... args) {
    printf("\n%s : %s @ %d\n", file, func, line);
    printf(fmt, args...);
    Kokkos::abort("kernel error");
  }

  Inline void KernelNotImplementedError(const char* file, const char* func, int line) {
    printf("\n%s : %s @ %d\n", file, func, line);
    Kokkos::abort("kernel not implemented");
  }

} // namespace raise

#endif // GLOBAL_UTILS_ERROR_H