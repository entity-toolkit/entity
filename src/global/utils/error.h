/**
 * @file utils/error.h
 * @brief Error handling utilities
 * @implements
 *   - raise::Error -> void
 *   - raise::Fatal -> void
 *   - raise::ErrorIf -> void
 *   - raise::KernelError<> -> void
 *   - raise::KernelNotImplementedError -> void
 * @namespaces:
 *   - raise::
 * @macros:
 *   - MPI_ENABLED
 * !TODO:
 *   - migrate to Kokkos::printf (4.2)
 *   - replace KernelErrors with static asserts
 */

#ifndef GLOBAL_UTILS_ERROR_H
#define GLOBAL_UTILS_ERROR_H

#include "arch/kokkos_aliases.h"

#include <Kokkos_Core.hpp>
#include <plog/Log.h>

#include <cstdio>
#include <string>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

namespace raise {
  using namespace files;

  [[noreturn]]
  inline void Error(const std::string& msg,
                    const std::string& file,
                    const std::string& func,
                    int                line) {
    PLOGE_(ErrFile) << "Error: " << file << ":" << line << " @ " << func;
#if defined(MPI_ENABLED)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PLOGE_(ErrFile) << ": rank : " << rank;
#endif
    PLOGE_(ErrFile) << ": message : " << msg;
    PLOGE << "";
    PLOGE << msg;
    PLOGE << "see the `*.err` file for more details";
#if defined(MPI_ENABLED)
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
#endif
    throw std::logic_error(
      (msg + " " + file + ":" + std::to_string(line) + " @ " + func).c_str());
  }

  inline void Fatal(const std::string& msg,
                    const std::string& file,
                    const std::string& func,
                    int                line) {
    PLOGF_(ErrFile) << "Fatal: " << file << ":" << line << " @ " << func;
#if defined(MPI_ENABLED)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PLOGF_(ErrFile) << ": rank : " << rank;
#endif
    PLOGF_(ErrFile) << ": message : " << msg;
    PLOGF << "";
    PLOGF << msg;
    PLOGF << "see the `*.err` file for more details";
#if defined(MPI_ENABLED)
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
#endif
    throw std::runtime_error(
      (msg + " " + file + ":" + std::to_string(line) + " @ " + func).c_str());
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

  Inline void KernelError(const char* file,
                          const char* func,
                          int         line,
                          const char* msg) {
    printf("\n%s:%d @ %s\nError: %s", file, line, func, msg);
    Kokkos::abort("kernel error");
  }

  Inline void KernelNotImplementedError(const char* file, const char* func, int line) {
    printf("\n%s:%d @ %s\n", file, line, func);
    Kokkos::abort("kernel not implemented");
  }

} // namespace raise

#endif // GLOBAL_UTILS_ERROR_H
