/**
 * @file utils/log.h
 * @brief Global logging utilities with proper fences and MPI blockings
 * @implements
 *   - HERE
 *   - logger::Checkpoint
 * @depends:
 *   - arch/mpi_aliases.h
 * @namespaces:
 *   - logger::
 * @macros:
 *   - MPI_ENABLED
 *   - DEBUG
 */

#ifndef GLOBAL_LOG_H
#define GLOBAL_LOG_H

#include "arch/mpi_aliases.h"

#include <Kokkos_Core.hpp>
#include <plog/Log.h>

#include <string>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#define HERE __FILE__, __func__, __LINE__

enum {
  LogFile = 1,
  ErrFile,
  InfoFile
};

namespace logger {

  inline void Checkpoint(const std::string& file, const std::string& func, int line) {
#if defined(DEBUG)
    Kokkos::fence();
  #if defined(MPI_ENABLED)
    MPI_Barrier(MPI_COMM_WORLD);
  #endif
#endif
    CallOnce(
      [](const std::string& file, const std::string& func, int line) {
        PLOGV_(LogFile) << "Checkpoint: " << file << " : " << func << " @ " << line;
      },
      file,
      func,
      line);
  }

  inline void Checkpoint(const std::string& msg,
                         const std::string& file,
                         const std::string& func,
                         int                line) {
#if defined(DEBUG)
    Kokkos::fence();
  #if defined(MPI_ENABLED)
    MPI_Barrier(MPI_COMM_WORLD);
  #endif
#endif
    CallOnce(
      [](const std::string& msg,
         const std::string& file,
         const std::string& func,
         int                line) {
        PLOGV_(LogFile) << "Checkpoint: " << file << " : " << func << " @ " << line;
        PLOGV_(LogFile) << " : message : " << msg;
      },
      msg,
      file,
      func,
      line);
  }

} // namespace logger

#endif // GLOBAL_LOG_H