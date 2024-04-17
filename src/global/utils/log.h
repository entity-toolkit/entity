/**
 * @file utils/log.h
 * @brief Global logging utilities with proper fences and MPI blockings
 * @implements
 *   - macro HERE
 *   - logger::Checkpoint -> void
 *   - info::Print -> void
 * @depends:
 *   - global.h
 *   - arch/mpi_aliases.h
 *   - utils/formatting.h
 * @namespaces:
 *   - logger::
 *   - info::
 * @macros:
 *   - MPI_ENABLED
 *   - DEBUG
 */

#ifndef GLOBAL_LOG_H
#define GLOBAL_LOG_H

#include "global.h"

#include "arch/mpi_aliases.h"
#include "utils/formatting.h"

#include <Kokkos_Core.hpp>
#include <plog/Log.h>

#include <iostream>
#include <string>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

namespace logger {
  using namespace files;

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

namespace info {
  using namespace files;

  inline void Print(const std::string& msg, bool stdout = true, bool once = true) {
    auto msg_nocol = color::strip(msg);
    if (once) {
      CallOnce(
        [](const std::string& msg, const std::string& msg_nocol, bool stdout) {
          PLOGN_(InfoFile) << msg_nocol;
          if (stdout) {
            std::cout << msg << std::endl;
          }
        },
        msg,
        msg_nocol,
        stdout);
    } else {
      PLOGN_(InfoFile) << msg_nocol;
      if (stdout) {
        std::cout << msg << std::endl;
      }
    }
  }

} // namespace info

#endif // GLOBAL_LOG_H