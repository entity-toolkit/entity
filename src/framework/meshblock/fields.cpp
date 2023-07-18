#include "wrapper.h"

#include "io/output.h"

#include <plog/Log.h>

#include <vector>

namespace ntt {

  using resolution_t = std::vector<unsigned int>;

  // * * * * * * * * * * * * * * * * * * * *
  // PIC-specific
  // * * * * * * * * * * * * * * * * * * * *
  template <>
  Fields<Dim1, PICEngine>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS },
      bckp { "BCKP", res[0] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS } {
    NTTLog();
  }

  template <>
  Fields<Dim2, PICEngine>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      bckp { "BCKP", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS } {
    NTTLog();
  }

  template <>
  Fields<Dim3, PICEngine>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      bckp { "BCKP", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS } {
    NTTLog();
  }

  // * * * * * * * * * * * * * * * * * * * *
  // GRPIC-specific
  // * * * * * * * * * * * * * * * * * * * *
  template <>
  Fields<Dim2, GRPICEngine>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      bckp { "BCKP", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      aux { "AUX", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      em0 { "EM0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      cur0 { "CUR0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS } {
    NTTLog();
  }

  template <>
  Fields<Dim3, GRPICEngine>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      bckp { "BCKP", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      aux { "AUX", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      em0 { "EM0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      cur0 { "CUR0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS } {
    NTTLog();
  }

  template <>
  Fields<Dim1, SANDBOXEngine>::Fields(resolution_t) {
    NTTLog();
  }
  template <>
  Fields<Dim2, SANDBOXEngine>::Fields(resolution_t) {
    NTTLog();
  }
  template <>
  Fields<Dim3, SANDBOXEngine>::Fields(resolution_t) {
    NTTLog();
  }
}    // namespace ntt