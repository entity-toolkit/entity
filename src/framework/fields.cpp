#include "global.h"
#include "fields.h"

#include <plog/Log.h>

#include <vector>

namespace ntt {
  using resolution_t = std::vector<unsigned int>;

#if SIMTYPE == PIC_SIMTYPE
  // * * * * * * * * * * * * * * * * * * * *
  // PIC-specific
  // * * * * * * * * * * * * * * * * * * * *
  template <>
  Fields<Dim1, TypePIC>::Fields(resolution_t res)
    : em {"EM", res[0] + 2 * N_GHOSTS}, cur {"J", res[0] + 2 * N_GHOSTS} {
    PLOGD << "Allocated field arrays.";
  }

  template <>
  Fields<Dim2, TypePIC>::Fields(resolution_t res)
    : em {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      cur {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS} {
    PLOGD << "Allocated field arrays.";
  }

  template <>
  Fields<Dim3, TypePIC>::Fields(resolution_t res)
    : em {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      cur {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS} {
    PLOGD << "Allocated field arrays.";
  }
#elif SIMTYPE == GRPIC_SIMTYPE
  // * * * * * * * * * * * * * * * * * * * *
  // GRPIC-specific
  // * * * * * * * * * * * * * * * * * * * *
  template <>
  Fields<Dim2, TypeGRPIC>::Fields(resolution_t res)
    : em {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      cur {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      aux {"AUX", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      em0 {"EM0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      cur0 {"J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      aphi {"APHI", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS} {
    PLOGD << "Allocated field arrays.";
  }

  template <>
  Fields<Dim3, TypeGRPIC>::Fields(resolution_t res)
    : em {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      cur {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      aux {"AUX", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      em0 {"EM0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      cur0 {"J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      aphi {"APHI", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS} {
    PLOGD << "Allocated field arrays.";
  }
#endif

} // namespace ntt

#if SIMTYPE == PIC_SIMTYPE
template class ntt::Fields<ntt::Dim1, ntt::SimulationType::PIC>;
template class ntt::Fields<ntt::Dim2, ntt::SimulationType::PIC>;
template class ntt::Fields<ntt::Dim3, ntt::SimulationType::PIC>;
#elif SIMTYPE == GRPIC_SIMTYPE
template class ntt::Fields<ntt::Dim2, ntt::SimulationType::GRPIC>;
template class ntt::Fields<ntt::Dim3, ntt::SimulationType::GRPIC>;
#endif