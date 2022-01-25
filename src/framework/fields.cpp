#include "global.h"
#include "fields.h"

#include <plog/Log.h>

#include <vector>

namespace ntt {
  // * * * * * * * * * * * * * * * * * * * *
  // PIC-specific
  // * * * * * * * * * * * * * * * * * * * *
  template <>
  Fields<Dimension::ONE_D, SimulationType::PIC>::Fields(std::vector<unsigned int> res)
    : em {"EM", res[0] + 2 * N_GHOSTS}, cur {"J", res[0] + 2 * N_GHOSTS} {
    PLOGD << "Allocated field arrays.";
  }

  template <>
  Fields<Dimension::TWO_D, SimulationType::PIC>::Fields(std::vector<unsigned int> res)
    : em {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      cur {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS} {
    PLOGD << "Allocated field arrays.";
  }

  template <>
  Fields<Dimension::THREE_D, SimulationType::PIC>::Fields(std::vector<unsigned int> res)
    : em {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      cur {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS} {
    PLOGD << "Allocated field arrays.";
  }

  // * * * * * * * * * * * * * * * * * * * *
  // GRPIC-specific
  // * * * * * * * * * * * * * * * * * * * *
  template <>
  Fields<Dimension::TWO_D, SimulationType::GRPIC>::Fields(std::vector<unsigned int> res)
    : em {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS}, cur {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS} {
    PLOGD << "Allocated field arrays.";
  }

  template <>
  Fields<Dimension::THREE_D, SimulationType::GRPIC>::Fields(std::vector<unsigned int> res)
    : em {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      cur {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS} {
    PLOGD << "Allocated field arrays.";
  }

} // namespace ntt

#if SIMTYPE == PIC_SIMTYPE
template class ntt::Fields<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template class ntt::Fields<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template class ntt::Fields<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;
#elif SIMTYPE == GRPIC_SIMTYPE
template class ntt::Fields<ntt::Dimension::TWO_D, ntt::SimulationType::GRPIC>;
template class ntt::Fields<ntt::Dimension::THREE_D, ntt::SimulationType::GRPIC>;
#endif