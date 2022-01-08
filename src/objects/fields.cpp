#include "global.h"
#include "fields.h"

#include <vector>

namespace ntt {

  template <>
  Fields<Dimension::ONE_D, SimulationType::PIC>::Fields(std::vector<std::size_t> res)
    : em {"EM", res[0] + 2 * N_GHOSTS}, cur {"J", res[0] + 2 * N_GHOSTS} {}

  template <>
  Fields<Dimension::TWO_D, SimulationType::PIC>::Fields(std::vector<std::size_t> res)
    : em {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      cur {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS} {}

  template <>
  Fields<Dimension::THREE_D, SimulationType::PIC>::Fields(std::vector<std::size_t> res)
    : em {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      cur {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS} {}

} // namespace ntt

template class ntt::Fields<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template class ntt::Fields<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template class ntt::Fields<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;
