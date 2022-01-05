#include "global.h"
#include "fields.h"

#include <vector>

namespace ntt {

  template <>
  Fields<ONE_D>::Fields(std::vector<std::size_t> res)
      : em_fields {"EM", res[0] + 2 * N_GHOSTS},
        j_fields {"J", res[0] + 2 * N_GHOSTS},
        i_min {N_GHOSTS},
        i_max {N_GHOSTS + res[0]},
        j_min {0},
        j_max {1},
        k_min {0},
        k_max {1},
        Ni {i_max - i_min},
        Nj {j_max - j_min},
        Nk {k_max - k_min} {}

  template <>
  Fields<TWO_D>::Fields(std::vector<std::size_t> res)
      : em_fields {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
        j_fields {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
        i_min {N_GHOSTS},
        i_max {N_GHOSTS + res[0]},
        j_min {N_GHOSTS},
        j_max {N_GHOSTS + res[1]},
        k_min {0},
        k_max {1},
        Ni {i_max - i_min},
        Nj {j_max - j_min},
        Nk {k_max - k_min} {}

  template <>
  Fields<THREE_D>::Fields(std::vector<std::size_t> res)
      : em_fields {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
        j_fields {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
        i_min {N_GHOSTS},
        i_max {N_GHOSTS + res[0]},
        j_min {N_GHOSTS},
        j_max {N_GHOSTS + res[1]},
        k_min {N_GHOSTS},
        k_max {N_GHOSTS + res[2]},
        Ni {i_max - i_min},
        Nj {j_max - j_min},
        Nk {k_max - k_min} {}

  // # # # # # # # # # # # # # # # #
  // custom range
  template <>
  auto Fields<ONE_D>::loopCells(const long int& i1, const long int& i2) -> ntt_1drange_t {
    return NTT1DRange((range_t)(N_GHOSTS + i1), (range_t)(i_max + i2));
  }
  template <>
  auto Fields<TWO_D>::loopCells(const long int& i1, const long int& i2, const long int& j1, const long int& j2)
      -> ntt_2drange_t {
    return NTT2DRange({N_GHOSTS + i1, N_GHOSTS + j1}, {i_max + i2, j_max + j2});
  }
  template <>
  auto Fields<THREE_D>::loopCells(const long int& i1,
                                  const long int& i2,
                                  const long int& j1,
                                  const long int& j2,
                                  const long int& k1,
                                  const long int& k2) -> ntt_3drange_t {
    return NTT3DRange({N_GHOSTS + i1, N_GHOSTS + j1, N_GHOSTS + k1},
                      {i_max + i2, j_max + j2, k_max + k2});
  }

  // # # # # # # # # # # # # # # # #
  template <>
  auto Fields<ONE_D>::loopActiveCells() -> ntt_1drange_t {
    return loopCells(0, 0);
  }
  template <>
  auto Fields<TWO_D>::loopActiveCells() -> ntt_2drange_t {
    return loopCells(0, 0, 0, 0);
  }
  template <>
  auto Fields<THREE_D>::loopActiveCells() -> ntt_3drange_t {
    return loopCells(0, 0, 0, 0, 0, 0);
  }

  // # # # # # # # # # # # # # # # #
  template <>
  auto Fields<ONE_D>::loopAllCells() -> ntt_1drange_t {
    return loopCells(-N_GHOSTS, N_GHOSTS);
  }
  template <>
  auto Fields<TWO_D>::loopAllCells() -> ntt_2drange_t {
    return loopCells(-N_GHOSTS, N_GHOSTS, -N_GHOSTS, N_GHOSTS);
  }
  template <>
  auto Fields<THREE_D>::loopAllCells() -> ntt_3drange_t {
    return loopCells(-N_GHOSTS, N_GHOSTS, -N_GHOSTS, N_GHOSTS, -N_GHOSTS, N_GHOSTS);
  }

} // namespace ntt

template struct ntt::Fields<ntt::ONE_D>;
template struct ntt::Fields<ntt::TWO_D>;
template struct ntt::Fields<ntt::THREE_D>;
