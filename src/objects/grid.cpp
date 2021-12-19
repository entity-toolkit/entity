#include "global.h"
#include "grid.h"

#include <vector>

namespace ntt {

  template <Dimension D>
  Grid<D>::Grid(std::vector<real_t> ext, std::vector<std::size_t> res) : m_extent {std::move(ext)}, m_resolution {std::move(res)} {}

  // # # # # # # # # # # # # # # # #
  // custom range
  template <>
  auto Grid<ONE_D>::loopCells(const long int& i1, const long int& i2) -> ntt_1drange_t {
    return NTT1DRange(static_cast<range_t>(N_GHOSTS + i1), static_cast<range_t>(get_imax() + i2));
  }
  template <>
  auto Grid<TWO_D>::loopCells(const long int& i1, const long int& i2, const long int& j1, const long int& j2) -> ntt_2drange_t {
    return NTT2DRange({N_GHOSTS + i1, N_GHOSTS + j1}, {get_imax() + i2, get_jmax() + j2});
  }
  template <>
  auto Grid<THREE_D>::loopCells(const long int& i1, const long int& i2, const long int& j1, const long int& j2, const long int& k1, const long int& k2) -> ntt_3drange_t {
    return NTT3DRange({N_GHOSTS + i1, N_GHOSTS + j1, N_GHOSTS + k1},
                      {get_imax() + i2, get_jmax() + j2, get_kmax() + k2});
  }

  // # # # # # # # # # # # # # # # #
  template <>
  auto Grid<ONE_D>::loopActiveCells() -> ntt_1drange_t {
    return loopCells(0, 0);
  }
  template <>
  auto Grid<TWO_D>::loopActiveCells() -> ntt_2drange_t {
    return loopCells(0, 0, 0, 0);
  }
  template <>
  auto Grid<THREE_D>::loopActiveCells() -> ntt_3drange_t {
    return loopCells(0, 0, 0, 0, 0, 0);
  }

  // # # # # # # # # # # # # # # # #
  template <>
  auto Grid<ONE_D>::loopAllCells() -> ntt_1drange_t {
    return loopCells(-N_GHOSTS, N_GHOSTS);
  }
  template <>
  auto Grid<TWO_D>::loopAllCells() -> ntt_2drange_t {
    return loopCells(-N_GHOSTS, N_GHOSTS, -N_GHOSTS, N_GHOSTS);
  }
  template <>
  auto Grid<THREE_D>::loopAllCells() -> ntt_3drange_t {
    return loopCells(-N_GHOSTS, N_GHOSTS, -N_GHOSTS, N_GHOSTS, -N_GHOSTS, N_GHOSTS);
  }

  // // # # # # # # # # # # # # # # # #
  // template <>
  // auto Grid<ONE_D>::loopX1MinCells() -> ntt_1drange_t {
  //   return NTT1DRange(0, static_cast<range_t>(get_imin() + 1));
  // }
  // template <>
  // auto Grid<TWO_D>::loopX1MinCells() -> ntt_2drange_t {
  //   return NTT2DRange({0, 0}, {get_imin() + 1, get_jmax() + N_GHOSTS});
  // }
  // template <>
  // auto Grid<THREE_D>::loopX1MinCells() -> ntt_3drange_t {
  //   return NTT3DRange({0, 0, 0}, {get_imin() + 1, get_jmax() + N_GHOSTS, get_kmax() + N_GHOSTS});
  // }

  // // # # # # # # # # # # # # # # # #
  // template <>
  // auto Grid<ONE_D>::loopX1MaxCells() -> ntt_1drange_t {
  //   return NTT1DRange(static_cast<range_t>(get_imax()), static_cast<range_t>(get_imax() + N_GHOSTS));
  // }
  // template <>
  // auto Grid<TWO_D>::loopX1MaxCells() -> ntt_2drange_t {
  //   return NTT2DRange({get_imax(), 0}, {get_imax() + N_GHOSTS, get_jmax() + N_GHOSTS});
  // }
  // template <>
  // auto Grid<THREE_D>::loopX1MaxCells() -> ntt_3drange_t {
  //   return NTT3DRange({get_imax(), 0, 0}, {get_imax() + N_GHOSTS, get_jmax() + N_GHOSTS, get_kmax() + N_GHOSTS});
  // }

} // namespace ntt

template struct ntt::Grid<ntt::ONE_D>;
template struct ntt::Grid<ntt::TWO_D>;
template struct ntt::Grid<ntt::THREE_D>;
