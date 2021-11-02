#ifndef OBJECTS_MESHBLOCK_H
#define OBJECTS_MESHBLOCK_H

#include "global.h"
#include "sim_params.h"
#include "particles.h"

#include <vector>
#include <type_traits>
#include <typeinfo>
#include <cmath>
#include <utility>

namespace ntt {

template <Dimension D>
struct Meshblock {
  // sizes of these arrays are ...
  //   resolution + 2 * N_GHOSTS in every direction
  // TESTPERF: maybe use VPIC-style ND array
  RealArrND<D> ex1, ex2, ex3;
  RealArrND<D> bx1, bx2, bx3;
  RealArrND<D> jx1, jx2, jx3;

  std::vector<Particles<D>> particles;

  CoordinateSystem m_coord_system;
  std::vector<real_t> m_extent;
  std::vector<std::size_t> m_resolution;

  Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts);
  ~Meshblock() = default;

  void verify(const SimulationParams&);

  void set_coord_system(const CoordinateSystem& coord_system) { m_coord_system = coord_system; }
  void set_extent(const std::vector<real_t>& extent) { m_extent = extent; }
  [[nodiscard]] auto get_dx1() const -> real_t {
    return (m_extent[1] - m_extent[0]) / static_cast<real_t>(m_resolution[0]);
  }
  [[nodiscard]] auto get_dx2() const -> real_t {
    if constexpr(D == ONE_D) {
      return 0.0;
    } else {
      return (m_extent[3] - m_extent[2]) / static_cast<real_t>(m_resolution[1]);
    }
  }
  [[nodiscard]] auto get_dx3() const -> real_t {
    if constexpr(D != THREE_D) {
      return 0.0;
    } else {
      return (m_extent[5] - m_extent[4]) / static_cast<real_t>(m_resolution[2]);
    }
  }
  [[nodiscard]] auto get_x1min() const -> real_t { return m_extent[0]; }
  [[nodiscard]] auto get_x1max() const -> real_t { return m_extent[1]; }
  [[nodiscard]] auto get_x2min() const -> real_t { return m_extent[2]; }
  [[nodiscard]] auto get_x2max() const -> real_t { return m_extent[3]; }
  [[nodiscard]] auto get_x3min() const -> real_t { return m_extent[4]; }
  [[nodiscard]] auto get_x3max() const -> real_t { return m_extent[5]; }
  [[nodiscard]] auto get_n1() const -> std::size_t { return m_resolution[0]; }
  [[nodiscard]] auto get_n2() const -> std::size_t { return m_resolution[1]; }
  [[nodiscard]] auto get_n3() const -> std::size_t { return m_resolution[2]; }

  [[nodiscard]] auto get_imin() const -> long int { return N_GHOSTS; }
  [[nodiscard]] auto get_imax() const -> long int { return N_GHOSTS + m_resolution[0]; }
  [[nodiscard]] auto get_jmin() const -> long int { return N_GHOSTS; }
  [[nodiscard]] auto get_jmax() const -> long int { return N_GHOSTS + m_resolution[1]; }
  [[nodiscard]] auto get_kmin() const -> long int { return N_GHOSTS; }
  [[nodiscard]] auto get_kmax() const -> long int { return N_GHOSTS + m_resolution[2]; }

  auto loopActiveCells() -> RangeND<D>;
};

template <Dimension D>
KOKKOS_INLINE_FUNCTION auto convert_iTOx1(const Meshblock<D>& mblock, const long int& i) -> real_t {
  return mblock.m_extent[0]
         + (static_cast<real_t>(i - N_GHOSTS) / static_cast<real_t>(mblock.m_resolution[0]))
               * (mblock.m_extent[1] - mblock.m_extent[0]);
}
template <Dimension D>
KOKKOS_INLINE_FUNCTION auto convert_jTOx2(const Meshblock<D>& mblock, const long int& j) -> real_t {
  return mblock.m_extent[2]
         + (static_cast<real_t>(j - N_GHOSTS) / static_cast<real_t>(mblock.m_resolution[1]))
               * (mblock.m_extent[3] - mblock.m_extent[2]);
}
template <Dimension D>
KOKKOS_INLINE_FUNCTION auto convert_kTOx3(const Meshblock<D>& mblock, const long int& k) -> real_t {
  return mblock.m_extent[4]
         + (static_cast<real_t>(k - N_GHOSTS) / static_cast<real_t>(mblock.m_resolution[2]))
               * (mblock.m_extent[5] - mblock.m_extent[4]);
}

template <Dimension D>
KOKKOS_INLINE_FUNCTION auto convert_x1TOidx1(const Meshblock<D>& mblock, const real_t& x1)
    -> std::pair<long int, float> {
  // TESTPERF: floor vs something else
  real_t dx1 {
      (x1 - mblock.m_extent[0])
      / ((mblock.m_extent[1] - mblock.m_extent[0]) / static_cast<real_t>(mblock.m_resolution[0]))};
  long int i {static_cast<long int>(std::floor(dx1))};
  dx1 = dx1 - static_cast<real_t>(i);
  return {i + N_GHOSTS, dx1};
}
template <Dimension D>
KOKKOS_INLINE_FUNCTION auto convert_x2TOjdx2(const Meshblock<D>& mblock, const real_t& x2)
    -> std::pair<long int, float> {
  real_t dx2 {
      (x2 - mblock.m_extent[2])
      / ((mblock.m_extent[3] - mblock.m_extent[2]) / static_cast<real_t>(mblock.m_resolution[1]))};
  long int j {static_cast<long int>(std::floor(dx2))};
  dx2 = dx2 - static_cast<real_t>(j);
  return {j + N_GHOSTS, dx2};
}
template <Dimension D>
KOKKOS_INLINE_FUNCTION auto convert_x3TOkdx3(const Meshblock<D>& mblock, const real_t& x3)
    -> std::pair<long int, float> {
  real_t dx3 {
      (x3 - mblock.m_extent[4])
      / ((mblock.m_extent[5] - mblock.m_extent[4]) / static_cast<real_t>(mblock.m_resolution[2]))};
  long int k {static_cast<long int>(std::floor(dx3))};
  dx3 = dx3 - static_cast<real_t>(k);
  return {k + N_GHOSTS, dx3};
}

} // namespace ntt

#endif
