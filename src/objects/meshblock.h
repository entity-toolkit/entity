#ifndef OBJECTS_MESHBLOCK_H
#define OBJECTS_MESHBLOCK_H

#include "global.h"
#include "particles.h"

#include <vector>

namespace ntt {

template <template <typename T = std::nullptr_t> class D>
struct Meshblock {
  // sizes of these arrays is ...
  //   resolution + 2 * N_GHOSTS in every direction
  NTTArray<typename D<real_t>::ndtype_t> ex1;
  NTTArray<typename D<real_t>::ndtype_t> ex2;
  NTTArray<typename D<real_t>::ndtype_t> ex3;
  NTTArray<typename D<real_t>::ndtype_t> bx1;
  NTTArray<typename D<real_t>::ndtype_t> bx2;
  NTTArray<typename D<real_t>::ndtype_t> bx3;
  NTTArray<typename D<real_t>::ndtype_t> jx1;
  NTTArray<typename D<real_t>::ndtype_t> jx2;
  NTTArray<typename D<real_t>::ndtype_t> jx3;

  std::vector<Particles<D>> particles;

  CoordinateSystem m_coord_system;
  std::vector<real_t> m_extent;
  std::vector<std::size_t> m_resolution;

  Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>&);
  ~Meshblock() = default;

  void set_coord_system(const CoordinateSystem& coord_system) { m_coord_system = coord_system; }
  void set_extent(const std::vector<real_t>& extent) { m_extent = extent; }
  [[nodiscard]] auto get_dx1() const -> real_t {
    return (m_extent[1] - m_extent[0]) / static_cast<real_t>(m_resolution[0]);
  }
  [[nodiscard]] auto get_dx2() const -> real_t {
    return (m_extent[3] - m_extent[2]) / static_cast<real_t>(m_resolution[1]);
  }
  [[nodiscard]] auto get_dx3() const -> real_t {
    return (m_extent[5] - m_extent[4]) / static_cast<real_t>(m_resolution[2]);
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
};

auto loopActiveCells(const Meshblock<One_D>&) -> ntt_1drange_t;
auto loopActiveCells(const Meshblock<Two_D>&) -> ntt_2drange_t;
auto loopActiveCells(const Meshblock<Three_D>&) -> ntt_3drange_t;

template <template <typename T> class D>
KOKKOS_INLINE_FUNCTION auto convert_iTOx1(const Meshblock<D>& mblock, const long int& i) -> real_t {
  return mblock.m_extent[0]
       + (static_cast<real_t>(i - N_GHOSTS) / static_cast<real_t>(mblock.m_resolution[0]))
             * (mblock.m_extent[1] - mblock.m_extent[0]);
}
template <template <typename T> class D>
KOKKOS_INLINE_FUNCTION auto convert_jTOx2(const Meshblock<D>& mblock, const long int& j) -> real_t {
  return mblock.m_extent[2]
       + (static_cast<real_t>(j - N_GHOSTS) / static_cast<real_t>(mblock.m_resolution[1]))
             * (mblock.m_extent[3] - mblock.m_extent[2]);
}
template <template <typename T> class D>
KOKKOS_INLINE_FUNCTION auto convert_kTOx3(const Meshblock<D>& mblock, const long int& k) -> real_t {
  return mblock.m_extent[4]
       + (static_cast<real_t>(k - N_GHOSTS) / static_cast<real_t>(mblock.m_resolution[2]))
             * (mblock.m_extent[5] - mblock.m_extent[4]);
}

} // namespace ntt

#endif
