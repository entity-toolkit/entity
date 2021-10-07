#ifndef OBJECTS_MESHBLOCK_H
#define OBJECTS_MESHBLOCK_H

#include "global.h"
#include "sim_params.h"
#include "particles.h"

#include <vector>

namespace ntt {

template <Dimension D>
struct Meshblock {
  std::vector<Particles<D>> particles;

  CoordinateSystem m_coord_system;
  std::vector<real_t> m_extent;
  std::vector<std::size_t> m_resolution;

  Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts);
  ~Meshblock() = default;

  virtual void verify(const SimulationParams&) {}
  void printDetails();

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

// sizes of these arrays is ...
//   resolution + 2 * N_GHOSTS in every direction
struct Meshblock1D : public Meshblock<ONE_D> {
  NTTArray<real_t*> ex1, ex2, ex3;
  NTTArray<real_t*> bx1, bx2, bx3;
  NTTArray<real_t*> jx1, jx2, jx3;
  Meshblock1D(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts);
  void verify(const SimulationParams& sim_params) override;
};
struct Meshblock2D : public Meshblock<TWO_D> {
  NTTArray<real_t**> ex1, ex2, ex3;
  NTTArray<real_t**> bx1, bx2, bx3;
  NTTArray<real_t**> jx1, jx2, jx3;
  Meshblock2D(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts);
  void verify(const SimulationParams& sim_params) override;
};
struct Meshblock3D : public Meshblock<THREE_D> {
  NTTArray<real_t***> ex1, ex2, ex3;
  NTTArray<real_t***> bx1, bx2, bx3;
  NTTArray<real_t***> jx1, jx2, jx3;

public:
  Meshblock3D(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts);
  void verify(const SimulationParams& sim_params) override;
};

auto loopActiveCells(const Meshblock1D&) -> ntt_1drange_t;
auto loopActiveCells(const Meshblock2D&) -> ntt_2drange_t;
auto loopActiveCells(const Meshblock3D&) -> ntt_3drange_t;

} // namespace ntt

#endif
