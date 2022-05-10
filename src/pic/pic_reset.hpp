#ifndef PIC_RESET_H
#define PIC_RESET_H

#include "global.h"
#include "pic.h"

namespace ntt {
  /**
   * Reset all particles of a particular species.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class ResetParticles {
    using index_t = const std::size_t;
    Meshblock<D, SimulationType::PIC> m_mblock;
    Particles<D, SimulationType::PIC> m_particles;

  public:
    ResetParticles(const Meshblock<D, SimulationType::PIC>& mblock,
                   const Particles<D, SimulationType::PIC>& particles)
      : m_mblock(mblock), m_particles(particles) {}

    void resetParticles() {
      auto range_policy = Kokkos::RangePolicy<AccelExeSpace>(0, m_particles.npart());
      Kokkos::parallel_for("reset_particles", range_policy, *this);
    }
    Inline void operator()(const index_t p) const {
      if constexpr ((D == Dimension::ONE_D) || (D == Dimension::TWO_D)
                    || (D == Dimension::THREE_D)) {
        m_particles.i1(p)  = 0;
        m_particles.dx1(p) = 0.0f;
      }
      if constexpr ((D == Dimension::TWO_D) || (D == Dimension::THREE_D)) {
        m_particles.i2(p)  = 0;
        m_particles.dx2(p) = 0.0f;
      }
      if constexpr (D == Dimension::THREE_D) {
        m_particles.i3(p)  = 0;
        m_particles.dx3(p) = 0.0f;
      }
      m_particles.ux1(p) = ZERO;
      m_particles.ux2(p) = ZERO;
      m_particles.ux3(p) = ZERO;
    }
  };

  /**
   * Reset all the currents to zero.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class ResetCurrents {
    using index_t = typename RealFieldND<D, 3>::size_type;
    Meshblock<D, SimulationType::PIC> m_mblock;

  public:
    ResetCurrents(const Meshblock<D, SimulationType::PIC>& mblock) : m_mblock {mblock} {}
    Inline void operator()(const index_t) const;
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void ResetCurrents<Dimension::ONE_D>::operator()(const index_t i) const {
    m_mblock.cur(i, cur::jx1) = ZERO;
    m_mblock.cur(i, cur::jx2) = ZERO;
    m_mblock.cur(i, cur::jx3) = ZERO;
  }

  template <>
  Inline void ResetCurrents<Dimension::TWO_D>::operator()(const index_t i,
                                                          const index_t j) const {
    m_mblock.cur(i, j, cur::jx1) = ZERO;
    m_mblock.cur(i, j, cur::jx2) = ZERO;
    m_mblock.cur(i, j, cur::jx3) = ZERO;
  }

  template <>
  Inline void ResetCurrents<Dimension::THREE_D>::operator()(const index_t i,
                                                            const index_t j,
                                                            const index_t k) const {
    m_mblock.cur(i, j, k, cur::jx1) = ZERO;
    m_mblock.cur(i, j, k, cur::jx2) = ZERO;
    m_mblock.cur(i, j, k, cur::jx3) = ZERO;
  }

  /**
   * Reset all the fields to zero.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class ResetFields {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::PIC> m_mblock;

  public:
    ResetFields(const Meshblock<D, SimulationType::PIC>& mblock) : m_mblock {mblock} {}
    Inline void operator()(const index_t) const;
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void ResetFields<Dimension::ONE_D>::operator()(const index_t i) const {
    m_mblock.em(i, em::ex1) = ZERO;
    m_mblock.em(i, em::ex2) = ZERO;
    m_mblock.em(i, em::ex3) = ZERO;
    m_mblock.em(i, em::bx1) = ZERO;
    m_mblock.em(i, em::bx2) = ZERO;
    m_mblock.em(i, em::bx3) = ZERO;
  }

  template <>
  Inline void ResetFields<Dimension::TWO_D>::operator()(const index_t i,
                                                        const index_t j) const {
    m_mblock.em(i, j, em::ex1) = ZERO;
    m_mblock.em(i, j, em::ex2) = ZERO;
    m_mblock.em(i, j, em::ex3) = ZERO;
    m_mblock.em(i, j, em::bx1) = ZERO;
    m_mblock.em(i, j, em::bx2) = ZERO;
    m_mblock.em(i, j, em::bx3) = ZERO;
  }

  template <>
  Inline void ResetFields<Dimension::THREE_D>::operator()(const index_t i,
                                                          const index_t j,
                                                          const index_t k) const {
    m_mblock.em(i, j, k, em::ex1) = ZERO;
    m_mblock.em(i, j, k, em::ex2) = ZERO;
    m_mblock.em(i, j, k, em::ex3) = ZERO;
    m_mblock.em(i, j, k, em::bx1) = ZERO;
    m_mblock.em(i, j, k, em::bx2) = ZERO;
    m_mblock.em(i, j, k, em::bx3) = ZERO;
  }
} // namespace ntt

#endif
