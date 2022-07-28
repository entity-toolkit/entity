#ifndef PIC_RESET_H
#define PIC_RESET_H

#include "global.h"
#include "pic.h"

namespace ntt {
  /**
   * @brief Reset all particles of a particular species.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class ResetParticles {
    Meshblock<D, SimulationType::PIC> m_mblock;
    Particles<D, SimulationType::PIC> m_particles;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param particles Particles.
     */
    ResetParticles(const Meshblock<D, SimulationType::PIC>& mblock,
                   const Particles<D, SimulationType::PIC>& particles)
      : m_mblock(mblock), m_particles(particles) {}
    
    /**
     * @brief Calling the loop over all particles.
     */
    void resetParticles() {
      auto range_policy = Kokkos::RangePolicy<AccelExeSpace>(0, m_particles.npart());
      Kokkos::parallel_for("reset_particles", range_policy, *this);
    }
    /**
     * @brief Loop iteration.
     * @param p index 
     */
    Inline void operator()(index_t p) const {
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
   * @brief Reset all the currents to zero.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class ResetCurrents {
    Meshblock<D, SimulationType::PIC> m_mblock;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     */
    ResetCurrents(const Meshblock<D, SimulationType::PIC>& mblock) : m_mblock {mblock} {}
    /**
     * @brief 1D implementation of the algorithm.
     * @param i1 index.
     */
    Inline void operator()(index_t) const;
    /**
     * @brief 2D implementation of the algorithm.
     * @param i1 index.
     * @param i2 index.
     */
    Inline void operator()(index_t, index_t) const;
    /**
     * @brief 3D implementation of the algorithm.
     * @param i1 index.
     * @param i2 index.
     * @param i3 index.
     */
    Inline void operator()(index_t, index_t, index_t) const;
  };

  template <>
  Inline void ResetCurrents<Dimension::ONE_D>::operator()(index_t i) const {
    m_mblock.cur(i, cur::jx1) = ZERO;
    m_mblock.cur(i, cur::jx2) = ZERO;
    m_mblock.cur(i, cur::jx3) = ZERO;
  }

  template <>
  Inline void ResetCurrents<Dimension::TWO_D>::operator()(index_t i,
                                                          index_t j) const {
    m_mblock.cur(i, j, cur::jx1) = ZERO;
    m_mblock.cur(i, j, cur::jx2) = ZERO;
    m_mblock.cur(i, j, cur::jx3) = ZERO;
  }

  template <>
  Inline void ResetCurrents<Dimension::THREE_D>::operator()(index_t i,
                                                            index_t j,
                                                            index_t k) const {
    m_mblock.cur(i, j, k, cur::jx1) = ZERO;
    m_mblock.cur(i, j, k, cur::jx2) = ZERO;
    m_mblock.cur(i, j, k, cur::jx3) = ZERO;
  }

  /**
   * @brief Reset all the fields to zero.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class ResetFields {
    Meshblock<D, SimulationType::PIC> m_mblock;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     */
    ResetFields(const Meshblock<D, SimulationType::PIC>& mblock) : m_mblock {mblock} {}
    /**
     * @brief 1D implementation of the algorithm.
     * @param i1 index.
     */
    Inline void operator()(index_t) const;
    /**
     * @brief 2D implementation of the algorithm.
     * @param i1 index.
     * @param i2 index.
     */
    Inline void operator()(index_t, index_t) const;
    /**
     * @brief 3D implementation of the algorithm.
     * @param i1 index.
     * @param i2 index.
     * @param i3 index.
     */
    Inline void operator()(index_t, index_t, index_t) const;
  };

  template <>
  Inline void ResetFields<Dimension::ONE_D>::operator()(index_t i) const {
    m_mblock.em(i, em::ex1) = ZERO;
    m_mblock.em(i, em::ex2) = ZERO;
    m_mblock.em(i, em::ex3) = ZERO;
    m_mblock.em(i, em::bx1) = ZERO;
    m_mblock.em(i, em::bx2) = ZERO;
    m_mblock.em(i, em::bx3) = ZERO;
  }

  template <>
  Inline void ResetFields<Dimension::TWO_D>::operator()(index_t i,
                                                        index_t j) const {
    m_mblock.em(i, j, em::ex1) = ZERO;
    m_mblock.em(i, j, em::ex2) = ZERO;
    m_mblock.em(i, j, em::ex3) = ZERO;
    m_mblock.em(i, j, em::bx1) = ZERO;
    m_mblock.em(i, j, em::bx2) = ZERO;
    m_mblock.em(i, j, em::bx3) = ZERO;
  }

  template <>
  Inline void ResetFields<Dimension::THREE_D>::operator()(index_t i,
                                                          index_t j,
                                                          index_t k) const {
    m_mblock.em(i, j, k, em::ex1) = ZERO;
    m_mblock.em(i, j, k, em::ex2) = ZERO;
    m_mblock.em(i, j, k, em::ex3) = ZERO;
    m_mblock.em(i, j, k, em::bx1) = ZERO;
    m_mblock.em(i, j, k, em::bx2) = ZERO;
    m_mblock.em(i, j, k, em::bx3) = ZERO;
  }
} // namespace ntt

#endif
