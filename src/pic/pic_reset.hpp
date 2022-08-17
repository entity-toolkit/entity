#ifndef PIC_RESET_H
#define PIC_RESET_H

#include "global.h"
#include "pic.h"

#include "field_macros.h"

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
    JX1(i) = ZERO;
    JX2(i) = ZERO;
    JX3(i) = ZERO;
  }

  template <>
  Inline void ResetCurrents<Dimension::TWO_D>::operator()(index_t i, index_t j) const {
    JX1(i, j) = ZERO;
    JX2(i, j) = ZERO;
    JX3(i, j) = ZERO;
  }

  template <>
  Inline void
  ResetCurrents<Dimension::THREE_D>::operator()(index_t i, index_t j, index_t k) const {
    JX1(i, j, k) = ZERO;
    JX2(i, j, k) = ZERO;
    JX3(i, j, k) = ZERO;
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
    EX1(i) = ZERO;
    EX2(i) = ZERO;
    EX3(i) = ZERO;
    BX1(i) = ZERO;
    BX2(i) = ZERO;
    BX3(i) = ZERO;
  }

  template <>
  Inline void ResetFields<Dimension::TWO_D>::operator()(index_t i, index_t j) const {
    EX1(i, j) = ZERO;
    EX2(i, j) = ZERO;
    EX3(i, j) = ZERO;
    BX1(i, j) = ZERO;
    BX2(i, j) = ZERO;
    BX3(i, j) = ZERO;
  }

  template <>
  Inline void
  ResetFields<Dimension::THREE_D>::operator()(index_t i, index_t j, index_t k) const {
    EX1(i, j, k) = ZERO;
    EX2(i, j, k) = ZERO;
    EX3(i, j, k) = ZERO;
    BX1(i, j, k) = ZERO;
    BX2(i, j, k) = ZERO;
    BX3(i, j, k) = ZERO;
  }
} // namespace ntt

#endif
