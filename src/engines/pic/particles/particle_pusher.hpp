#ifndef PIC_PARTICLE_PUSHER_H
#define PIC_PARTICLE_PUSHER_H

#include "wrapper.h"

#include "field_macros.h"
#include "particle_macros.h"
#include "pic.h"

#include "io/output.h"
#include "meshblock/meshblock.h"
#include "meshblock/particles.h"
#include "utils/qmath.h"

#ifdef EXTERNAL_FORCE
#  include PGEN_HEADER
#endif

namespace ntt {
  struct Boris_t {};
  struct Photon_t {};

  /**
   * @brief Algorithm for the Particle pusher.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Pusher_kernel {
    Meshblock<D, PICEngine> m_mblock;
    Particles<D, PICEngine> m_particles;
#ifdef EXTERNAL_FORCE
    // PgenForceField<D, PICEngine> m_force_field;
    ProblemGenerator<D, PICEngine> m_pgen;
    array_t<real_t*>               m_work;
#endif
    const real_t m_time, m_coeff, m_dt;
    const int    m_ni2;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param particles Particles.
     * @param coeff Coefficient to be multiplied by dE/dt = coeff * curl B.
     * @param dt Time step.
     */
    Pusher_kernel(const Meshblock<D, PICEngine>& mblock,
                  const Particles<D, PICEngine>& particles,
#ifdef EXTERNAL_FORCE
                  const ProblemGenerator<D, PICEngine>& pgen,
                  array_t<real_t*>&                     work,
#endif
                  const real_t& time,
                  const real_t& coeff,
                  const real_t& dt)
      : m_mblock { mblock },
        m_particles { particles },
#ifdef EXTERNAL_FORCE
        m_pgen { pgen },
        m_work { work },
#endif
        m_time { time },
        m_coeff { coeff },
        m_dt { dt },
        m_ni2 { (int)mblock.Ni2() } {
    }

    /**
     * @brief Pusher for the forward Boris algorithm.
     * @param p index.
     */
    Inline void operator()(const Boris_t&, index_t p) const {
      if (m_particles.tag(p) == ParticleTag::alive) {
        vec_t<Dim3> e_int, b_int, e_int_Cart, b_int_Cart;
        interpolateFields(p, e_int, b_int);

#ifdef MINKOWSKI_METRIC
        coord_t<D> xp { ZERO };
#else
        coord_t<Dim3> xp { ZERO };
#endif
        getParticleCoordinate(p, xp);
        m_mblock.metric.v3_Cntrv2Cart(xp, e_int, e_int_Cart);
        m_mblock.metric.v3_Cntrv2Cart(xp, b_int, b_int_Cart);

#ifdef EXTERNAL_FORCE
        coord_t<D> xp_ph { ZERO };
#  ifdef MINKOWSKI_METRIC
        m_mblock.metric.x_Code2Cart(xp, xp_ph);
#  else
        coord_t<D> xp_ND { ZERO };
#    pragma unroll
        for (short d { 0 }; d < static_cast<short>(D); ++d) {
          xp_ND[d] = xp[d];
        }
        m_mblock.metric.x_Code2Sph(xp_ND, xp_ph);
#  endif

        const vec_t<Dim3> force_Hat { m_pgen.ext_force_x1(m_time, xp_ph),
                                      m_pgen.ext_force_x2(m_time, xp_ph),
                                      m_pgen.ext_force_x3(m_time, xp_ph) };
        vec_t<Dim3>       force_Cart { ZERO };
        m_mblock.metric.v3_Hat2Cart(xp, force_Hat, force_Cart);

        real_t t_gamma = math::sqrt(ONE + SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p))
                                    + SQR(m_particles.ux3(p)));
        real_t t_fdotu = force_Cart[0] * m_particles.ux1(p)
                         + force_Cart[1] * m_particles.ux2(p)
                         + force_Cart[2] * m_particles.ux3(p);
        m_work(p) += HALF * m_dt * t_fdotu / t_gamma;

        m_particles.ux1(p) += HALF * m_dt * force_Cart[0];
        m_particles.ux2(p) += HALF * m_dt * force_Cart[1];
        m_particles.ux3(p) += HALF * m_dt * force_Cart[2];

#endif

        BorisUpdate(p, e_int_Cart, b_int_Cart);

#ifdef EXTERNAL_FORCE

        t_gamma = math::sqrt(ONE + SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p))
                             + SQR(m_particles.ux3(p)));
        t_fdotu = force_Cart[0] * m_particles.ux1(p) + force_Cart[1] * m_particles.ux2(p)
                  + force_Cart[2] * m_particles.ux3(p);
        m_work(p) += HALF * m_dt * t_fdotu / t_gamma;

        m_particles.ux1(p) += HALF * m_dt * force_Cart[0];
        m_particles.ux2(p) += HALF * m_dt * force_Cart[1];
        m_particles.ux3(p) += HALF * m_dt * force_Cart[2];

#endif

        real_t inv_energy;
        inv_energy = ONE / get_prtl_Gamma_SR(m_particles, p);

        // contravariant 3-velocity: u^i / gamma
        vec_t<Dim3> v;
        m_mblock.metric.v3_Cart2Cntrv(
          xp, { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) }, v);
        // avoid problem for a particle right at the axes
        if ((m_particles.i2(p) == 0) && AlmostEqual(m_particles.dx2(p), 0.0f)) {
          v[2] = ZERO;
        } else if ((m_particles.i2(p) == m_ni2 - 1)
                   && AlmostEqual(m_particles.dx2(p), static_cast<prtldx_t>(1.0))) {
          v[2] = ZERO;
        }
        v[0] *= inv_energy;
        v[1] *= inv_energy;
        v[2] *= inv_energy;

        positionUpdate(p, v);

#ifndef MINKOWSKI_METRIC
        // !HOTFIX: THIS NEEDS TO BE FIXED FOR MPI
        reflectFromAxis(p);
#endif
      }
    }
    /**
     * @brief Pusher for the photon.
     * @param p index.
     */
    Inline void operator()(const Photon_t&, index_t p) const {
      if (m_particles.tag(p) == ParticleTag::alive) {
#ifdef MINKOWSKI_METRIC
        coord_t<D> xp;
#else
        coord_t<Dim3> xp;
#endif
        getParticleCoordinate(p, xp);
        vec_t<Dim3> v;
        m_mblock.metric.v3_Cart2Cntrv(
          xp, { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) }, v);

        real_t inv_energy;
        inv_energy = ONE / math::sqrt(get_prtl_Usqr_SR(m_particles, p));
        v[0] *= inv_energy;
        v[1] *= inv_energy;
        v[2] *= inv_energy;

        positionUpdate(p, v);
#ifndef MINKOWSKI_METRIC
        reflectFromAxis(p);
#endif
      }
    }

#ifdef MINKOWSKI_METRIC
    /**
     * @brief Transform particle coordinate from code units i+di to `real_t` type.
     * @param p index of the particle.
     * @param coord coordinate of the particle as a vector (of size D).
     */
    Inline void getParticleCoordinate(index_t&, coord_t<D>&) const;
#else
    /**
     * @brief Transform particle coordinate from code units i+di to `real_t` type.
     * @param p index of the particle.
     * @param coord coordinate of the particle as a vector (of size 3).
     */
    Inline void getParticleCoordinate(index_t&, coord_t<Dim3>&) const;

    /**
     * @brief Reflect particle coordinate and velocity from the axis ...
     * @brief ... only for 2D non-minkowski metric.
     * @param p index of the particle.
     */
    Inline void reflectFromAxis(index_t&) const;
#endif

    /**
     * @brief First order Yee mesh field interpolation to particle position.
     * @param p index of the particle.
     * @param e interpolated e-field vector of size 3 [return].
     * @param b interpolated b-field vector of size 3 [return].
     */
    Inline void interpolateFields(index_t&, vec_t<Dim3>&, vec_t<Dim3>&) const;

    /**
     * @brief Update particle positions according to updated velocities.
     * @param p index of the particle.
     * @param v particle 3-velocity.
     */
    Inline void positionUpdate(index_t&, const vec_t<Dim3>&) const;

    /**
     * @brief Update each position component.
     * @param p index of the particle.
     * @param v corresponding 3-velocity component.
     */
    Inline void positionUpdate_x1(index_t&, const real_t&) const;
    Inline void positionUpdate_x2(index_t&, const real_t&) const;
    Inline void positionUpdate_x3(index_t&, const real_t&) const;

    /**
     * @brief Boris algorithm.
     * @note Fields are modified inside the function and cannot be reused.
     * @param p index of the particle.
     * @param e interpolated e-field vector of size 3 [modified].
     * @param b interpolated b-field vector of size 3 [modified].
     */
    Inline void BorisUpdate(index_t&, vec_t<Dim3>&, vec_t<Dim3>&) const;
  };

#ifdef MINKOWSKI_METRIC
  template <>
  Inline void Pusher_kernel<Dim1>::getParticleCoordinate(index_t& p, coord_t<Dim1>& xp) const {
    xp[0] = get_prtl_x1(m_particles, p);
  }
  template <>
  Inline void Pusher_kernel<Dim2>::getParticleCoordinate(index_t& p, coord_t<Dim2>& xp) const {
    xp[0] = get_prtl_x1(m_particles, p);
    xp[1] = get_prtl_x2(m_particles, p);
  }
#else
  template <>
  Inline void Pusher_kernel<Dim1>::getParticleCoordinate(index_t&, coord_t<Dim3>&) const {
    NTTError("not applicable");
  }
  template <>
  Inline void Pusher_kernel<Dim2>::getParticleCoordinate(index_t& p, coord_t<Dim3>& xp) const {
    xp[0] = get_prtl_x1(m_particles, p);
    xp[1] = get_prtl_x2(m_particles, p);
    xp[2] = m_particles.phi(p);
  }

  template <>
  Inline void Pusher_kernel<Dim1>::reflectFromAxis(index_t&) const {
    NTTError("not applicable");
  }

  template <>
  Inline void Pusher_kernel<Dim2>::reflectFromAxis(index_t& p) const {
    if ((m_particles.i2(p) < 0) || (m_particles.i2(p) >= m_ni2)) {
      m_particles.dx2(p) = ONE - m_particles.dx2(p);
      m_particles.i2(p)  = IMIN(IMAX(m_particles.i2(p), 0), m_ni2 - 1);
      coord_t<Dim3> x_cu { get_prtl_x1(m_particles, p),
                           get_prtl_x2(m_particles, p),
                           m_particles.phi(p) };
      vec_t<Dim3>   v_Cntrv { ZERO }, v_Cart { ZERO };
      m_mblock.metric.v3_Cart2Cntrv(
        x_cu, { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) }, v_Cntrv);
      m_mblock.metric.v3_Cntrv2Cart(x_cu, { v_Cntrv[0], -v_Cntrv[1], v_Cntrv[2] }, v_Cart);
      m_particles.ux1(p) = v_Cart[0];
      m_particles.ux2(p) = v_Cart[1];
      m_particles.ux3(p) = v_Cart[2];
    }
  }

  template <>
  Inline void Pusher_kernel<Dim3>::reflectFromAxis(index_t&) const {}
#endif

  template <>
  Inline void Pusher_kernel<Dim3>::getParticleCoordinate(index_t& p, coord_t<Dim3>& xp) const {
    xp[0] = get_prtl_x1(m_particles, p);
    xp[1] = get_prtl_x2(m_particles, p);
    xp[2] = get_prtl_x3(m_particles, p);
  }

  // * * * * * * * * * * * * * * *
  // General position update
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Pusher_kernel<Dim1>::positionUpdate(index_t& p, const vec_t<Dim3>& v) const {
    positionUpdate_x1(p, v[0]);
  }
  template <>
  Inline void Pusher_kernel<Dim2>::positionUpdate(index_t& p, const vec_t<Dim3>& v) const {
    positionUpdate_x1(p, v[0]);
    positionUpdate_x2(p, v[1]);
#ifndef MINKOWSKI_METRIC
    m_particles.phi(p) += m_dt * v[2];
#endif
  }
  template <>
  Inline void Pusher_kernel<Dim3>::positionUpdate(index_t& p, const vec_t<Dim3>& v) const {
    positionUpdate_x1(p, v[0]);
    positionUpdate_x2(p, v[1]);
    positionUpdate_x3(p, v[2]);
  }

  template <Dimension D>
  Inline void Pusher_kernel<D>::positionUpdate_x1(index_t& p, const real_t& vx1) const {
    m_particles.dx1(p) = m_particles.dx1(p) + static_cast<prtldx_t>(m_dt * vx1);
    int      temp_i { static_cast<int>(m_particles.dx1(p)) };
    prtldx_t temp_r { math::fmax(SIGNf(m_particles.dx1(p)) + temp_i,
                                 static_cast<prtldx_t>(temp_i))
                      - static_cast<prtldx_t>(1.0) };
    temp_i             = static_cast<int>(temp_r);
    m_particles.i1(p)  = m_particles.i1(p) + temp_i;
    m_particles.dx1(p) = m_particles.dx1(p) - temp_r;
  }
  template <Dimension D>
  Inline void Pusher_kernel<D>::positionUpdate_x2(index_t& p, const real_t& vx2) const {
    m_particles.dx2(p) = m_particles.dx2(p) + static_cast<prtldx_t>(m_dt * vx2);
    int      temp_i { static_cast<int>(m_particles.dx2(p)) };
    prtldx_t temp_r { math::fmax(SIGNf(m_particles.dx2(p)) + temp_i,
                                 static_cast<prtldx_t>(temp_i))
                      - static_cast<prtldx_t>(1.0) };
    temp_i             = static_cast<int>(temp_r);
    m_particles.i2(p)  = m_particles.i2(p) + temp_i;
    m_particles.dx2(p) = m_particles.dx2(p) - temp_r;
  }
  template <Dimension D>
  Inline void Pusher_kernel<D>::positionUpdate_x3(index_t& p, const real_t& vx3) const {
    m_particles.dx3(p) = m_particles.dx3(p) + static_cast<prtldx_t>(m_dt * vx3);
    int      temp_i { static_cast<int>(m_particles.dx3(p)) };
    prtldx_t temp_r { math::fmax(SIGNf(m_particles.dx3(p)) + temp_i,
                                 static_cast<prtldx_t>(temp_i))
                      - static_cast<prtldx_t>(1.0) };
    temp_i             = static_cast<int>(temp_r);
    m_particles.i3(p)  = m_particles.i3(p) + temp_i;
    m_particles.dx3(p) = m_particles.dx3(p) - temp_r;
  }

  // * * * * * * * * * * * * * * *
  // Boris velocity update
  // * * * * * * * * * * * * * * *
  template <Dimension D>
  Inline void Pusher_kernel<D>::BorisUpdate(index_t&     p,
                                            vec_t<Dim3>& e0,
                                            vec_t<Dim3>& b0) const {
    real_t COEFF { m_coeff };

    e0[0] *= COEFF;
    e0[1] *= COEFF;
    e0[2] *= COEFF;
    vec_t<Dim3> u0 { m_particles.ux1(p) + e0[0],
                     m_particles.ux2(p) + e0[1],
                     m_particles.ux3(p) + e0[2] };

    COEFF *= ONE / math::sqrt(ONE + SQR(u0[0]) + SQR(u0[1]) + SQR(u0[2]));
    b0[0] *= COEFF;
    b0[1] *= COEFF;
    b0[2] *= COEFF;
    COEFF = TWO / (ONE + SQR(b0[0]) + SQR(b0[1]) + SQR(b0[2]));

    vec_t<Dim3> u1 { (u0[0] + u0[1] * b0[2] - u0[2] * b0[1]) * COEFF,
                     (u0[1] + u0[2] * b0[0] - u0[0] * b0[2]) * COEFF,
                     (u0[2] + u0[0] * b0[1] - u0[1] * b0[0]) * COEFF };

    u0[0] += u1[1] * b0[2] - u1[2] * b0[1] + e0[0];
    u0[1] += u1[2] * b0[0] - u1[0] * b0[2] + e0[1];
    u0[2] += u1[0] * b0[1] - u1[1] * b0[0] + e0[2];

    m_particles.ux1(p) = u0[0];
    m_particles.ux2(p) = u0[1];
    m_particles.ux3(p) = u0[2];
  }

  // * * * * * * * * * * * * * * *
  // Field interpolations
  // * * * * * * * * * * * * * * *

  template <>
  Inline void Pusher_kernel<Dim1>::interpolateFields(index_t&     p,
                                                     vec_t<Dim3>& e0,
                                                     vec_t<Dim3>& b0) const {
    const auto   i { m_particles.i1(p) + N_GHOSTS };
    const real_t dx1 { static_cast<real_t>(m_particles.dx1(p)) };

    // first order
    real_t       c0, c1;

    // Ex1
    // interpolate to nodes
    c0    = HALF * (EX1(i) + EX1(i - 1));
    c1    = HALF * (EX1(i) + EX1(i + 1));
    // interpolate from nodes to the particle position
    e0[0] = c0 * (ONE - dx1) + c1 * dx1;
    // Ex2
    c0    = EX2(i);
    c1    = EX2(i + 1);
    e0[1] = c0 * (ONE - dx1) + c1 * dx1;
    // Ex3
    c0    = EX3(i);
    c1    = EX3(i + 1);
    e0[2] = c0 * (ONE - dx1) + c1 * dx1;

    // Bx1
    c0    = BX1(i);
    c1    = BX1(i + 1);
    b0[0] = c0 * (ONE - dx1) + c1 * dx1;
    // Bx2
    c0    = HALF * (BX2(i - 1) + BX2(i));
    c1    = HALF * (BX2(i) + BX2(i + 1));
    b0[1] = c0 * (ONE - dx1) + c1 * dx1;
    // Bx3
    c0    = HALF * (BX3(i - 1) + BX3(i));
    c1    = HALF * (BX3(i) + BX3(i + 1));
    b0[2] = c0 * (ONE - dx1) + c1 * dx1;
  }

  template <>
  Inline void Pusher_kernel<Dim2>::interpolateFields(index_t&     p,
                                                     vec_t<Dim3>& e0,
                                                     vec_t<Dim3>& b0) const {
    const auto   i { m_particles.i1(p) + N_GHOSTS };
    const real_t dx1 { static_cast<real_t>(m_particles.dx1(p)) };
    const auto   j { m_particles.i2(p) + N_GHOSTS };
    const real_t dx2 { static_cast<real_t>(m_particles.dx2(p)) };

    // first order
    real_t       c000, c100, c010, c110, c00, c10;

    // Ex1
    // interpolate to nodes
    c000  = HALF * (EX1(i, j) + EX1(i - 1, j));
    c100  = HALF * (EX1(i, j) + EX1(i + 1, j));
    c010  = HALF * (EX1(i, j + 1) + EX1(i - 1, j + 1));
    c110  = HALF * (EX1(i, j + 1) + EX1(i + 1, j + 1));
    // interpolate from nodes to the particle position
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    e0[0] = c00 * (ONE - dx2) + c10 * dx2;
    // Ex2
    c000  = HALF * (EX2(i, j) + EX2(i, j - 1));
    c100  = HALF * (EX2(i + 1, j) + EX2(i + 1, j - 1));
    c010  = HALF * (EX2(i, j) + EX2(i, j + 1));
    c110  = HALF * (EX2(i + 1, j) + EX2(i + 1, j + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    e0[1] = c00 * (ONE - dx2) + c10 * dx2;
    // Ex3
    c000  = EX3(i, j);
    c100  = EX3(i + 1, j);
    c010  = EX3(i, j + 1);
    c110  = EX3(i + 1, j + 1);
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    e0[2] = c00 * (ONE - dx2) + c10 * dx2;

    // Bx1
    c000  = HALF * (BX1(i, j) + BX1(i, j - 1));
    c100  = HALF * (BX1(i + 1, j) + BX1(i + 1, j - 1));
    c010  = HALF * (BX1(i, j) + BX1(i, j + 1));
    c110  = HALF * (BX1(i + 1, j) + BX1(i + 1, j + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    b0[0] = c00 * (ONE - dx2) + c10 * dx2;
    // Bx2
    c000  = HALF * (BX2(i - 1, j) + BX2(i, j));
    c100  = HALF * (BX2(i, j) + BX2(i + 1, j));
    c010  = HALF * (BX2(i - 1, j + 1) + BX2(i, j + 1));
    c110  = HALF * (BX2(i, j + 1) + BX2(i + 1, j + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    b0[1] = c00 * (ONE - dx2) + c10 * dx2;
    // Bx3
    c000  = INV_4 * (BX3(i - 1, j - 1) + BX3(i - 1, j) + BX3(i, j - 1) + BX3(i, j));
    c100  = INV_4 * (BX3(i, j - 1) + BX3(i, j) + BX3(i + 1, j - 1) + BX3(i + 1, j));
    c010  = INV_4 * (BX3(i - 1, j) + BX3(i - 1, j + 1) + BX3(i, j) + BX3(i, j + 1));
    c110  = INV_4 * (BX3(i, j) + BX3(i, j + 1) + BX3(i + 1, j) + BX3(i + 1, j + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    b0[2] = c00 * (ONE - dx2) + c10 * dx2;
  }

  template <>
  Inline void Pusher_kernel<Dim3>::interpolateFields(index_t&     p,
                                                     vec_t<Dim3>& e0,
                                                     vec_t<Dim3>& b0) const {
    const auto   i { m_particles.i1(p) + N_GHOSTS };
    const real_t dx1 { static_cast<real_t>(m_particles.dx1(p)) };
    const auto   j { m_particles.i2(p) + N_GHOSTS };
    const real_t dx2 { static_cast<real_t>(m_particles.dx2(p)) };
    const auto   k { m_particles.i3(p) + N_GHOSTS };
    const real_t dx3 { static_cast<real_t>(m_particles.dx3(p)) };

    // first order
    real_t       c000, c100, c010, c110, c001, c101, c011, c111, c00, c10, c01, c11, c0, c1;

    // Ex1
    // interpolate to nodes
    c000  = HALF * (EX1(i, j, k) + EX1(i - 1, j, k));
    c100  = HALF * (EX1(i, j, k) + EX1(i + 1, j, k));
    c010  = HALF * (EX1(i, j + 1, k) + EX1(i - 1, j + 1, k));
    c110  = HALF * (EX1(i, j + 1, k) + EX1(i + 1, j + 1, k));
    // interpolate from nodes to the particle position
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    // interpolate to nodes
    c001  = HALF * (EX1(i, j, k + 1) + EX1(i - 1, j, k + 1));
    c101  = HALF * (EX1(i, j, k + 1) + EX1(i + 1, j, k + 1));
    c011  = HALF * (EX1(i, j + 1, k + 1) + EX1(i - 1, j + 1, k + 1));
    c111  = HALF * (EX1(i, j + 1, k + 1) + EX1(i + 1, j + 1, k + 1));
    // interpolate from nodes to the particle position
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    e0[0] = c0 * (ONE - dx3) + c1 * dx3;

    // Ex2
    c000  = HALF * (EX2(i, j, k) + EX2(i, j - 1, k));
    c100  = HALF * (EX2(i + 1, j, k) + EX2(i + 1, j - 1, k));
    c010  = HALF * (EX2(i, j, k) + EX2(i, j + 1, k));
    c110  = HALF * (EX2(i + 1, j, k) + EX2(i + 1, j + 1, k));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c001  = HALF * (EX2(i, j, k + 1) + EX2(i, j - 1, k + 1));
    c101  = HALF * (EX2(i + 1, j, k + 1) + EX2(i + 1, j - 1, k + 1));
    c011  = HALF * (EX2(i, j, k + 1) + EX2(i, j + 1, k + 1));
    c111  = HALF * (EX2(i + 1, j, k + 1) + EX2(i + 1, j + 1, k + 1));
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    e0[1] = c0 * (ONE - dx3) + c1 * dx3;

    // Ex3
    c000  = HALF * (EX3(i, j, k) + EX3(i, j, k - 1));
    c100  = HALF * (EX3(i + 1, j, k) + EX3(i + 1, j, k - 1));
    c010  = HALF * (EX3(i, j + 1, k) + EX3(i, j + 1, k - 1));
    c110  = HALF * (EX3(i + 1, j + 1, k) + EX3(i + 1, j + 1, k - 1));
    c001  = HALF * (EX3(i, j, k) + EX3(i, j, k + 1));
    c101  = HALF * (EX3(i + 1, j, k) + EX3(i + 1, j, k + 1));
    c011  = HALF * (EX3(i, j + 1, k) + EX3(i, j + 1, k + 1));
    c111  = HALF * (EX3(i + 1, j + 1, k) + EX3(i + 1, j + 1, k + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    e0[2] = c0 * (ONE - dx3) + c1 * dx3;

    // Bx1
    c000 = INV_4 * (BX1(i, j, k) + BX1(i, j - 1, k) + BX1(i, j, k - 1) + BX1(i, j - 1, k - 1));
    c100 = INV_4
           * (BX1(i + 1, j, k) + BX1(i + 1, j - 1, k) + BX1(i + 1, j, k - 1)
              + BX1(i + 1, j - 1, k - 1));
    c001 = INV_4 * (BX1(i, j, k) + BX1(i, j, k + 1) + BX1(i, j - 1, k) + BX1(i, j - 1, k + 1));
    c101 = INV_4
           * (BX1(i + 1, j, k) + BX1(i + 1, j, k + 1) + BX1(i + 1, j - 1, k)
              + BX1(i + 1, j - 1, k + 1));
    c010 = INV_4 * (BX1(i, j, k) + BX1(i, j + 1, k) + BX1(i, j, k - 1) + BX1(i, j + 1, k - 1));
    c110 = INV_4
           * (BX1(i + 1, j, k) + BX1(i + 1, j, k - 1) + BX1(i + 1, j + 1, k - 1)
              + BX1(i + 1, j + 1, k));
    c011 = INV_4 * (BX1(i, j, k) + BX1(i, j + 1, k) + BX1(i, j + 1, k + 1) + BX1(i, j, k + 1));
    c111 = INV_4
           * (BX1(i + 1, j, k) + BX1(i + 1, j + 1, k) + BX1(i + 1, j + 1, k + 1)
              + BX1(i + 1, j, k + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    b0[0] = c0 * (ONE - dx3) + c1 * dx3;

    // Bx2
    c000 = INV_4 * (BX2(i - 1, j, k - 1) + BX2(i - 1, j, k) + BX2(i, j, k - 1) + BX2(i, j, k));
    c100 = INV_4 * (BX2(i, j, k - 1) + BX2(i, j, k) + BX2(i + 1, j, k - 1) + BX2(i + 1, j, k));
    c001 = INV_4 * (BX2(i - 1, j, k) + BX2(i - 1, j, k + 1) + BX2(i, j, k) + BX2(i, j, k + 1));
    c101 = INV_4 * (BX2(i, j, k) + BX2(i, j, k + 1) + BX2(i + 1, j, k) + BX2(i + 1, j, k + 1));
    c010 = INV_4
           * (BX2(i - 1, j + 1, k - 1) + BX2(i - 1, j + 1, k) + BX2(i, j + 1, k - 1)
              + BX2(i, j + 1, k));
    c110 = INV_4
           * (BX2(i, j + 1, k - 1) + BX2(i, j + 1, k) + BX2(i + 1, j + 1, k - 1)
              + BX2(i + 1, j + 1, k));
    c011 = INV_4
           * (BX2(i - 1, j + 1, k) + BX2(i - 1, j + 1, k + 1) + BX2(i, j + 1, k)
              + BX2(i, j + 1, k + 1));
    c111 = INV_4
           * (BX2(i, j + 1, k) + BX2(i, j + 1, k + 1) + BX2(i + 1, j + 1, k)
              + BX2(i + 1, j + 1, k + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    b0[1] = c0 * (ONE - dx3) + c1 * dx3;

    // Bx3
    c000 = INV_4 * (BX3(i - 1, j - 1, k) + BX3(i - 1, j, k) + BX3(i, j - 1, k) + BX3(i, j, k));
    c100 = INV_4 * (BX3(i, j - 1, k) + BX3(i, j, k) + BX3(i + 1, j - 1, k) + BX3(i + 1, j, k));
    c001 = INV_4
           * (BX3(i - 1, j - 1, k + 1) + BX3(i - 1, j, k + 1) + BX3(i, j - 1, k + 1)
              + BX3(i, j, k + 1));
    c101 = INV_4
           * (BX3(i, j - 1, k + 1) + BX3(i, j, k + 1) + BX3(i + 1, j - 1, k + 1)
              + BX3(i + 1, j, k + 1));
    c010 = INV_4 * (BX3(i - 1, j, k) + BX3(i - 1, j + 1, k) + BX3(i, j, k) + BX3(i, j + 1, k));
    c110 = INV_4 * (BX3(i, j, k) + BX3(i, j + 1, k) + BX3(i + 1, j, k) + BX3(i + 1, j + 1, k));
    c011 = INV_4
           * (BX3(i - 1, j, k + 1) + BX3(i - 1, j + 1, k + 1) + BX3(i, j, k + 1)
              + BX3(i, j + 1, k + 1));
    c111 = INV_4
           * (BX3(i, j, k + 1) + BX3(i, j + 1, k + 1) + BX3(i + 1, j, k + 1)
              + BX3(i + 1, j + 1, k + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    b0[2] = c0 * (ONE - dx3) + c1 * dx3;
  }

}    // namespace ntt

/*
// Inline void operator()(const BorisBwd_t&, index_t p) const {
//   real_t inv_energy;
//   inv_energy = SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p)) +
//   SQR(m_particles.ux3(p)); inv_energy = ONE / math::sqrt(ONE + inv_energy);

//   coord_t<D> xp;
//   getParticleCoordinate(p, xp);

//   vec_t<Dim3> v;
//   m_mblock.metric.v3_Cart2Cntrv(
//     xp, {m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p)}, v);
//   v[0] *= inv_energy;
//   v[1] *= inv_energy;
//   v[2] *= inv_energy;
//   positionUpdate(p, v);
//   getParticleCoordinate(p, xp);

//   vec_t<Dim3> e_int, b_int, e_int_Cart, b_int_Cart;
//   interpolateFields(p, e_int, b_int);

//   m_mblock.metric.v3_Cntrv2Cart(xp, e_int, e_int_Cart);
//   m_mblock.metric.v3_Cntrv2Cart(xp, b_int, b_int_Cart);

//   BorisUpdate(p, e_int_Cart, b_int_Cart);
// }

// !HACK: hack for sync. radiation
// real_t ux1_init { m_particles.ux1(p) };
// real_t ux2_init { m_particles.ux2(p) };
// real_t ux3_init { m_particles.ux3(p) };
// real_t ex1 { e0[0] };
// real_t ex2 { e0[1] };
// real_t ex3 { e0[2] };
// real_t bx1 { b0[0] };
// real_t bx2 { b0[1] };
// real_t bx3 { b0[2] };

// {
// !HACK: radiation
// ux1_init     = HALF * (ux1_init + u0[0]);
// ux2_init     = HALF * (ux2_init + u0[1]);
// ux3_init     = HALF * (ux3_init + u0[2]);
// real_t gamma = math::sqrt(ONE + SQR(ux1_init) + SQR(ux2_init) + SQR(ux3_init));
// if (gamma > 5.0) {
//   real_t beta       = math::sqrt(ONE - ONE / SQR(gamma));
//   real_t e_bar_x1   = ex1 + (ux2_init * bx3 - ux3_init * bx2) / gamma;
//   real_t e_bar_x2   = ex2 + (ux3_init * bx1 - ux1_init * bx3) / gamma;
//   real_t e_bar_x3   = ex3 + (ux1_init * bx2 - ux2_init * bx1) / gamma;
//   real_t e_bar_sq   = SQR(e_bar_x1) + SQR(e_bar_x2) + SQR(e_bar_x3);
//   real_t beta_dot_e = (ex1 * ux1_init + ex2 * ux2_init + ex3 * ux3_init) / gamma;
//   real_t chiR_sq    = math::abs(e_bar_sq - beta_dot_e * beta_dot_e);
//   real_t kappaR_x1  = (bx3 * e_bar_x2 - bx2 * e_bar_x3) + ex1 * beta_dot_e;
//   real_t kappaR_x2  = (bx1 * e_bar_x3 - bx3 * e_bar_x1) + ex2 * beta_dot_e;
//   real_t kappaR_x3  = (bx2 * e_bar_x1 - bx1 * e_bar_x2) + ex3 * beta_dot_e;

//   real_t dummy = TWO * COEFF * static_cast<real_t>(0.1) / SQR(static_cast<real_t>(1.0));
//   u0[0] += dummy * (kappaR_x1 - chiR_sq * gamma * ux1_init);
//   u0[1] += dummy * (kappaR_x2 - chiR_sq * gamma * ux2_init);
//   u0[2] += dummy * (kappaR_x3 - chiR_sq * gamma * ux3_init);
// }
// !HACK: parallel + ExB velocity only
// real_t b_sq { SQR(bx1) + SQR(bx2) + SQR(bx3) };
// if (b_sq > 0.1) {
//   real_t Gamma { math::sqrt(ONE + SQR(u0[0]) + SQR(u0[1]) + SQR(u0[2])) };
//   real_t e_sq { SQR(ex1) + SQR(ex2) + SQR(ex3) };
//   real_t e_dot_b { ex1 * bx1 + ex2 * bx2 + ex3 * bx3 };
//   real_t e_prime_sq { TWO * SQR(e_dot_b)
//                       / ((b_sq - e_sq)
//                          + math::sqrt(SQR(b_sq - e_sq) + FOUR * SQR(e_dot_b))) };
//   real_t beta0_x1 { (ex2 * bx3 - ex3 * bx2) / (b_sq + e_prime_sq) };
//   real_t beta0_x2 { (ex3 * bx1 - ex1 * bx3) / (b_sq + e_prime_sq) };
//   real_t beta0_x3 { (ex1 * bx2 - ex2 * bx1) / (b_sq + e_prime_sq) };
//   real_t beta0_sq { SQR(beta0_x1) + SQR(beta0_x2) + SQR(beta0_x3) };
//   real_t bprime_x1 { bx1 - (beta0_x2 * ex3 - beta0_x3 * ex2) };
//   real_t bprime_x2 { bx2 - (beta0_x3 * ex1 - beta0_x1 * ex3) };
//   real_t bprime_x3 { bx3 - (beta0_x1 * ex2 - beta0_x2 * ex1) };
//   real_t bprime_sq { SQR(bprime_x1) + SQR(bprime_x2) + SQR(bprime_x3) };
//   real_t u_dot_bprime { u0[0] * bprime_x1 + u0[1] * bprime_x2 + u0[2] * bprime_x3 };
//   real_t uprime_x1 { u_dot_bprime * bprime_x1 / bprime_sq };
//   real_t uprime_x2 { u_dot_bprime * bprime_x2 / bprime_sq };
//   real_t uprime_x3 { u_dot_bprime * bprime_x3 / bprime_sq };
//   real_t uprime_sq { SQR(uprime_x1) + SQR(uprime_x2) + SQR(uprime_x3) };
//   Gamma = math::sqrt((ONE + uprime_sq) / (ONE - beta0_sq));
//   u0[0] = beta0_x1 * Gamma + uprime_x1;
//   u0[1] = beta0_x2 * Gamma + uprime_x2;
//   u0[2] = beta0_x3 * Gamma + uprime_x3;
// }
// }

// // Interpolation ifdefs

// #define C0_EX1     HALF*(EX1(i) + EX1(i - 1))
// #define C1_EX1     HALF*(EX1(i) + EX1(i + 1))
// #define EX1_INTERP C0_EX1*(ONE - dx1) + C1_EX1* dx1

// #define C0_EX2     EX2(i)
// #define C1_EX2     EX2(i + 1)
// #define EX2_INTERP C0_EX2*(ONE - dx1) + C1_EX2* dx1

// #define C0_EX3     EX3(i)
// #define C1_EX3     EX3(i + 1)
// #define EX3_INTERP C0_EX3*(ONE - dx1) + C1_EX3* dx1

// #define C0_BX1     BX1(i)
// #define C1_BX1     BX1(i + 1)
// #define BX1_INTERP C0_BX1*(ONE - dx1) + C1_BX1* dx1

// #define C0_BX2     HALF*(BX2(i - 1) + BX2(i))
// #define C1_BX2     HALF*(BX2(i) + BX2(i + 1))
// #define BX2_INTERP C0_BX2*(ONE - dx1) + C1_BX2* dx1

// #define C0_BX3     HALF*(BX3(i - 1) + BX3(i))
// #define C1_BX3     HALF*(BX3(i) + BX3(i + 1))
// #define BX3_INTERP C0_BX3*(ONE - dx1) + C1_BX3* dx1

//   template <>
//   Inline void Pusher_kernel<Dim1>::interpolateFields(index_t&     p,
//                                                      vec_t<Dim3>& e0,
//                                                      vec_t<Dim3>& b0) const {
//     const auto i { m_particles.i1(p) + N_GHOSTS };
//     const auto dx1 { static_cast<real_t>(m_particles.dx1(p)) };

//     e0[0] = EX1_INTERP;
//     e0[1] = EX2_INTERP;
//     e0[2] = EX3_INTERP;
//     b0[0] = BX1_INTERP;
//     b0[1] = BX2_INTERP;
//     b0[2] = BX3_INTERP;
//   }

// #undef C0_EX1
// #undef C1_EX1
// #undef EX1_INTERP

// #undef C0_EX2
// #undef C1_EX2
// #undef EX2_INTERP

// #undef C0_EX3
// #undef C1_EX3
// #undef EX3_INTERP

// #undef C0_BX1
// #undef C1_BX1
// #undef BX1_INTERP

// #undef C0_BX2
// #undef C1_BX2
// #undef BX2_INTERP

// #undef C0_BX3
// #undef C1_BX3
// #undef BX3_INTERP

// #define C000_EX1   HALF*(EX1(i, j) + EX1(i - 1, j))
// #define C100_EX1   HALF*(EX1(i, j) + EX1(i + 1, j))
// #define C010_EX1   HALF*(EX1(i, j + 1) + EX1(i - 1, j + 1))
// #define C110_EX1   HALF*(EX1(i, j + 1) + EX1(i + 1, j + 1))
// #define C00_EX1    C000_EX1*(ONE - dx1) + C100_EX1* dx1
// #define C10_EX1    C010_EX1*(ONE - dx1) + C110_EX1* dx1
// #define EX1_INTERP C00_EX1*(ONE - dx2) + C10_EX1* dx2

// #define C000_EX2   HALF*(EX2(i, j) + EX2(i, j - 1))
// #define C100_EX2   HALF*(EX2(i + 1, j) + EX2(i + 1, j - 1))
// #define C010_EX2   HALF*(EX2(i, j) + EX2(i, j + 1))
// #define C110_EX2   HALF*(EX2(i + 1, j) + EX2(i + 1, j + 1))
// #define C00_EX2    C000_EX2*(ONE - dx1) + C100_EX2* dx1
// #define C10_EX2    C010_EX2*(ONE - dx1) + C110_EX2* dx1
// #define EX2_INTERP C00_EX2*(ONE - dx2) + C10_EX2* dx2

// #define C000_EX3   EX3(i, j)
// #define C100_EX3   EX3(i + 1, j)
// #define C010_EX3   EX3(i, j + 1)
// #define C110_EX3   EX3(i + 1, j + 1)
// #define C00_EX3    C000_EX3*(ONE - dx1) + C100_EX3* dx1
// #define C10_EX3    C010_EX3*(ONE - dx1) + C110_EX3* dx1
// #define EX3_INTERP C00_EX3*(ONE - dx2) + C10_EX3* dx2

// #define C000_BX1   HALF*(BX1(i, j) + BX1(i, j - 1))
// #define C100_BX1   HALF*(BX1(i + 1, j) + BX1(i + 1, j - 1))
// #define C010_BX1   HALF*(BX1(i, j) + BX1(i, j + 1))
// #define C110_BX1   HALF*(BX1(i + 1, j) + BX1(i + 1, j + 1))
// #define C00_BX1    C000_BX1*(ONE - dx1) + C100_BX1* dx1
// #define C10_BX1    C010_BX1*(ONE - dx1) + C110_BX1* dx1
// #define BX1_INTERP C00_BX1*(ONE - dx2) + C10_BX1* dx2

// #define C000_BX2   HALF*(BX2(i - 1, j) + BX2(i, j))
// #define C100_BX2   HALF*(BX2(i, j) + BX2(i + 1, j))
// #define C010_BX2   HALF*(BX2(i - 1, j + 1) + BX2(i, j + 1))
// #define C110_BX2   HALF*(BX2(i, j + 1) + BX2(i + 1, j + 1))
// #define C00_BX2    C000_BX2*(ONE - dx1) + C100_BX2* dx1
// #define C10_BX2    C010_BX2*(ONE - dx1) + C110_BX2* dx1
// #define BX2_INTERP C00_BX2*(ONE - dx2) + C10_BX2* dx2

// #define C000_BX3   INV_4*(BX3(i - 1, j - 1) + BX3(i - 1, j) + BX3(i, j - 1) + BX3(i, j))
// #define C100_BX3   INV_4*(BX3(i, j - 1) + BX3(i, j) + BX3(i + 1, j - 1) + BX3(i + 1, j))
// #define C010_BX3   INV_4*(BX3(i - 1, j) + BX3(i - 1, j + 1) + BX3(i, j) + BX3(i, j + 1))
// #define C110_BX3   INV_4*(BX3(i, j) + BX3(i, j + 1) + BX3(i + 1, j) + BX3(i + 1, j + 1))
// #define C00_BX3    C000_BX3*(ONE - dx1) + C100_BX3* dx1
// #define C10_BX3    C010_BX3*(ONE - dx1) + C110_BX3* dx1
// #define BX3_INTERP C00_BX3*(ONE - dx2) + C10_BX3* dx2

//   template <>
//   Inline void Pusher_kernel<Dim2>::interpolateFields(index_t&     p,
//                                                      vec_t<Dim3>& e0,
//                                                      vec_t<Dim3>& b0) const {
//     const auto i { m_particles.i1(p) + N_GHOSTS };
//     const auto dx1 { static_cast<real_t>(m_particles.dx1(p)) };
//     const auto j { m_particles.i2(p) + N_GHOSTS };
//     const auto dx2 { static_cast<real_t>(m_particles.dx2(p)) };

//     e0[0] = EX1_INTERP;
//     e0[1] = EX2_INTERP;
//     e0[2] = EX3_INTERP;
//     b0[0] = BX1_INTERP;
//     b0[1] = BX2_INTERP;
//     b0[2] = BX3_INTERP;
//   }

// #undef C000_EX1
// #undef C100_EX1
// #undef C010_EX1
// #undef C110_EX1
// #undef C00_EX1
// #undef C10_EX1
// #undef EX1_INTERP

// #undef C000_EX2
// #undef C100_EX2
// #undef C010_EX2
// #undef C110_EX2
// #undef C00_EX2
// #undef C10_EX2
// #undef EX2_INTERP

// #undef C000_EX3
// #undef C100_EX3
// #undef C010_EX3
// #undef C110_EX3
// #undef C00_EX3
// #undef C10_EX3
// #undef EX3_INTERP

// #undef C000_BX1
// #undef C100_BX1
// #undef C010_BX1
// #undef C110_BX1
// #undef C00_BX1
// #undef C10_BX1
// #undef BX1_INTERP

// #undef C000_BX2
// #undef C100_BX2
// #undef C010_BX2
// #undef C110_BX2
// #undef C00_BX2
// #undef C10_BX2
// #undef BX2_INTERP

// #undef C000_BX3
// #undef C100_BX3
// #undef C010_BX3
// #undef C110_BX3
// #undef C00_BX3
// #undef C10_BX3
// #undef BX3_INTERP

// #define C000_EX1   (HALF * (EX1(i, j, k) + EX1(i - 1, j, k)))
// #define C100_EX1   (HALF * (EX1(i, j, k) + EX1(i + 1, j, k)))
// #define C010_EX1   (HALF * (EX1(i, j + 1, k) + EX1(i - 1, j + 1, k)))
// #define C110_EX1   (HALF * (EX1(i, j + 1, k) + EX1(i + 1, j + 1, k)))
// #define C001_EX1   (HALF * (EX1(i, j, k + 1) + EX1(i - 1, j, k + 1)))
// #define C101_EX1   (HALF * (EX1(i, j, k + 1) + EX1(i + 1, j, k + 1)))
// #define C011_EX1   (HALF * (EX1(i, j + 1, k + 1) + EX1(i - 1, j + 1, k + 1)))
// #define C111_EX1   (HALF * (EX1(i, j + 1, k + 1) + EX1(i + 1, j + 1, k + 1)))
// #define C00_EX1    (C000_EX1 * (ONE - dx1) + C100_EX1 * dx1)
// #define C10_EX1    (C010_EX1 * (ONE - dx1) + C110_EX1 * dx1)
// #define C01_EX1    (C001_EX1 * (ONE - dx1) + C101_EX1 * dx1)
// #define C11_EX1    (C011_EX1 * (ONE - dx1) + C111_EX1 * dx1)
// #define C0_EX1     (C00_EX1 * (ONE - dx2) + C10_EX1 * dx2)
// #define C1_EX1     (C01_EX1 * (ONE - dx2) + C11_EX1 * dx2)
// #define EX1_INTERP (C0_EX1 * (ONE - dx3) + C1_EX1 * dx3)

// #define C000_EX2   HALF*(EX2(i, j, k) + EX2(i, j - 1, k))
// #define C100_EX2   HALF*(EX2(i + 1, j, k) + EX2(i + 1, j - 1, k))
// #define C010_EX2   HALF*(EX2(i, j, k) + EX2(i, j + 1, k))
// #define C110_EX2   HALF*(EX2(i + 1, j, k) + EX2(i + 1, j + 1, k))
// #define C00_EX2    C000_EX2*(ONE - dx1) + C100_EX2* dx1
// #define C10_EX2    C010_EX2*(ONE - dx1) + C110_EX2* dx1
// #define C0_EX2     C00_EX2*(ONE - dx2) + C10_EX2* dx2
// #define C001_EX2   HALF*(EX2(i, j, k + 1) + EX2(i, j - 1, k + 1))
// #define C101_EX2   HALF*(EX2(i + 1, j, k + 1) + EX2(i + 1, j - 1, k + 1))
// #define C011_EX2   HALF*(EX2(i, j, k + 1) + EX2(i, j + 1, k + 1))
// #define C111_EX2   HALF*(EX2(i + 1, j, k + 1) + EX2(i + 1, j + 1, k + 1))
// #define C01_EX2    C001_EX2*(ONE - dx1) + C101_EX2* dx1
// #define C11_EX2    C011_EX2*(ONE - dx1) + C111_EX2* dx1
// #define C1_EX2     C01_EX2*(ONE - dx2) + C11_EX2* dx2
// #define EX2_INTERP C0_EX2*(ONE - dx3) + C1_EX2* dx3

// #define C000_EX3   HALF*(EX3(i, j, k) + EX3(i, j, k - 1))
// #define C100_EX3   HALF*(EX3(i + 1, j, k) + EX3(i + 1, j, k - 1))
// #define C010_EX3   HALF*(EX3(i, j + 1, k) + EX3(i, j + 1, k - 1))
// #define C110_EX3   HALF*(EX3(i + 1, j + 1, k) + EX3(i + 1, j + 1, k - 1))
// #define C001_EX3   HALF*(EX3(i, j, k) + EX3(i, j, k + 1))
// #define C101_EX3   HALF*(EX3(i + 1, j, k) + EX3(i + 1, j, k + 1))
// #define C011_EX3   HALF*(EX3(i, j + 1, k) + EX3(i, j + 1, k + 1))
// #define C111_EX3   HALF*(EX3(i + 1, j + 1, k) + EX3(i + 1, j + 1, k + 1))
// #define C00_EX3    C000_EX3*(ONE - dx1) + C100_EX3* dx1
// #define C01_EX3    C001_EX3*(ONE - dx1) + C101_EX3* dx1
// #define C10_EX3    C010_EX3*(ONE - dx1) + C110_EX3* dx1
// #define C11_EX3    C011_EX3*(ONE - dx1) + C111_EX3* dx1
// #define C0_EX3     C00_EX3*(ONE - dx2) + C10_EX3* dx2
// #define C1_EX3     C01_EX3*(ONE - dx2) + C11_EX3* dx2
// #define EX3_INTERP C0_EX2*(ONE - dx3) + C1_EX2* dx3

// #define C000_BX1 \
//   INV_4*(BX1(i, j, k) + BX1(i, j - 1, k) + BX1(i, j, k - 1) + BX1(i, j - 1, k - 1))
// #define C100_BX1 \
//   INV_4*(BX1(i + 1, j, k) + BX1(i + 1, j - 1, k) + BX1(i + 1, j, k - 1) \
//          + BX1(i + 1, j - 1, k - 1))
// #define C001_BX1 \
//   INV_4*(BX1(i, j, k) + BX1(i, j, k + 1) + BX1(i, j - 1, k) + BX1(i, j - 1, k + 1))
// #define C101_BX1 \
//   INV_4*(BX1(i + 1, j, k) + BX1(i + 1, j, k + 1) + BX1(i + 1, j - 1, k) \
//          + BX1(i + 1, j - 1, k + 1))
// #define C010_BX1 \
//   INV_4*(BX1(i, j, k) + BX1(i, j + 1, k) + BX1(i, j, k - 1) + BX1(i, j + 1, k - 1))
// #define C110_BX1 \
//   INV_4*(BX1(i + 1, j, k) + BX1(i + 1, j, k - 1) + BX1(i + 1, j + 1, k - 1) \
//          + BX1(i + 1, j + 1, k))
// #define C011_BX1 \
//   INV_4*(BX1(i, j, k) + BX1(i, j + 1, k) + BX1(i, j + 1, k + 1) + BX1(i, j, k + 1))
// #define C111_BX1 \
//   INV_4*(BX1(i + 1, j, k) + BX1(i + 1, j + 1, k) + BX1(i + 1, j + 1, k + 1) \
//          + BX1(i + 1, j, k + 1))
// #define C00_BX1    C000_BX1*(ONE - dx1) + C100_BX1* dx1
// #define C01_BX1    C001_BX1*(ONE - dx1) + C101_BX1* dx1
// #define C10_BX1    C010_BX1*(ONE - dx1) + C110_BX1* dx1
// #define C11_BX1    C011_BX1*(ONE - dx1) + C111_BX1* dx1
// #define C0_BX1     C00_BX1*(ONE - dx2) + C10_BX1* dx2
// #define C1_BX1     C01_BX1*(ONE - dx2) + C11_BX1* dx2
// #define BX1_INTERP C0_BX1*(ONE - dx3) + C1_BX1* dx3

// #define C000_BX2 \
//   INV_4*(BX2(i - 1, j, k - 1) + BX2(i - 1, j, k) + BX2(i, j, k - 1) + BX2(i, j, k))
// #define C100_BX2 \
//   INV_4*(BX2(i, j, k - 1) + BX2(i, j, k) + BX2(i + 1, j, k - 1) + BX2(i + 1, j, k))
// #define C001_BX2 \
//   INV_4*(BX2(i - 1, j, k) + BX2(i - 1, j, k + 1) + BX2(i, j, k) + BX2(i, j, k + 1))
// #define C101_BX2 \
//   INV_4*(BX2(i, j, k) + BX2(i, j, k + 1) + BX2(i + 1, j, k) + BX2(i + 1, j, k + 1))
// #define C010_BX2 \
//   INV_4*(BX2(i - 1, j + 1, k - 1) + BX2(i - 1, j + 1, k) + BX2(i, j + 1, k - 1) \
//          + BX2(i, j + 1, k))
// #define C110_BX2 \
//   INV_4*(BX2(i, j + 1, k - 1) + BX2(i, j + 1, k) + BX2(i + 1, j + 1, k - 1) \
//          + BX2(i + 1, j + 1, k))
// #define C011_BX2 \
//   INV_4*(BX2(i - 1, j + 1, k) + BX2(i - 1, j + 1, k + 1) + BX2(i, j + 1, k) \
//          + BX2(i, j + 1, k + 1))
// #define C111_BX2 \
//   INV_4*(BX2(i, j + 1, k) + BX2(i, j + 1, k + 1) + BX2(i + 1, j + 1, k) \
//          + BX2(i + 1, j + 1, k + 1))
// #define C00_BX2    C000_BX2*(ONE - dx1) + C100_BX2* dx1
// #define C01_BX2    C001_BX2*(ONE - dx1) + C101_BX2* dx1
// #define C10_BX2    C010_BX2*(ONE - dx1) + C110_BX2* dx1
// #define C11_BX2    C011_BX2*(ONE - dx1) + C111_BX2* dx1
// #define C0_BX2     C00_BX2*(ONE - dx2) + C10_BX2* dx2
// #define C1_BX2     C01_BX2*(ONE - dx2) + C11_BX2* dx2
// #define BX2_INTERP C0_BX2*(ONE - dx3) + C1_BX2* dx3

// #define C000_BX3 \
//   INV_4*(BX3(i - 1, j - 1, k) + BX3(i - 1, j, k) + BX3(i, j - 1, k) + BX3(i, j, k))
// #define C100_BX3 \
//   INV_4*(BX3(i, j - 1, k) + BX3(i, j, k) + BX3(i + 1, j - 1, k) + BX3(i + 1, j, k))
// #define C001_BX3 \
//   INV_4*(BX3(i - 1, j - 1, k + 1) + BX3(i - 1, j, k + 1) + BX3(i, j - 1, k + 1) \
//          + BX3(i, j, k + 1))
// #define C101_BX3 \
//   INV_4*(BX3(i, j - 1, k + 1) + BX3(i, j, k + 1) + BX3(i + 1, j - 1, k + 1) \
//          + BX3(i + 1, j, k + 1))
// #define C010_BX3 \
//   INV_4*(BX3(i - 1, j, k) + BX3(i - 1, j + 1, k) + BX3(i, j, k) + BX3(i, j + 1, k))
// #define C110_BX3 \
//   INV_4*(BX3(i, j, k) + BX3(i, j + 1, k) + BX3(i + 1, j, k) + BX3(i + 1, j + 1, k))
// #define C011_BX3 \
//   INV_4*(BX3(i - 1, j, k + 1) + BX3(i - 1, j + 1, k + 1) + BX3(i, j, k + 1) \
//          + BX3(i, j + 1, k + 1))
// #define C111_BX3 \
//   INV_4*(BX3(i, j, k + 1) + BX3(i, j + 1, k + 1) + BX3(i + 1, j, k + 1) \
//          + BX3(i + 1, j + 1, k + 1))
// #define C00_BX3    C000_BX3*(ONE - dx1) + C100_BX3* dx1
// #define C01_BX3    C001_BX3*(ONE - dx1) + C101_BX3* dx1
// #define C10_BX3    C010_BX3*(ONE - dx1) + C110_BX3* dx1
// #define C11_BX3    C011_BX3*(ONE - dx1) + C111_BX3* dx1
// #define C0_BX3     C00_BX3*(ONE - dx2) + C10_BX3* dx2
// #define C1_BX3     C01_BX3*(ONE - dx2) + C11_BX3* dx2
// #define BX3_INTERP C0_BX3*(ONE - dx3) + C1_BX3* dx3

//   template <>
//   Inline void Pusher_kernel<Dim3>::interpolateFields(index_t&     p,
//                                                      vec_t<Dim3>& e0,
//                                                      vec_t<Dim3>& b0) const {
//     const auto i { m_particles.i1(p) + N_GHOSTS };
//     const auto dx1 { static_cast<real_t>(m_particles.dx1(p)) };
//     const auto j { m_particles.i2(p) + N_GHOSTS };
//     const auto dx2 { static_cast<real_t>(m_particles.dx2(p)) };
//     const auto k { m_particles.i3(p) + N_GHOSTS };
//     const auto dx3 { static_cast<real_t>(m_particles.dx3(p)) };
//     e0[0] = EX1_INTERP;
//     e0[1] = EX2_INTERP;
//     e0[2] = EX3_INTERP;
//     b0[0] = BX1_INTERP;
//     b0[1] = BX2_INTERP;
//     b0[2] = BX3_INTERP;
//   }

// #undef C000_EX1
// #undef C100_EX1
// #undef C010_EX1
// #undef C110_EX1
// #undef C001_EX1
// #undef C101_EX1
// #undef C011_EX1
// #undef C111_EX1
// #undef C00_EX1
// #undef C10_EX1
// #undef C01_EX1
// #undef C11_EX1
// #undef C0_EX1
// #undef C1_EX1
// #undef EX1_INTERP

// #undef C000_EX2
// #undef C100_EX2
// #undef C010_EX2
// #undef C110_EX2
// #undef C00_EX2
// #undef C10_EX2
// #undef C0_EX2
// #undef C001_EX2
// #undef C101_EX2
// #undef C011_EX2
// #undef C111_EX2
// #undef C01_EX2
// #undef C11_EX2
// #undef C1_EX2
// #undef EX2_INTERP

// #undef C000_EX3
// #undef C100_EX3
// #undef C010_EX3
// #undef C110_EX3
// #undef C001_EX3
// #undef C101_EX3
// #undef C011_EX3
// #undef C111_EX3
// #undef C00_EX3
// #undef C01_EX3
// #undef C10_EX3
// #undef C11_EX3
// #undef C0_EX3
// #undef C1_EX3
// #undef EX3_INTERP

// #undef C000_BX1
// #undef C100_BX1
// #undef C001_BX1
// #undef C101_BX1
// #undef C010_BX1
// #undef C110_BX1
// #undef C011_BX1
// #undef C111_BX1
// #undef C00_BX1
// #undef C01_BX1
// #undef C10_BX1
// #undef C11_BX1
// #undef C0_BX1
// #undef C1_BX1
// #undef BX1_INTERP

// #undef C000_BX2
// #undef C100_BX2
// #undef C001_BX2
// #undef C101_BX2
// #undef C010_BX2
// #undef C110_BX2
// #undef C011_BX2
// #undef C111_BX2
// #undef C00_BX2
// #undef C01_BX2
// #undef C10_BX2
// #undef C11_BX2
// #undef C0_BX2
// #undef C1_BX2
// #undef BX2_INTERP

// #undef C000_BX3
// #undef C100_BX3
// #undef C001_BX3
// #undef C101_BX3
// #undef C010_BX3
// #undef C110_BX3
// #undef C011_BX3
// #undef C111_BX3
// #undef C00_BX3
// #undef C01_BX3
// #undef C10_BX3
// #undef C11_BX3
// #undef C0_BX3
// #undef C1_BX3
// #undef BX3_INTERP
*/
#endif