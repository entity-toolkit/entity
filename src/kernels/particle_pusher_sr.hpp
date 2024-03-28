/**
 * @file particle_pusher_sr.h
 * @brief Particle pusher for the SR
 * @implements
 *   - ntt::PusherBase_kernel<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - arch/kokkos_aliases.h
 *   - utils/error.h
 *   - utils/log.h
 *   - utils/numeric.h
 * @namespaces:
 *   - ntt::
 */

#ifndef KERNELS_PARTICLE_PUSHER_SR_H
#define KERNELS_PARTICLE_PUSHER_SR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include <type_traits>

/* -------------------------------------------------------------------------- */
/* Local macros                                                               */
/* -------------------------------------------------------------------------- */
#define from_Xi_to_i(XI, I)                                                    \
  { I = static_cast<int>((XI)); }

#define from_Xi_to_i_di(XI, I, DI)                                             \
  {                                                                            \
    from_Xi_to_i((XI), (I));                                                   \
    DI = static_cast<prtldx_t>((XI)) - static_cast<prtldx_t>(I);               \
  }

/* -------------------------------------------------------------------------- */

namespace ntt {
  // Pushers
  struct Boris_t {};

  struct Vay_t {};

  struct Photon_t {};

  struct Boris_GCA_t {};

  struct Vay_GCA_t {};

  template <typename T>
  using Reduced_t = std::conditional_t<std::is_same_v<T, Boris_GCA_t>, Boris_t, Vay_t>;

  struct GCA_t {};

  struct Massive_t {};

  struct Massless_t {};

  template <typename T>
  using Mass_t = std::conditional_t<std::is_same_v<T, Photon_t>, Massless_t, Massive_t>;

  // Cooling
  struct NoCooling_t {};

  struct Synchrotron_t {};

  /**
   * @brief Algorithm for the Particle pusher.
   * @tparam D Dimension.
   * @tparam M Metric.
   * @tparam PG Problem generator.
   */
  template <Dimension D, class M, class PG>
  class PusherBase_kernel {
  protected:
    ndfield_t<D, 6>    EB;
    array_t<int*>      i1, i2, i3;
    array_t<int*>      i1_prev, i2_prev, i3_prev;
    array_t<prtldx_t*> dx1, dx2, dx3;
    array_t<prtldx_t*> dx1_prev, dx2_prev, dx3_prev;
    array_t<real_t*>   ux1, ux2, ux3;
    array_t<real_t*>   phi;
    array_t<short*>    tag;
    const M            metric;
    const PG           pgen;

    const real_t time, coeff, dt;
    const int    ni1, ni2, ni3;
    bool         is_absorb_i1min { false }, is_absorb_i1max { false };
    bool         is_absorb_i2min { false }, is_absorb_i2max { false };
    bool         is_absorb_i3min { false }, is_absorb_i3max { false };
    bool         is_periodic_i1min { false }, is_periodic_i1max { false };
    bool         is_periodic_i2min { false }, is_periodic_i2max { false };
    bool         is_periodic_i3min { false }, is_periodic_i3max { false };
    bool         is_axis_i2min { false }, is_axis_i2max { false };

  public:
    PusherBase_kernel(const ndfield_t<D, 6>&    EB,
                      const array_t<int*>&      i1,
                      const array_t<int*>&      i2,
                      const array_t<int*>&      i3,
                      const array_t<int*>&      i1_prev,
                      const array_t<int*>&      i2_prev,
                      const array_t<int*>&      i3_prev,
                      const array_t<prtldx_t*>& dx1,
                      const array_t<prtldx_t*>& dx2,
                      const array_t<prtldx_t*>& dx3,
                      const array_t<prtldx_t*>& dx1_prev,
                      const array_t<prtldx_t*>& dx2_prev,
                      const array_t<prtldx_t*>& dx3_prev,
                      const array_t<real_t*>&   ux1,
                      const array_t<real_t*>&   ux2,
                      const array_t<real_t*>&   ux3,
                      const array_t<real_t*>&   phi,
                      const array_t<short*>&    tag,
                      const M&                  metric,
                      const PG&                 pgen,
                      real_t                    time,
                      real_t                    coeff,
                      real_t                    dt,
                      int                       ni1,
                      int                       ni2,
                      int                       ni3,
                      const std::vector<std::vector<ParticleBC::type>>& boundaries) :
      EB { EB },
      i1 { i1 },
      i2 { i2 },
      i3 { i3 },
      i1_prev { i1_prev },
      i2_prev { i2_prev },
      i3_prev { i3_prev },
      dx1 { dx1 },
      dx2 { dx2 },
      dx3 { dx3 },
      dx1_prev { dx1_prev },
      dx2_prev { dx2_prev },
      dx3_prev { dx3_prev },
      ux1 { ux1 },
      ux2 { ux2 },
      ux3 { ux3 },
      phi { phi },
      tag { tag },
      metric { metric },
      pgen { pgen },
      time { time },
      coeff { coeff },
      dt { dt },
      ni1 { ni1 },
      ni2 { ni2 },
      ni3 { ni3 } {
      NTTHostErrorIf(boundaries.size() < 1, "boundaries defined incorrectly");
      is_absorb_i1min = (boundaries[0][0] == ParticleBC::ATMOSPHERE) ||
                        (boundaries[0][0] == ParticleBC::ABSORB);
      is_absorb_i1max = (boundaries[0][1] == ParticleBC::ATMOSPHERE) ||
                        (boundaries[0][1] == ParticleBC::ABSORB);
      is_periodic_i1min = (boundaries[0][0] == ParticleBC::PERIODIC);
      is_periodic_i1max = (boundaries[0][1] == ParticleBC::PERIODIC);
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        NTTHostErrorIf(boundaries.size() < 2, "boundaries defined incorrectly");
        is_absorb_i2min = (boundaries[1][0] == ParticleBC::ATMOSPHERE) ||
                          (boundaries[1][0] == ParticleBC::ABSORB);
        is_absorb_i2max = (boundaries[1][1] == ParticleBC::ATMOSPHERE) ||
                          (boundaries[1][1] == ParticleBC::ABSORB);
        is_periodic_i2min = (boundaries[1][0] == ParticleBC::PERIODIC);
        is_periodic_i2max = (boundaries[1][1] == ParticleBC::PERIODIC);
        is_axis_i2min     = (boundaries[1][0] == ParticleBC::AXIS);
        is_axis_i2max     = (boundaries[1][1] == ParticleBC::AXIS);
      }
      if constexpr (D == Dim::_3D) {
        NTTHostErrorIf(boundaries.size() < 3, "boundaries defined incorrectly");
        is_absorb_i3min = (boundaries[2][0] == ParticleBC::ATMOSPHERE) ||
                          (boundaries[2][0] == ParticleBC::ABSORB);
        is_absorb_i3max = (boundaries[2][1] == ParticleBC::ATMOSPHERE) ||
                          (boundaries[2][1] == ParticleBC::ABSORB);
        is_periodic_i3min = (boundaries[2][0] == ParticleBC::PERIODIC);
        is_periodic_i3max = (boundaries[2][1] == ParticleBC::PERIODIC);
      }
    }

    // Updaters

    /**
     * @brief update particle velocities
     * @param P pusher algorithm
     * @param p, e0, b0 index & interpolated fields
     */
    Inline void velUpd(Boris_t, index_t&, vec_t<Dim::_3D>&, vec_t<Dim::_3D>&) const;
    Inline void velUpd(Vay_t, index_t&, vec_t<Dim::_3D>&, vec_t<Dim::_3D>&) const;
    Inline void velUpd(GCA_t, index_t&, vec_t<Dim::_3D>&, vec_t<Dim::_3D>&) const;
    Inline void velUpd(GCA_t,
                       index_t&,
                       vec_t<Dim::_3D>&,
                       vec_t<Dim::_3D>&,
                       vec_t<Dim::_3D>&) const;

    // Getters
    Inline void getPrtlPos(index_t&, coord_t<M::PrtlD>&) const;
    Inline auto getEnergy(Massive_t, index_t& p) const -> real_t;
    Inline auto getEnergy(Massless_t, index_t& p) const -> real_t;
    Inline void getInterpFlds(index_t&, vec_t<Dim::_3D>&, vec_t<Dim::_3D>&) const;

    // Extra
    Inline void boundaryConditions(index_t&) const;
    Inline void initForce(coord_t<M::PrtlD>&, vec_t<Dim::_3D>&) const;
  };

  /**
   * @tparam D Dimension
   * @tparam M Metric
   * @tparam PG Problem generator
   * @tparam P Particle pusher
   * @tparam ExtForce External force toggle
   * @tparam Cs Cooling algorithms
   */
  template <Dimension D, class M, class PG, typename P, bool ExtForce, typename... Cs>
  struct Pusher_kernel : public PusherBase_kernel<D, M, PG> {
  private:
    // gca parameters
    const real_t gca_larmor, gca_EovrB_sqr;
    // synchrotron cooling parameters
    const real_t coeff_sync;

  public:
    Pusher_kernel(const ndfield_t<D, 6>&                             EB,
                  const array_t<int*>&                               i1,
                  const array_t<int*>&                               i2,
                  const array_t<int*>&                               i3,
                  const array_t<int*>&                               i1_prev,
                  const array_t<int*>&                               i2_prev,
                  const array_t<int*>&                               i3_prev,
                  const array_t<prtldx_t*>&                          dx1,
                  const array_t<prtldx_t*>&                          dx2,
                  const array_t<prtldx_t*>&                          dx3,
                  const array_t<prtldx_t*>&                          dx1_prev,
                  const array_t<prtldx_t*>&                          dx2_prev,
                  const array_t<prtldx_t*>&                          dx3_prev,
                  const array_t<real_t*>&                            ux1,
                  const array_t<real_t*>&                            ux2,
                  const array_t<real_t*>&                            ux3,
                  const array_t<real_t*>&                            phi,
                  const array_t<short*>&                             tag,
                  const M&                                           metric,
                  const PG&                                          pgen,
                  real_t                                             time,
                  real_t                                             coeff,
                  real_t                                             dt,
                  int                                                ni1,
                  int                                                ni2,
                  int                                                ni3,
                  const std::vector<std::vector<BoundaryCondition>>& boundaries,
                  real_t gca_larmor_max,
                  real_t gca_eovrb_max,
                  real_t coeff_sync) :
      PusherBase_kernel<D, M, PG>(EB,
                                  i1,
                                  i2,
                                  i3,
                                  i1_prev,
                                  i2_prev,
                                  i3_prev,
                                  dx1,
                                  dx2,
                                  dx3,
                                  dx1_prev,
                                  dx2_prev,
                                  dx3_prev,
                                  ux1,
                                  ux2,
                                  ux3,
                                  phi,
                                  tag,
                                  metric,
                                  pgen,
                                  time,
                                  coeff,
                                  dt,
                                  ni1,
                                  ni2,
                                  ni3,
                                  boundaries),
      gca_larmor { gca_larmor_max },
      gca_EovrB_sqr { SQR(gca_eovrb_max) },
      coeff_sync { coeff_sync } {}

    Inline void synchrotronDrag(index_t&               p,
                                vec_t<Dim::_3D>&       u_prime,
                                const vec_t<Dim::_3D>& e0,
                                const vec_t<Dim::_3D>& b0) const {
      real_t gamma_prime_sqr  = ONE / math::sqrt(ONE + NORM_SQR(u_prime[0],
                                                               u_prime[1],
                                                               u_prime[2]));
      u_prime[0]             *= gamma_prime_sqr;
      u_prime[1]             *= gamma_prime_sqr;
      u_prime[2]             *= gamma_prime_sqr;
      gamma_prime_sqr         = SQR(ONE / gamma_prime_sqr);
      const real_t beta_dot_e {
        DOT(u_prime[0], u_prime[1], u_prime[2], e0[0], e0[1], e0[2])
      };
      vec_t<Dim::_3D> e_plus_beta_cross_b {
        e0[0] + CROSS_x1(u_prime[0], u_prime[1], u_prime[2], b0[0], b0[1], b0[2]),
        e0[1] + CROSS_x2(u_prime[0], u_prime[1], u_prime[2], b0[0], b0[1], b0[2]),
        e0[2] + CROSS_x3(u_prime[0], u_prime[1], u_prime[2], b0[0], b0[1], b0[2])
      };
      vec_t<Dim::_3D> kappaR {
        CROSS_x1(e_plus_beta_cross_b[0],
                 e_plus_beta_cross_b[1],
                 e_plus_beta_cross_b[2],
                 b0[0],
                 b0[1],
                 b0[2]) +
          beta_dot_e * e0[0],
        CROSS_x2(e_plus_beta_cross_b[0],
                 e_plus_beta_cross_b[1],
                 e_plus_beta_cross_b[2],
                 b0[0],
                 b0[1],
                 b0[2]) +
          beta_dot_e * e0[1],
        CROSS_x3(e_plus_beta_cross_b[0],
                 e_plus_beta_cross_b[1],
                 e_plus_beta_cross_b[2],
                 b0[0],
                 b0[1],
                 b0[2]) +
          beta_dot_e * e0[2],
      };
      const real_t chiR_sqr { NORM_SQR(e_plus_beta_cross_b[0],
                                       e_plus_beta_cross_b[1],
                                       e_plus_beta_cross_b[2]) -
                              SQR(beta_dot_e) };
      this->ux1(p) += coeff_sync *
                      (kappaR[0] - gamma_prime_sqr * u_prime[0] * chiR_sqr);
      this->ux2(p) += coeff_sync *
                      (kappaR[1] - gamma_prime_sqr * u_prime[1] * chiR_sqr);
      this->ux3(p) += coeff_sync *
                      (kappaR[2] - gamma_prime_sqr * u_prime[2] * chiR_sqr);
    }

    Inline void operator()(P, index_t p) const {
      if (this->tag(p) == ParticleTag::alive) {
        coord_t<M::PrtlD> xp { ZERO };
        this->getPrtlPos(p, xp);
        // update cartesian velocity
        if constexpr (!std::is_same_v<P, Photon_t>) {
          // not a photon
          vec_t<Dim::_3D> ei { ZERO }, bi { ZERO };
          vec_t<Dim::_3D> ei_Cart { ZERO }, bi_Cart { ZERO };
          vec_t<Dim::_3D> force_Cart { ZERO };
          vec_t<Dim::_3D> u_prime { ZERO };
          vec_t<Dim::_3D> ei_Cart_rad { ZERO }, bi_Cart_rad { ZERO };
          bool            is_gca { false };

          this->getInterpFlds(p, ei, bi);
          this->metric.v3_Cntrv2Cart(xp, ei, ei_Cart);
          this->metric.v3_Cntrv2Cart(xp, bi, bi_Cart);
          if constexpr (!std::disjunction_v<std::is_same<NoCooling_t, Cs>...>) {
            // backup fields & velocities to use later in cooling
            ei_Cart_rad[0] = ei_Cart[0];
            ei_Cart_rad[1] = ei_Cart[1];
            ei_Cart_rad[2] = ei_Cart[2];
            bi_Cart_rad[0] = bi_Cart[0];
            bi_Cart_rad[1] = bi_Cart[1];
            bi_Cart_rad[2] = bi_Cart[2];
            u_prime[0]     = this->ux1(p);
            u_prime[1]     = this->ux2(p);
            u_prime[2]     = this->ux3(p);
          }
          if constexpr (ExtForce) {
            this->initForce(xp, force_Cart);
          }
          if constexpr (std::is_same_v<P, Boris_GCA_t> ||
                        std::is_same_v<P, Vay_GCA_t>) {
            const auto E2 { NORM_SQR(ei_Cart[0], ei_Cart[1], ei_Cart[2]) };
            const auto B2 { NORM_SQR(bi_Cart[0], bi_Cart[1], bi_Cart[2]) };
            const auto rL {
              math::sqrt(ONE + NORM_SQR(this->ux1(p), this->ux2(p), this->ux3(p))) *
              this->dt / (TWO * math::abs(this->coeff) * math::sqrt(B2))
            };
            if (B2 > ZERO && rL < gca_larmor && (E2 / B2) < gca_EovrB_sqr) {
              is_gca = true;
              // update with GCA
              if constexpr (ExtForce) {
                this->velUpd(GCA_t {}, p, force_Cart, ei_Cart, bi_Cart);
              } else {
                this->velUpd(GCA_t {}, p, ei_Cart, bi_Cart);
              }
            } else {
              // update with conventional pusher
              if constexpr (ExtForce) {
                this->ux1(p) += HALF * this->dt * force_Cart[0];
                this->ux2(p) += HALF * this->dt * force_Cart[1];
                this->ux3(p) += HALF * this->dt * force_Cart[2];
              }
              this->velUpd(Reduced_t<P> {}, p, ei_Cart, bi_Cart);
              if constexpr (ExtForce) {
                this->ux1(p) += HALF * this->dt * force_Cart[0];
                this->ux2(p) += HALF * this->dt * force_Cart[1];
                this->ux3(p) += HALF * this->dt * force_Cart[2];
              }
            }
          } else {
            // update with conventional pusher
            if constexpr (ExtForce) {
              this->ux1(p) += HALF * this->dt * force_Cart[0];
              this->ux2(p) += HALF * this->dt * force_Cart[1];
              this->ux3(p) += HALF * this->dt * force_Cart[2];
            }
            this->velUpd(P {}, p, ei_Cart, bi_Cart);
            if constexpr (ExtForce) {
              this->ux1(p) += HALF * this->dt * force_Cart[0];
              this->ux2(p) += HALF * this->dt * force_Cart[1];
              this->ux3(p) += HALF * this->dt * force_Cart[2];
            }
          }
          // cooling
          if constexpr (std::disjunction_v<std::is_same<Synchrotron_t, Cs>...>) {
            if (!is_gca) {
              u_prime[0] = HALF * (u_prime[0] + this->ux1(p));
              u_prime[1] = HALF * (u_prime[1] + this->ux2(p));
              u_prime[2] = HALF * (u_prime[2] + this->ux3(p));
              this->synchrotronDrag(p, u_prime, ei_Cart_rad, bi_Cart_rad);
            }
          }
        }
        // update position
        {
          // get cartesian velocity
          const real_t    inv_energy { ONE / this->getEnergy(Mass_t<P> {}, p) };
          vec_t<Dim::_3D> vp_Cart { this->ux1(p) * inv_energy,
                                    this->ux2(p) * inv_energy,
                                    this->ux3(p) * inv_energy };
          // get cartesian position
          coord_t<M::PrtlD> xp_Cart { ZERO };
          this->metric.x_Code2Cart(xp, xp_Cart);
          // update cartesian position
          for (auto d = 0u; d < M::PrtlD; ++d) {
            xp_Cart[d] += vp_Cart[d] * this->dt;
          }
          // transform back to code
          this->metric.x_Cart2Code(xp_Cart, xp);

          // update x1
          if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
            this->i1_prev(p)  = this->i1(p);
            this->dx1_prev(p) = this->dx1(p);
            from_Xi_to_i_di(xp[0], this->i1(p), this->dx1(p));
          }

          // update x2 & phi
          if constexpr (D == Dim::_2D || D == Dim::_3D) {
            this->i2_prev(p)  = this->i2(p);
            this->dx2_prev(p) = this->dx2(p);
            from_Xi_to_i_di(xp[1], this->i2(p), this->dx2(p));
            if constexpr (D == Dim::_2D && M::PrtlD == Dim::_3D) {
              this->phi(p) = xp[2];
            }
          }

          // update x3
          if constexpr (D == Dim::_3D) {
            this->i3_prev(p)  = this->i3(p);
            this->dx3_prev(p) = this->dx3(p);
            from_Xi_to_i_di(xp[2], this->i3(p), this->dx3(p));
          }
        }
        this->boundaryConditions(p);
      }
    }
  };

  // Velocity update

  template <Dimension D, class M, class PG>
  Inline void PusherBase_kernel<D, M, PG>::velUpd(Boris_t,
                                                  index_t&         p,
                                                  vec_t<Dim::_3D>& e0,
                                                  vec_t<Dim::_3D>& b0) const {
    real_t COEFF { coeff };

    e0[0] *= COEFF;
    e0[1] *= COEFF;
    e0[2] *= COEFF;
    vec_t<Dim::_3D> u0 { ux1(p) + e0[0], ux2(p) + e0[1], ux3(p) + e0[2] };

    COEFF *= ONE / math::sqrt(ONE + NORM_SQR(u0[0], u0[1], u0[2]));
    b0[0] *= COEFF;
    b0[1] *= COEFF;
    b0[2] *= COEFF;
    COEFF  = TWO / (ONE + NORM_SQR(b0[0], b0[1], b0[2]));

    vec_t<Dim::_3D> u1 {
      (u0[0] + CROSS_x1(u0[0], u0[1], u0[2], b0[0], b0[1], b0[2])) * COEFF,
      (u0[1] + CROSS_x2(u0[0], u0[1], u0[2], b0[0], b0[1], b0[2])) * COEFF,
      (u0[2] + CROSS_x3(u0[0], u0[1], u0[2], b0[0], b0[1], b0[2])) * COEFF
    };

    u0[0] += CROSS_x1(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) + e0[0];
    u0[1] += CROSS_x2(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) + e0[1];
    u0[2] += CROSS_x3(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) + e0[2];

    ux1(p) = u0[0];
    ux2(p) = u0[1];
    ux3(p) = u0[2];
  }

  template <Dimension D, class M, class PG>
  Inline void PusherBase_kernel<D, M, PG>::velUpd(Vay_t,
                                                  index_t&         p,
                                                  vec_t<Dim::_3D>& e0,
                                                  vec_t<Dim::_3D>& b0) const {
    auto COEFF { coeff };
    e0[0] *= COEFF;
    e0[1] *= COEFF;
    e0[2] *= COEFF;

    b0[0] *= COEFF;
    b0[1] *= COEFF;
    b0[2] *= COEFF;

    COEFF = ONE / math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p)));

    vec_t<Dim::_3D> u1 {
      (ux1(p) + TWO * e0[0] +
       CROSS_x1(ux1(p), ux2(p), ux3(p), b0[0], b0[1], b0[2]) * COEFF),
      (ux2(p) + TWO * e0[1] +
       CROSS_x2(ux1(p), ux2(p), ux3(p), b0[0], b0[1], b0[2]) * COEFF),
      (ux3(p) + TWO * e0[2] +
       CROSS_x3(ux1(p), ux2(p), ux3(p), b0[0], b0[1], b0[2]) * COEFF)
    };
    COEFF = DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]);
    auto COEFF2 { ONE + NORM_SQR(u1[0], u1[1], u1[2]) -
                  NORM_SQR(b0[0], b0[1], b0[2]) };

    COEFF = ONE /
            math::sqrt(
              INV_2 * (COEFF2 + math::sqrt(SQR(COEFF2) +
                                           FOUR * (SQR(b0[0]) + SQR(b0[1]) +
                                                   SQR(b0[2]) + SQR(COEFF)))));
    COEFF2 = ONE /
             (ONE + SQR(b0[0] * COEFF) + SQR(b0[1] * COEFF) + SQR(b0[2] * COEFF));

    ux1(p) = COEFF2 * (u1[0] +
                       COEFF * DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) *
                         (b0[0] * COEFF) +
                       u1[1] * b0[2] * COEFF - u1[2] * b0[1] * COEFF);
    ux2(p) = COEFF2 * (u1[1] +
                       COEFF * DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) *
                         (b0[1] * COEFF) +
                       u1[2] * b0[0] * COEFF - u1[0] * b0[2] * COEFF);
    ux3(p) = COEFF2 * (u1[2] +
                       COEFF * DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) *
                         (b0[2] * COEFF) +
                       u1[0] * b0[1] * COEFF - u1[1] * b0[0] * COEFF);
  }

  template <Dimension D, class M, class PG>
  Inline void PusherBase_kernel<D, M, PG>::velUpd(GCA_t,
                                                  index_t&         p,
                                                  vec_t<Dim::_3D>& f0,
                                                  vec_t<Dim::_3D>& e0,
                                                  vec_t<Dim::_3D>& b0) const {
    const auto eb_sqr { NORM_SQR(e0[0], e0[1], e0[2]) +
                        NORM_SQR(b0[0], b0[1], b0[2]) };

    const vec_t<Dim::_3D> wE {
      CROSS_x1(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr,
      CROSS_x2(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr,
      CROSS_x3(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr
    };

    {
      const auto b_norm_inv { ONE / NORM(b0[0], b0[1], b0[2]) };
      b0[0] *= b_norm_inv;
      b0[1] *= b_norm_inv;
      b0[2] *= b_norm_inv;
    }
    auto upar { DOT(ux1(p), ux2(p), ux3(p), b0[0], b0[1], b0[2]) +
                coeff * TWO * DOT(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) +
                dt * DOT(f0[0], f0[1], f0[2], b0[0], b0[1], b0[2]) };

    real_t factor;
    {
      const auto wE_sqr { NORM_SQR(wE[0], wE[1], wE[2]) };
      if (wE_sqr < static_cast<real_t>(0.01)) {
        factor = ONE + wE_sqr + TWO * SQR(wE_sqr) + FIVE * SQR(wE_sqr) * wE_sqr;
      } else {
        factor = (ONE - math::sqrt(ONE - FOUR * wE_sqr)) / (TWO * wE_sqr);
      }
    }
    const vec_t<Dim::_3D> vE_Cart { wE[0] * factor, wE[1] * factor, wE[2] * factor };
    const auto Gamma { math::sqrt(ONE + SQR(upar)) /
                       math::sqrt(
                         ONE - NORM_SQR(vE_Cart[0], vE_Cart[1], vE_Cart[2])) };
    ux1(p) = upar * b0[0] + vE_Cart[0] * Gamma;
    ux2(p) = upar * b0[1] + vE_Cart[1] * Gamma;
    ux3(p) = upar * b0[2] + vE_Cart[2] * Gamma;
  }

  template <Dimension D, class M, class PG>
  Inline void PusherBase_kernel<D, M, PG>::velUpd(GCA_t,
                                                  index_t&         p,
                                                  vec_t<Dim::_3D>& e0,
                                                  vec_t<Dim::_3D>& b0) const {
    const auto eb_sqr { NORM_SQR(e0[0], e0[1], e0[2]) +
                        NORM_SQR(b0[0], b0[1], b0[2]) };

    const vec_t<Dim::_3D> wE {
      CROSS_x1(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr,
      CROSS_x2(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr,
      CROSS_x3(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr
    };

    {
      const auto b_norm_inv { ONE / NORM(b0[0], b0[1], b0[2]) };
      b0[0] *= b_norm_inv;
      b0[1] *= b_norm_inv;
      b0[2] *= b_norm_inv;
    }
    auto upar { DOT(ux1(p), ux2(p), ux3(p), b0[0], b0[1], b0[2]) +
                coeff * TWO * DOT(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) };

    real_t factor;
    {
      const auto wE_sqr { NORM_SQR(wE[0], wE[1], wE[2]) };
      if (wE_sqr < static_cast<real_t>(0.01)) {
        factor = ONE + wE_sqr + TWO * SQR(wE_sqr) + FIVE * SQR(wE_sqr) * wE_sqr;
      } else {
        factor = (ONE - math::sqrt(ONE - FOUR * wE_sqr)) / (TWO * wE_sqr);
      }
    }
    const vec_t<Dim::_3D> vE_Cart { wE[0] * factor, wE[1] * factor, wE[2] * factor };
    const auto Gamma { math::sqrt(ONE + SQR(upar)) /
                       math::sqrt(
                         ONE - NORM_SQR(vE_Cart[0], vE_Cart[1], vE_Cart[2])) };
    ux1(p) = upar * b0[0] + vE_Cart[0] * Gamma;
    ux2(p) = upar * b0[1] + vE_Cart[1] * Gamma;
    ux3(p) = upar * b0[2] + vE_Cart[2] * Gamma;
  }

  template <Dimension D, class M, class PG>
  Inline void PusherBase_kernel<D, M, PG>::getPrtlPos(index_t& p,
                                                      coord_t<M::PrtlD>& xp) const {
    if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
      xp[0] = i_di_to_Xi(i1(p), dx1(p));
    }
    if constexpr (D == Dim::_2D) {
      xp[1] = i_di_to_Xi(i2(p), dx2(p));
      if constexpr (M::PrtlD == Dim::_3D) {
        xp[2] = phi(p);
      }
    }
    if constexpr (D == Dim::_3D) {
      xp[1] = i_di_to_Xi(i2(p), dx2(p));
      xp[2] = i_di_to_Xi(i3(p), dx3(p));
    }
  }

  template <Dimension D, class M, class PG>
  Inline auto PusherBase_kernel<D, M, PG>::getEnergy(Massive_t,
                                                     index_t& p) const -> real_t {
    return math::sqrt(ONE + SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p)));
  }

  template <Dimension D, class M, class PG>
  Inline auto PusherBase_kernel<D, M, PG>::getEnergy(Massless_t,
                                                     index_t& p) const -> real_t {
    return math::sqrt(SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p)));
  }

  template <Dimension D, class M, class PG>
  Inline void PusherBase_kernel<D, M, PG>::getInterpFlds(index_t&         p,
                                                         vec_t<Dim::_3D>& e0,
                                                         vec_t<Dim::_3D>& b0) const {
    if constexpr (D == Dim::_1D) {
      const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
      const auto dx1_ { static_cast<real_t>(dx1(p)) };

      // first order
      real_t c0, c1;

      // Ex1
      // interpolate to nodes
      c0    = HALF * (EB(i, em::ex1) + EB(i - 1, em::ex1));
      c1    = HALF * (EB(i, em::ex1) + EB(i + 1, em::ex1));
      // interpolate from nodes to the particle position
      e0[0] = c0 * (ONE - dx1_) + c1 * dx1_;
      // Ex2
      c0    = EB(i, em::ex2);
      c1    = EB(i + 1, em::ex2);
      e0[1] = c0 * (ONE - dx1_) + c1 * dx1_;
      // Ex3
      c0    = EB(i, em::ex3);
      c1    = EB(i + 1, em::ex3);
      e0[2] = c0 * (ONE - dx1_) + c1 * dx1_;

      // Bx1
      c0    = EB(i, em::bx1);
      c1    = EB(i + 1, em::bx1);
      b0[0] = c0 * (ONE - dx1_) + c1 * dx1_;
      // Bx2
      c0    = HALF * (EB(i - 1, em::bx2) + EB(i, em::bx2));
      c1    = HALF * (EB(i, em::bx2) + EB(i + 1, em::bx2));
      b0[1] = c0 * (ONE - dx1_) + c1 * dx1_;
      // Bx3
      c0    = HALF * (EB(i - 1, em::bx3) + EB(i, em::bx3));
      c1    = HALF * (EB(i, em::bx3) + EB(i + 1, em::bx3));
      b0[2] = c0 * (ONE - dx1_) + c1 * dx1_;
    } else if constexpr (D == Dim::_2D) {
      const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
      const int  j { i2(p) + static_cast<int>(N_GHOSTS) };
      const auto dx1_ { static_cast<real_t>(dx1(p)) };
      const auto dx2_ { static_cast<real_t>(dx2(p)) };

      // first order
      real_t c000, c100, c010, c110, c00, c10;

      // Ex1
      // interpolate to nodes
      c000  = HALF * (EB(i, j, em::ex1) + EB(i - 1, j, em::ex1));
      c100  = HALF * (EB(i, j, em::ex1) + EB(i + 1, j, em::ex1));
      c010  = HALF * (EB(i, j + 1, em::ex1) + EB(i - 1, j + 1, em::ex1));
      c110  = HALF * (EB(i, j + 1, em::ex1) + EB(i + 1, j + 1, em::ex1));
      // interpolate from nodes to the particle position
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      e0[0] = c00 * (ONE - dx2_) + c10 * dx2_;
      // Ex2
      c000  = HALF * (EB(i, j, em::ex2) + EB(i, j - 1, em::ex2));
      c100  = HALF * (EB(i + 1, j, em::ex2) + EB(i + 1, j - 1, em::ex2));
      c010  = HALF * (EB(i, j, em::ex2) + EB(i, j + 1, em::ex2));
      c110  = HALF * (EB(i + 1, j, em::ex2) + EB(i + 1, j + 1, em::ex2));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      e0[1] = c00 * (ONE - dx2_) + c10 * dx2_;
      // Ex3
      c000  = EB(i, j, em::ex3);
      c100  = EB(i + 1, j, em::ex3);
      c010  = EB(i, j + 1, em::ex3);
      c110  = EB(i + 1, j + 1, em::ex3);
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      e0[2] = c00 * (ONE - dx2_) + c10 * dx2_;

      // Bx1
      c000  = HALF * (EB(i, j, em::bx1) + EB(i, j - 1, em::bx1));
      c100  = HALF * (EB(i + 1, j, em::bx1) + EB(i + 1, j - 1, em::bx1));
      c010  = HALF * (EB(i, j, em::bx1) + EB(i, j + 1, em::bx1));
      c110  = HALF * (EB(i + 1, j, em::bx1) + EB(i + 1, j + 1, em::bx1));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      b0[0] = c00 * (ONE - dx2_) + c10 * dx2_;
      // Bx2
      c000  = HALF * (EB(i - 1, j, em::bx2) + EB(i, j, em::bx2));
      c100  = HALF * (EB(i, j, em::bx2) + EB(i + 1, j, em::bx2));
      c010  = HALF * (EB(i - 1, j + 1, em::bx2) + EB(i, j + 1, em::bx2));
      c110  = HALF * (EB(i, j + 1, em::bx2) + EB(i + 1, j + 1, em::bx2));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      b0[1] = c00 * (ONE - dx2_) + c10 * dx2_;
      // Bx3
      c000  = INV_4 * (EB(i - 1, j - 1, em::bx3) + EB(i - 1, j, em::bx3) +
                      EB(i, j - 1, em::bx3) + EB(i, j, em::bx3));
      c100  = INV_4 * (EB(i, j - 1, em::bx3) + EB(i, j, em::bx3) +
                      EB(i + 1, j - 1, em::bx3) + EB(i + 1, j, em::bx3));
      c010  = INV_4 * (EB(i - 1, j, em::bx3) + EB(i - 1, j + 1, em::bx3) +
                      EB(i, j, em::bx3) + EB(i, j + 1, em::bx3));
      c110  = INV_4 * (EB(i, j, em::bx3) + EB(i, j + 1, em::bx3) +
                      EB(i + 1, j, em::bx3) + EB(i + 1, j + 1, em::bx3));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      b0[2] = c00 * (ONE - dx2_) + c10 * dx2_;
    } else if constexpr (D == Dim::_3D) {
      const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
      const int  j { i2(p) + static_cast<int>(N_GHOSTS) };
      const int  k { i3(p) + static_cast<int>(N_GHOSTS) };
      const auto dx1_ { static_cast<real_t>(dx1(p)) };
      const auto dx2_ { static_cast<real_t>(dx2(p)) };
      const auto dx3_ { static_cast<real_t>(dx3(p)) };

      // first order
      real_t c000, c100, c010, c110, c001, c101, c011, c111, c00, c10, c01, c11,
        c0, c1;

      // Ex1
      // interpolate to nodes
      c000 = HALF * (EB(i, j, k, em::ex1) + EB(i - 1, j, k, em::ex1));
      c100 = HALF * (EB(i, j, k, em::ex1) + EB(i + 1, j, k, em::ex1));
      c010 = HALF * (EB(i, j + 1, k, em::ex1) + EB(i - 1, j + 1, k, em::ex1));
      c110 = HALF * (EB(i, j + 1, k, em::ex1) + EB(i + 1, j + 1, k, em::ex1));
      // interpolate from nodes to the particle position
      c00  = c000 * (ONE - dx1_) + c100 * dx1_;
      c10  = c010 * (ONE - dx1_) + c110 * dx1_;
      c0   = c00 * (ONE - dx2_) + c10 * dx2_;
      // interpolate to nodes
      c001 = HALF * (EB(i, j, k + 1, em::ex1) + EB(i - 1, j, k + 1, em::ex1));
      c101 = HALF * (EB(i, j, k + 1, em::ex1) + EB(i + 1, j, k + 1, em::ex1));
      c011 = HALF *
             (EB(i, j + 1, k + 1, em::ex1) + EB(i - 1, j + 1, k + 1, em::ex1));
      c111 = HALF *
             (EB(i, j + 1, k + 1, em::ex1) + EB(i + 1, j + 1, k + 1, em::ex1));
      // interpolate from nodes to the particle position
      c01   = c001 * (ONE - dx1_) + c101 * dx1_;
      c11   = c011 * (ONE - dx1_) + c111 * dx1_;
      c1    = c01 * (ONE - dx2_) + c11 * dx2_;
      e0[0] = c0 * (ONE - dx3_) + c1 * dx3_;

      // Ex2
      c000 = HALF * (EB(i, j, k, em::ex2) + EB(i, j - 1, k, em::ex2));
      c100 = HALF * (EB(i + 1, j, k, em::ex2) + EB(i + 1, j - 1, k, em::ex2));
      c010 = HALF * (EB(i, j, k, em::ex2) + EB(i, j + 1, k, em::ex2));
      c110 = HALF * (EB(i + 1, j, k, em::ex2) + EB(i + 1, j + 1, k, em::ex2));
      c00  = c000 * (ONE - dx1_) + c100 * dx1_;
      c10  = c010 * (ONE - dx1_) + c110 * dx1_;
      c0   = c00 * (ONE - dx2_) + c10 * dx2_;
      c001 = HALF * (EB(i, j, k + 1, em::ex2) + EB(i, j - 1, k + 1, em::ex2));
      c101 = HALF *
             (EB(i + 1, j, k + 1, em::ex2) + EB(i + 1, j - 1, k + 1, em::ex2));
      c011 = HALF * (EB(i, j, k + 1, em::ex2) + EB(i, j + 1, k + 1, em::ex2));
      c111 = HALF *
             (EB(i + 1, j, k + 1, em::ex2) + EB(i + 1, j + 1, k + 1, em::ex2));
      c01   = c001 * (ONE - dx1_) + c101 * dx1_;
      c11   = c011 * (ONE - dx1_) + c111 * dx1_;
      c1    = c01 * (ONE - dx2_) + c11 * dx2_;
      e0[1] = c0 * (ONE - dx3_) + c1 * dx3_;

      // Ex3
      c000 = HALF * (EB(i, j, k, em::ex3) + EB(i, j, k - 1, em::ex3));
      c100 = HALF * (EB(i + 1, j, k, em::ex3) + EB(i + 1, j, k - 1, em::ex3));
      c010 = HALF * (EB(i, j + 1, k, em::ex3) + EB(i, j + 1, k - 1, em::ex3));
      c110 = HALF *
             (EB(i + 1, j + 1, k, em::ex3) + EB(i + 1, j + 1, k - 1, em::ex3));
      c001 = HALF * (EB(i, j, k, em::ex3) + EB(i, j, k + 1, em::ex3));
      c101 = HALF * (EB(i + 1, j, k, em::ex3) + EB(i + 1, j, k + 1, em::ex3));
      c011 = HALF * (EB(i, j + 1, k, em::ex3) + EB(i, j + 1, k + 1, em::ex3));
      c111 = HALF *
             (EB(i + 1, j + 1, k, em::ex3) + EB(i + 1, j + 1, k + 1, em::ex3));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c01   = c001 * (ONE - dx1_) + c101 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      c11   = c011 * (ONE - dx1_) + c111 * dx1_;
      c0    = c00 * (ONE - dx2_) + c10 * dx2_;
      c1    = c01 * (ONE - dx2_) + c11 * dx2_;
      e0[2] = c0 * (ONE - dx3_) + c1 * dx3_;

      // Bx1
      c000 = INV_4 * (EB(i, j, k, em::bx1) + EB(i, j - 1, k, em::bx1) +
                      EB(i, j, k - 1, em::bx1) + EB(i, j - 1, k - 1, em::bx1));
      c100 = INV_4 *
             (EB(i + 1, j, k, em::bx1) + EB(i + 1, j - 1, k, em::bx1) +
              EB(i + 1, j, k - 1, em::bx1) + EB(i + 1, j - 1, k - 1, em::bx1));
      c001 = INV_4 * (EB(i, j, k, em::bx1) + EB(i, j, k + 1, em::bx1) +
                      EB(i, j - 1, k, em::bx1) + EB(i, j - 1, k + 1, em::bx1));
      c101 = INV_4 *
             (EB(i + 1, j, k, em::bx1) + EB(i + 1, j, k + 1, em::bx1) +
              EB(i + 1, j - 1, k, em::bx1) + EB(i + 1, j - 1, k + 1, em::bx1));
      c010 = INV_4 * (EB(i, j, k, em::bx1) + EB(i, j + 1, k, em::bx1) +
                      EB(i, j, k - 1, em::bx1) + EB(i, j + 1, k - 1, em::bx1));
      c110 = INV_4 *
             (EB(i + 1, j, k, em::bx1) + EB(i + 1, j, k - 1, em::bx1) +
              EB(i + 1, j + 1, k - 1, em::bx1) + EB(i + 1, j + 1, k, em::bx1));
      c011 = INV_4 * (EB(i, j, k, em::bx1) + EB(i, j + 1, k, em::bx1) +
                      EB(i, j + 1, k + 1, em::bx1) + EB(i, j, k + 1, em::bx1));
      c111 = INV_4 *
             (EB(i + 1, j, k, em::bx1) + EB(i + 1, j + 1, k, em::bx1) +
              EB(i + 1, j + 1, k + 1, em::bx1) + EB(i + 1, j, k + 1, em::bx1));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c01   = c001 * (ONE - dx1_) + c101 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      c11   = c011 * (ONE - dx1_) + c111 * dx1_;
      c0    = c00 * (ONE - dx2_) + c10 * dx2_;
      c1    = c01 * (ONE - dx2_) + c11 * dx2_;
      b0[0] = c0 * (ONE - dx3_) + c1 * dx3_;

      // Bx2
      c000 = INV_4 * (EB(i - 1, j, k - 1, em::bx2) + EB(i - 1, j, k, em::bx2) +
                      EB(i, j, k - 1, em::bx2) + EB(i, j, k, em::bx2));
      c100 = INV_4 * (EB(i, j, k - 1, em::bx2) + EB(i, j, k, em::bx2) +
                      EB(i + 1, j, k - 1, em::bx2) + EB(i + 1, j, k, em::bx2));
      c001 = INV_4 * (EB(i - 1, j, k, em::bx2) + EB(i - 1, j, k + 1, em::bx2) +
                      EB(i, j, k, em::bx2) + EB(i, j, k + 1, em::bx2));
      c101 = INV_4 * (EB(i, j, k, em::bx2) + EB(i, j, k + 1, em::bx2) +
                      EB(i + 1, j, k, em::bx2) + EB(i + 1, j, k + 1, em::bx2));
      c010 = INV_4 *
             (EB(i - 1, j + 1, k - 1, em::bx2) + EB(i - 1, j + 1, k, em::bx2) +
              EB(i, j + 1, k - 1, em::bx2) + EB(i, j + 1, k, em::bx2));
      c110 = INV_4 *
             (EB(i, j + 1, k - 1, em::bx2) + EB(i, j + 1, k, em::bx2) +
              EB(i + 1, j + 1, k - 1, em::bx2) + EB(i + 1, j + 1, k, em::bx2));
      c011 = INV_4 *
             (EB(i - 1, j + 1, k, em::bx2) + EB(i - 1, j + 1, k + 1, em::bx2) +
              EB(i, j + 1, k, em::bx2) + EB(i, j + 1, k + 1, em::bx2));
      c111 = INV_4 *
             (EB(i, j + 1, k, em::bx2) + EB(i, j + 1, k + 1, em::bx2) +
              EB(i + 1, j + 1, k, em::bx2) + EB(i + 1, j + 1, k + 1, em::bx2));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c01   = c001 * (ONE - dx1_) + c101 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      c11   = c011 * (ONE - dx1_) + c111 * dx1_;
      c0    = c00 * (ONE - dx2_) + c10 * dx2_;
      c1    = c01 * (ONE - dx2_) + c11 * dx2_;
      b0[1] = c0 * (ONE - dx3_) + c1 * dx3_;

      // Bx3
      c000 = INV_4 * (EB(i - 1, j - 1, k, em::bx3) + EB(i - 1, j, k, em::bx3) +
                      EB(i, j - 1, k, em::bx3) + EB(i, j, k, em::bx3));
      c100 = INV_4 * (EB(i, j - 1, k, em::bx3) + EB(i, j, k, em::bx3) +
                      EB(i + 1, j - 1, k, em::bx3) + EB(i + 1, j, k, em::bx3));
      c001 = INV_4 *
             (EB(i - 1, j - 1, k + 1, em::bx3) + EB(i - 1, j, k + 1, em::bx3) +
              EB(i, j - 1, k + 1, em::bx3) + EB(i, j, k + 1, em::bx3));
      c101 = INV_4 *
             (EB(i, j - 1, k + 1, em::bx3) + EB(i, j, k + 1, em::bx3) +
              EB(i + 1, j - 1, k + 1, em::bx3) + EB(i + 1, j, k + 1, em::bx3));
      c010 = INV_4 * (EB(i - 1, j, k, em::bx3) + EB(i - 1, j + 1, k, em::bx3) +
                      EB(i, j, k, em::bx3) + EB(i, j + 1, k, em::bx3));
      c110 = INV_4 * (EB(i, j, k, em::bx3) + EB(i, j + 1, k, em::bx3) +
                      EB(i + 1, j, k, em::bx3) + EB(i + 1, j + 1, k, em::bx3));
      c011 = INV_4 *
             (EB(i - 1, j, k + 1, em::bx3) + EB(i - 1, j + 1, k + 1, em::bx3) +
              EB(i, j, k + 1, em::bx3) + EB(i, j + 1, k + 1, em::bx3));
      c111 = INV_4 *
             (EB(i, j, k + 1, em::bx3) + EB(i, j + 1, k + 1, em::bx3) +
              EB(i + 1, j, k + 1, em::bx3) + EB(i + 1, j + 1, k + 1, em::bx3));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c01   = c001 * (ONE - dx1_) + c101 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      c11   = c011 * (ONE - dx1_) + c111 * dx1_;
      c0    = c00 * (ONE - dx2_) + c10 * dx2_;
      c1    = c01 * (ONE - dx2_) + c11 * dx2_;
      b0[2] = c0 * (ONE - dx3_) + c1 * dx3_;
    }
  }

  // Boundary conditions

  template <Dimension D, class M, class PG>
  Inline void PusherBase_kernel<D, M, PG>::boundaryConditions(index_t& p) const {
    if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
      if (i1(p) < 0) {
        if (is_periodic_i1min) {
          i1(p)      += ni1;
          i1_prev(p) += ni1;
        } else if (is_absorb_i1min) {
          tag(p) = ParticleTag::dead;
        }
      } else if (i1(p) >= ni1) {
        if (is_periodic_i1max) {
          i1(p)      -= ni1;
          i1_prev(p) -= ni1;
        } else if (is_absorb_i1max) {
          tag(p) = ParticleTag::dead;
        }
      }
    }
    if constexpr (D == Dim::_2D || D == Dim::_3D) {
      if (i2(p) < 0) {
        if (is_periodic_i2min) {
          i2(p)      += ni2;
          i2_prev(p) += ni2;
        } else if (is_absorb_i2min) {
          tag(p) = ParticleTag::dead;
        } else if (is_axis_i2min) {
          i2(p)  = 0;
          dx2(p) = ONE - dx2(p);
        }
      } else if (i2(p) >= ni2) {
        if (is_periodic_i2max) {
          i2(p)      -= ni2;
          i2_prev(p) -= ni2;
        } else if (is_absorb_i2max) {
          tag(p) = ParticleTag::dead;
        } else if (is_axis_i2max) {
          i2(p)  = ni2 - 1;
          dx2(p) = ONE - dx2(p);
        }
      }
    }
    if constexpr (D == Dim::_3D) {
      if (i3(p) < 0) {
        if (is_periodic_i3min) {
          i3(p)      += ni3;
          i3_prev(p) += ni3;
        } else if (is_absorb_i3min) {
          tag(p) = ParticleTag::dead;
        }
      } else if (i3(p) >= ni3) {
        if (is_periodic_i3max) {
          i3(p)      -= ni3;
          i3_prev(p) -= ni3;
        } else if (is_absorb_i3max) {
          tag(p) = ParticleTag::dead;
        }
      }
    }
  }

  // External force

  template <Dimension D, class M, class PG>
  Inline void PusherBase_kernel<D, M, PG>::initForce(
    coord_t<M::PrtlD>& xp,
    vec_t<Dim::_3D>&   force_Cart) const {
    coord_t<M::PrtlD> xp_Ph { ZERO };
    xp_Ph[0] = metric.x1_Code2Phys(xp[0]);
    if constexpr (M::PrtlD != Dim::_1D) {
      xp_Ph[1] = metric.x2_Code2Phys(xp[1]);
    }
    if constexpr (M::PrtlD == Dim::_3D) {
      xp_Ph[2] = metric.x3_Code2Phys(xp[2]);
    }
    const vec_t<Dim::_3D> force_Hat { pgen.ext_force_x1(time, xp_Ph),
                                      pgen.ext_force_x2(time, xp_Ph),
                                      pgen.ext_force_x3(time, xp_Ph) };
    metric.v3_Hat2Cart(xp, force_Hat, force_Cart);
  }

} // namespace ntt

#undef from_Xi_to_i_di
#undef from_Xi_to_i

#endif // KERNELS_PARTICLE_PUSHER_SR_H
