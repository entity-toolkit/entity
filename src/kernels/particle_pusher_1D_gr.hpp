/**
 * @file kernels/particle_pusher_1D_gr.h
 * @brief Implementation of the particle pusher for 1D GR
 * @implements
 *   - kernel::gr::Pusher_kernel_1D<>
 * @namespaces:
 *   - kernel::gr::
 * @macros:
 *   - MPI_ENABLED
*/

#ifndef KERNELS_PARTICLE_PUSHER_1D_GR_HPP
#define KERNELS_PARTICLE_PUSHER_1D_GR_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_tags.h"
#endif

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

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

#define DERIVATIVE(func, x)                                               \
  ((func({ x + epsilon }) - func({ x - epsilon })) /         \
   (TWO * epsilon))


/* -------------------------------------------------------------------------- */

namespace kernel::gr {
  using namespace ntt;

  struct Massive_t {};

  struct Massless_t {};

  /**
   * @brief Algorithm for the Particle pusher
   * @tparam M Metric
   */
  template <class M>
  class Pusher_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;
    static_assert(D == Dim::_1D, "Only 1d available for this pusher");

    const ndfield_t<D, 1> Dfield;
    array_t<int*>         i1;
    array_t<int*>         i1_prev;
    array_t<prtldx_t*>    dx1;
    array_t<prtldx_t*>    dx1_prev;
    array_t<real_t*>      ux1, ux2, ux3;
    array_t<short*>       tag;
    const M               metric;

    const real_t coeff, dt;
    const int    ni1;
    const real_t epsilon;
    const int    niter;
    const int    i1_absorb;

    bool is_absorb_i1min { false }, is_absorb_i1max { false };

  public:
    Pusher_kernel(const ndfield_t<D, 1>&            Dfield,
                  const array_t<int*>&              i1,
                  const array_t<int*>&              i1_prev,
                  const array_t<prtldx_t*>&         dx1,
                  const array_t<prtldx_t*>&         dx1_prev,
                  const array_t<real_t*>&           ux1,
                  const array_t<real_t*>&           ux2,
                  const array_t<real_t*>&           ux3,
                  const array_t<short*>&            tag,
                  const M&                          metric,
                  const real_t&                     coeff,
                  const real_t&                     dt,
                  const int&                        ni1,
                  const real_t&                     epsilon,
                  const int&                        niter,
                  const boundaries_t<PrtlBC>& boundaries)
      : Dfield { Dfield }
      , i1 { i1 }
      , i1_prev { i1_prev }
      , dx1 { dx1 }
      , dx1_prev { dx1_prev }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , metric { metric }
      , coeff { coeff }
      , dt { dt }
      , ni1 { ni1 }
      , epsilon { epsilon }
      , niter { niter }
      , i1_absorb { 2 } {

      raise::ErrorIf(boundaries.size() < 1, "boundaries defined incorrectly", HERE);
      is_absorb_i1min = (boundaries[0].first == PrtlBC::ABSORB) ||
                        (boundaries[0].first == PrtlBC::HORIZON);
      is_absorb_i1max = (boundaries[0].second == PrtlBC::ABSORB) ||
                        (boundaries[0].second == PrtlBC::HORIZON);
    }

    /**
     * @brief Main pusher subroutine for photon particles.
     */
    //Inline void operator()(Massless_t, index_t) const;

    /**
     * @brief Main pusher subroutine for massive particles.
     */
    Inline void operator()(index_t) const;

    /**
     * @brief Iterative geodesic pusher substep for momentum only.
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param vp_upd updated particle velocity [return].
     */
    Inline void ForceFreePush( const coord_t<D>& xp,
                                                const real_t&     vp,
                                                        real_t&     vp_upd) const;

    /**
     * @brief Iterative geodesic pusher substep for coordinate only.
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param xp_upd updated particle coordinate [return].
     */
    Inline void CoordinatePush(const coord_t<D>&  xp,
                                                  const real_t&      vp,
                                                  coord_t<D>&  xp_upd) const;


    /**
     * @brief EM pusher substep.
     * @param xp coordinate of the particle.
     * @param vp covariant velocity of the particle.
     * @param e0 electric field at the particle position.
     * @param v_upd updated covarient velocity of the particle [return].
     */
    Inline void EfieldHalfPush(const coord_t<D>& xp,
                                                  const real_t&     vp,
                                                  const real_t&     e0,
                                                  real_t&       vp_upd) const;
    // Helper functions

     /**
     * @brief First order Yee mesh field interpolation to particle position.
     * @param p index of the particle.
     * @param e interpolated e-field 
     */
    Inline void interpolateFields(index_t& p,
                                  real_t&  e) const {
      const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
      const auto dx1_ { static_cast<real_t>(dx1(p)) };
      real_t c1  = HALF * (Dfield(i, em::dx1) + Dfield(i - 1, em::dx1));
      real_t c2  = HALF * (Dfield(i, em::dx1) + Dfield(i + 1, em::dx1));
      e = c1 * (ONE - dx1_) + c2 * dx1_;
    }
    
    /**
     * @brief Compute controvariant component u^0 for massive particles.
     */

    Inline auto compute_u0_v(const real_t& v, 
                              const coord_t<D>& xi) const {
      return ONE / math::sqrt(SQR(metric.alpha(xi)) - 
                 metric.f2(xi) * SQR(v) - TWO * metric.f1(xi) * v - metric.f0(xi));
    }

    Inline auto compute_u0_u(const coord_t<Dim::_3D>& u_cov, 
                           const coord_t<Dim::_3D>& u_ccov,
                           const coord_t<D>& xi) const {
      return math::sqrt((u_cov[0] * u_ccov[0] + u_cov[1] * u_ccov[1] + u_cov[2] * u_ccov[2]) / 
                        (SQR(metric.alpha(xi)) + SQR(metric.beta(xi))));
    }

    // Extra
    Inline void boundaryConditions(index_t& p) const{
      if (i1(p) < i1_absorb && is_absorb_i1min) {
        tag(p) = ParticleTag::dead;
      } else if (i1(p) >= ni1 && is_absorb_i1max) {
        tag(p) = ParticleTag::dead;
      }

#if defined(MPI_ENABLED)
      tag(p) = mpi::SendTag(tag(p), i1(p) < 0, i1(p) >= ni1);
#endif
    }
    
  };



  /* -------------------------------------------------------------------------- */
  /*                                 1D Pushers                                 */
  /* -------------------------------------------------------------------------- */

  /**
  * massive particle electric field pusher
  */
  template <class M>
  Inline void Pusher_kernel<M>::EfieldHalfPush(const coord_t<D>& xp,
                                                  const real_t&     vp,
                                                  const real_t&     e0,
                                                  real_t&       vp_upd) const {
    real_t pp { ZERO };
    real_t pp_upd { ZERO };
    
    real_t COEFF { HALF * coeff * metric.alpha(xp) };

    //calculate canonical momentum
    real_t u0 { compute_u0_v(vp, xp) };
    pp = u0 * (metric.f2(xp) * vp + metric.f1(xp));

    //calculate updated canonical momentum
    pp_upd = pp + COEFF * e0;

    //calculate updated velocity
    vp_upd = (pp_upd / u0 - metric.f1(xp)) / metric.f2(xp);
  } 
  /**
  * massive particle momentum pusher under force-free constraint
  */
  template <class M>
  Inline void Pusher_kernel<M>::ForceFreePush( const coord_t<D>& xp,
                                               const real_t&     vp,
                                                     real_t&     vp_upd) const {
    real_t vp_mid { ZERO };
    //canonical momentum of particles
    real_t pp { ZERO };
    real_t pp_upd { ZERO };

    vp_upd = vp;

    real_t u0 { compute_u0_v(vp, xp) };
    //calculate canonical momentum
    pp = u0 * (metric.f2(xp) * vp + metric.f1(xp));
    pp_upd = pp ;

    for (auto i { 0 }; i < niter; ++i) {
    // find midpoint values
    vp_mid = HALF * (vp + vp_upd);
    
    u0 = compute_u0_v(vp_mid, xp);

    // find updated canonical momentum
    pp_upd = pp + 
            dt * u0 * 
            (- metric.alpha(xp) * DERIVATIVE(metric.alpha, xp[0]) +
                HALF * (DERIVATIVE(metric.f2, xp[0]) * SQR(vp_mid) + 
                        TWO * DERIVATIVE(metric.f1, xp[0]) * vp_mid +
                        DERIVATIVE(metric.f0, xp[0])));
    
    // find updated velocity
    vp_upd = (pp_upd / u0 - metric.f1(xp)) / metric.f2(xp);
    }
  }

  /**
  * coordinate pusher
  */
  template <class M>
  Inline void Pusher_kernel<M>::CoordinatePush(const coord_t<D>&  xp,
                                                  const real_t&      vp,
                                                  coord_t<D>&  xp_upd) const {
    xp_upd[0] = dt * vp + xp[0];
  }

  /* ------------------------- Massive particle pusher ------------------------ */
template <class M>
  Inline void Pusher_kernel<M>::operator()(index_t p) const {
    if (tag(p) != ParticleTag::alive) {
      if (tag(p) != ParticleTag::dead) {
        raise::KernelError(HERE, "Invalid particle tag in pusher");
      }
      return;
    }
    // record previous coordinate
    i1_prev(p)  = i1(p);
    dx1_prev(p) = dx1(p);

    coord_t<D> xp { ZERO };
    xp[0] = i_di_to_Xi(i1(p), dx1(p));

    real_t Dp_contrv { ZERO };
    interpolateFields(p, Dp_contrv);

    real_t ep_controv { metric.alpha(xp) * metric.template transform<1, Idx::U, Idx::D>(xp, Dp_contrv) };

    vec_t<Dim::_3D> up_cov { ZERO };
    vec_t<Dim::_3D> up_ccov { ZERO };

    up_cov[0] = ux1(p);
    up_cov[1] = ux2(p); 
    up_cov[2] = ux3(p);

    metric.template transform<Idx::D, Idx::U>(xp, up_cov, up_ccov);

    real_t vp { up_cov[0] / compute_u0_u(up_cov, up_ccov, xp) };
    

    /* -------------------------------- Leapfrog -------------------------------- */
    /* u_i(n - 1/2) -> u*_i(n) */
    real_t vp_upd { ZERO };
    EfieldHalfPush(xp, vp, ep_controv, vp_upd);

    /* u*_i(n) -> u**_i(n) */
    vp = vp_upd;
    ForceFreePush(xp, vp, vp_upd);
    /* u**_i(n) -> u_i(n + 1/2) */
    vp = vp_upd;
    EfieldHalfPush(xp, vp, ep_controv, vp_upd);

    /* x^i(n) -> x^i(n + 1) */
    coord_t<D> xp_upd { ZERO };
    CoordinatePush(xp, vp_upd, xp_upd);

    /* update coordinate */
    int      i1_;
    prtldx_t dx1_;
    from_Xi_to_i_di(xp_upd[0], i1_, dx1_);
    i1(p)  = i1_;
    dx1(p) = dx1_;

    /* update velocity */
    real_t u0 { compute_u0_v(vp_upd, xp) };
    ux1(p) = vp_upd * u0;
    ux3(p) = u0 * (metric.template h<3,3>(xp) * metric.OmegaF()
                   + metric.f1(xp) * vp_upd / (metric.OmegaF()+ metric.beta3(xp)));
 

    boundaryConditions(p);
    } 






} // namespace kernel::gr





#undef DERIVATIVE

#undef i_di_to_Xi
#undef from_Xi_to_i_di
#undef from_Xi_to_i

#endif // KERNELS_PARTICLE_PUSHER_GR_1D_HPP

