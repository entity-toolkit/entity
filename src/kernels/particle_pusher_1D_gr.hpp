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
    array_t<real_t*>      px1;
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
                  const array_t<real_t*>&           px1,
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
      , px1 { px1 }
      , tag { tag }
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
                               const real_t&     pp,
			       const real_t&     ex,
                                                        real_t&     pp_upd) const;

    /**
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param xp_upd updated particle coordinate [return].
     */
    Inline void CoordinatePush(const coord_t<D>&  xp,
                                                  const real_t&      vp,
                                                  coord_t<D>&  xp_upd) const;

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

    Inline auto compute_u0(const real_t& pxi, 
                              const coord_t<D>& xi) const -> real_t{
      return math::sqrt((SQR(pxi) + metric.f2(xi)) / 
                         (metric.f2(xi) * (SQR(metric.alpha(xi)) - metric.f0(xi)) + SQR(metric.f1(xi))));
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
  * massive particle momentum pusher under force-free constraint
  */
  template <class M>
  Inline void Pusher_kernel<M>::ForceFreePush( const coord_t<D>& xp,
                                               const real_t&     pp,
					                                     const real_t&     ex,
                                                     real_t&     pp_upd) const {
    

    real_t pp_mid { pp };
    pp_upd = pp;


    //printf("Iteration starts.\n");
    for (auto i { 0 }; i < niter; ++i) {
      // find midpoint values
      pp_mid = 0.5 * (pp + pp_upd);
       
      // find updated momentum
      pp_upd = pp + 
               dt * (coeff * ex +
                     HALF * compute_u0(pp_mid, xp) * 
                          (-TWO * metric.alpha(xp) * DERIVATIVE(metric.alpha, xp[0]) +
                           DERIVATIVE(metric.f2, xp[0]) * SQR((pp_mid / compute_u0(pp_mid, xp) - metric.f1(xp)) / metric.f2(xp)) +
                           TWO * DERIVATIVE(metric.f1, xp[0]) * (pp_mid / compute_u0(pp_mid, xp) - metric.f1(xp)) / metric.f2(xp) + 
                           DERIVATIVE(metric.f0, xp[0])
                          )
                    );
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

    real_t Dp_cntrv { ZERO };
    interpolateFields(p, Dp_cntrv);

    real_t ep_cntrv { metric.alpha(xp) * metric.template transform<1, Idx::U, Idx::D>(xp, Dp_cntrv) };

    real_t pp { px1(p) };
    

    /* -------------------------------- Leapfrog -------------------------------- */
    /* u_i(n - 1/2) ->  u_i(n + 1/2) */
    real_t pp_upd { ZERO };

    ForceFreePush(xp, pp, ep_cntrv, pp_upd);


    /* x^i(n) -> x^i(n + 1) */
    coord_t<D> xp_upd { ZERO };
    CoordinatePush(xp, (pp_upd / compute_u0(pp_upd, xp) - metric.f1(xp)) / metric.f2(xp), xp_upd);

    /* update coordinate */
    int      i1_;
    prtldx_t dx1_;
    from_Xi_to_i_di(xp_upd[0], i1_, dx1_);
    i1(p)  = i1_;
    dx1(p) = dx1_;

    /* update velocity */
    px1(p) = pp_upd;
 

    boundaryConditions(p);
    } 






} // namespace kernel::gr





#undef DERIVATIVE

#undef i_di_to_Xi
#undef from_Xi_to_i_di
#undef from_Xi_to_i

#endif // KERNELS_PARTICLE_PUSHER_GR_1D_HPP

