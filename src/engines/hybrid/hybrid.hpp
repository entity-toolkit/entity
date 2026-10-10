#ifndef ENGINES_HYBRID_HYBRID_HPP
#define ENGINES_HYBRID_HYBRID_HPP

#include "enums.h"

#include "traits/metric.h"
#include "utils/timer.h"

#include "engines/hybrid/fields_bcs.h"
#include "engines/hybrid/fieldsolvers.h"
#include "engines/hybrid/moments_filter.h"
#include "engines/hybrid/particle_pusher.h"

#include "engines/engine.hpp"

namespace ntt {

  template <CartesianMetricClass M>
  class HYBRIDEngine : public Engine<SimEngine::HYBRID, M> {
    using base_t   = Engine<SimEngine::HYBRID, M>;
    using pgen_t   = user::PGen<SimEngine::HYBRID, M>;
    using domain_t = Domain<SimEngine::HYBRID, M>;
    // contents
    using base_t::m_metadomain;
    using base_t::m_params;
    using base_t::m_pgen;
    // methods
    using base_t::init;
    // variables
    using base_t::dt;
    using base_t::max_steps;
    using base_t::runtime;
    using base_t::start_step;
    using base_t::step;
    using base_t::time;

    // time-centered moments for the corrector field advance (subcycle_centered)
    ndfield_t<M::Dim, 6> m_moments_half;

  public:
    static constexpr auto S { SimEngine::HYBRID };

    HYBRIDEngine(const SimulationParams& params) : base_t { params } {}

    ~HYBRIDEngine() override = default;

    void step_forward(timer::Timers& timers, domain_t& dom) override {
      /**
       * Initially: em::012    --
       *            em::345    Bf^(n)
       *
       *            em0::012   --
       *            em0::345   --
       *
       *            aux::012   V^(n) (except at the first step of a run)
       *            aux::3     N^(n) (except at the first step of a run)
       *
       *            bckp::012  --
       *            bckp::345  --
       *
       *            cur::012   --
       *
       *            x_prtl   at n
       *            u_prtl   at n
       */
      // Pegasus-style sub-cycled field advance (fieldsolvers.h::SubcycledFaraday):
      // the whistler-stiff Ohm's-law/Faraday loop is integrated with adaptive
      // SSP-RK3 sub-steps at its own CFL instead of single full-dt Euler pushes.
      // false -> the legacy 3-push scheme.
      const bool subcycle = m_params.template get<bool>("hybrid.subcycle");
      // corrector field advance driven by the time-centered moments
      // (M^(n) + M') / 2 instead of the predicted M' (subcycled mode only)
      const bool centered = subcycle and
                            m_params.template get<bool>("hybrid.subcycle_centered");

      // first step of this run (step 0, or the first step after resuming from a
      // checkpoint, which stores neither the moments nor the field ghosts)
      if (step == start_step) {
        // fill Bf^(n) ghosts (periodic / MPI) so the field-solver stencils are valid
        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, ::Comm::EM_345);
        timers.stop("Communications");
        // non-periodic (reflecting-wall) boundaries on the initial Bf, if any
        timers.start("FieldBoundaries");
        hybrid::FieldBoundaries<M::Dim>(dom, m_metadomain.mesh(), BC::B);
        timers.stop("FieldBoundaries");

        // compute N^(0) and V^(0) -> aux::012, aux::3 (deposit-only, no push)
        hybrid::DepositMoments(dom, m_params);

        timers.start("Communications");
        m_metadomain.SynchronizeFields(dom, ::Comm::AUX); // additive remap of deposit tails
        timers.stop("Communications");
        // reflecting wall: fold the wall-ghost deposit tails back (image plasma)
        // before the ghost fill, so the halo copies carry the folded values
        timers.start("FieldBoundaries");
        hybrid::MomentsWallBC(dom, m_metadomain.mesh(), /* fold */ true);
        timers.stop("FieldBoundaries");
        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, ::Comm::AUX); // fill ghosts for EMF reads
        timers.stop("Communications");
        // mirror-fill the wall ghosts again, now that the transverse halo is filled
        timers.start("FieldBoundaries");
        hybrid::MomentsWallBC(dom, m_metadomain.mesh(), /* fold */ false);
        timers.stop("FieldBoundaries");
        // smooth N^(0),V^(0) (kills grid-scale shot noise before the field solve)
        timers.start("MomentFiltering");
        hybrid::MomentsFilter(m_metadomain, dom, m_params);
        timers.stop("MomentFiltering");
      }

      // EMF calculation #0 — the un-averaged field E^(n).
      // Using: aux::012 [V^(n)], aux::3 [N^(n)], em::345 [Bf^(n)]
      //
      // Ee^(n) = EMF(N^(n), V^(n), Bf^(n))           -> em::012
      // Ec^(n) = EMF(N^(n), V^(n), Bc^(n)=interp Bf) -> em0::012
      // (these seed the trapezoidal average E' = 1/2(E* + E^(n)) in EMF #1/#2)
      timers.start("FieldSolver");
      hybrid::EMF(dom, this->engineParams(), m_params, hybrid::emf::push0);
      timers.stop("FieldSolver");
      timers.start("Communications");
      m_metadomain.CommunicateFields(dom, ::Comm::EM_012 | ::Comm::EM0_012);
      timers.stop("Communications");
      // reflecting wall: E_tan = 0 on the wall plane of Ee^(n) before Faraday #1
      // (this is what freezes the wall-plane B_n for an oblique background field)
      timers.start("FieldBoundaries");
      hybrid::FieldBoundaries<M::Dim>(dom, m_metadomain.mesh(), BC::E);
      timers.stop("FieldBoundaries");

      // Faraday push #1 (predictor field advance)
      // Using: em::012 [Ee^(n)], em::345 [Bf^(n)]
      //
      // legacy:    Bf* = Bf^(n) + dt * curl Ee^(n)
      // subcycled: Bf* = integrate dB/dt = -curl EMF(N^(n), V^(n), B) over dt
      //            (E refreshed every sub-step; comms + wall conditions inside,
      //            cur comes back with valid ghosts)
      //
      // Now: cur::012 <-- Bf*
      timers.start("FieldSolver");
      if (subcycle) {
        hybrid::SubcycledFaraday(m_metadomain,
                                 dom,
                                 this->engineParams(),
                                 m_params,
                                 dom.fields.aux);
      } else {
        hybrid::Faraday(dom, this->engineParams(), hybrid::faraday::push1);
      }
      timers.stop("FieldSolver");
      if (not subcycle) {
        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, ::Comm::CUR); // fill Bf* ghosts for EMF #1
        timers.stop("Communications");
        // reflecting wall: even-mirror the Bf* scratch ghosts (a physical wall
        // has no halo exchange to fill them) so the Hall stencil sees no fake
        // wall current
        timers.start("FieldBoundaries");
        hybrid::WallScratchB(dom, m_metadomain.mesh());
        timers.stop("FieldBoundaries");
      }

      // EMF calculation #1
      // Using:
      //   aux::012 [P^(n)]
      //   aux::3 [N^(n)]
      //   em::012 [Ee^(n)]
      //   em::345 [Bf^(n)]
      //   em0::012 [Ec^(n)]
      //   cur::012 [Bf*]
      //
      // Bc* = interpolate Bf*
      // Bc^(n) = interpolate Bf^(n)
      // Ee* = EMF(N^(n), P^(n), Bf*)
      // Ec* = EMF(N^(n), P^(n), Bc*)
      //
      // Ee' = 0.5 * (Ee* + Ee^(n))
      // Ec' = 0.5 * (Ec* + Ec^(n))
      // Bc' = 0.5 * (Bc* + Bc^(n))
      //
      // Now:
      //   em0::345 <-- Ee'
      //   bckp::012 <-- Ec'
      //   bckp::345 <-- Bc'
      timers.start("FieldSolver");
      hybrid::EMF(dom, this->engineParams(), m_params, hybrid::emf::push12);
      timers.stop("FieldSolver");

      // Particle push #1 (predictor) — Pegasus Fig. 2 steps 7+8.
      // Using: bckp::012 [Ec'], bckp::345 [Bc']; transient (no store, no particle BCs).
      // Pushes in registers and deposits predicted moments: aux::012 <-- V', aux::3 <-- N'
      timers.start("Communications");
      // Ee' (em0::345) ghosts for Faraday push #2; Ec'/Bc' (bckp) ghosts for the gather
      m_metadomain.CommunicateFields(dom, ::Comm::EM0_345 | ::Comm::Bckp);
      timers.stop("Communications");
      // reflecting wall: E_tan = 0 on the wall plane of Ee' (consumed by
      // Faraday #2) + conductor-mirror the bckp wall ghosts for the gather
      timers.start("FieldBoundaries");
      hybrid::WallEPrime(dom, m_metadomain.mesh());
      hybrid::WallBckpFill(dom, m_metadomain.mesh());
      timers.stop("FieldBoundaries");
      if (centered) {
        // keep M^(n) (aux, valid ghosts) before the predictor overwrites it
        timers.start("FieldSolver");
        if (m_moments_half.span() != dom.fields.aux.span()) {
          m_moments_half = ndfield_t<M::Dim, 6> {
            Kokkos::view_alloc("moments_half", Kokkos::WithoutInitializing),
            dom.fields.aux.layout()
          };
        }
        Kokkos::deep_copy(m_moments_half, dom.fields.aux);
        timers.stop("FieldSolver");
      }
      timers.start("ParticlePusher");
      hybrid::ParticlePush(dom, this->engineParams(), m_params, /* corrector */ false);
      timers.stop("ParticlePusher");
      timers.start("Communications");
      m_metadomain.SynchronizeFields(dom,
                                     ::Comm::AUX); // additive remap of deposit tails
      timers.stop("Communications");
      // reflecting wall: image-plasma fold of the predicted moments, before the
      // ghost fill so the halo copies carry the folded values
      timers.start("FieldBoundaries");
      hybrid::MomentsWallBC(dom, m_metadomain.mesh(), /* fold */ true);
      timers.stop("FieldBoundaries");
      timers.start("Communications");
      m_metadomain.CommunicateFields(dom, ::Comm::AUX); // fill ghosts for EMF #2
      timers.stop("Communications");
      // mirror-fill the wall ghosts again, now that the transverse halo is filled
      timers.start("FieldBoundaries");
      hybrid::MomentsWallBC(dom, m_metadomain.mesh(), /* fold */ false);
      timers.stop("FieldBoundaries");
      // smooth predicted N',V' before EMF #2
      timers.start("MomentFiltering");
      hybrid::MomentsFilter(m_metadomain, dom, m_params);
      timers.stop("MomentFiltering");
      if (centered) {
        // M^(n+1/2) = (M^(n) + M') / 2 over the full extent (ghosts of both valid)
        timers.start("FieldSolver");
        for (const uint8_t c : { 0u, 3u }) {
          Kokkos::parallel_for(
            "FieldCombine",
            dom.mesh.rangeAllCells(),
            kernel::hybrid::FieldCombine_kernel<M::Dim, 6, 6>(m_moments_half,
                                                              dom.fields.aux,
                                                              HALF,
                                                              HALF,
                                                              c,
                                                              c));
        }
        timers.stop("FieldSolver");
      }

      // Faraday push #2 (corrector field advance)
      // Using: em::345 [Bf^(n)] + em0::345 [Ee'] (legacy) / aux [N', V'] (subcycled)
      //
      // legacy:    Bf** = Bf^(n) + dt * curl Ee'
      // subcycled: Bf** = integrate dB/dt = -curl EMF(N', V', B) over dt
      //            (E refreshed from the PREDICTED moments every sub-step, or
      //            from (M^(n) + M') / 2 with hybrid.subcycle_centered;
      //            this advance IS the full-interval B^(n) -> B^(n+1)
      //            integration, accepted as Bf^(n+1) below)
      //
      // Now: cur::012 <-- Bf**
      timers.start("FieldSolver");
      if (subcycle) {
        hybrid::SubcycledFaraday(m_metadomain,
                                 dom,
                                 this->engineParams(),
                                 m_params,
                                 centered ? m_moments_half : dom.fields.aux);
      } else {
        hybrid::Faraday(dom, this->engineParams(), hybrid::faraday::push2);
      }
      timers.stop("FieldSolver");
      if (not subcycle) {
        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, ::Comm::CUR); // fill Bf** ghosts for EMF #2
        timers.stop("Communications");
        // reflecting wall: even-mirror the Bf** scratch ghosts (see Faraday #1)
        timers.start("FieldBoundaries");
        hybrid::WallScratchB(dom, m_metadomain.mesh());
        timers.stop("FieldBoundaries");
      }

      // EMF calculation #2
      // Using:
      //   aux::012 [P']
      //   aux::3 [N']
      //   em::012 [Ee^(n)]
      //   em::345 [Bf^(n)]
      //   em0::012 [Ec^(n)]
      //   cur::012 [Bf**]
      //
      // Bc** = interpolate Bf**
      // Bc^(n) = interpolate Bf^(n)
      // Ee** = EMF(N', P', Bf**)
      // Ec** = EMF(N', P', Bc**)
      //
      // Ee'' = 0.5 * (Ee** + Ee^(n))
      // Ec'' = 0.5 * (Ec** + Ec^(n))
      // Bc'' = 0.5 * (Bc** + Bc^(n))
      //
      // Now:
      //   em0::345 <-- Ee''
      //   bckp:012 <-- Ec''
      //   bckp:345 <-- Bc''
      timers.start("FieldSolver");
      hybrid::EMF(dom, this->engineParams(), m_params, hybrid::emf::push12);
      timers.stop("FieldSolver");
      timers.start("Communications");
      // Ee'' (em0::345) ghosts for Faraday push #3; Ec''/Bc'' (bckp) ghosts for the gather
      m_metadomain.CommunicateFields(dom, ::Comm::EM0_345 | ::Comm::Bckp);
      timers.stop("Communications");
      // reflecting wall: E_tan = 0 on the wall plane of Ee'' (consumed by
      // Faraday #3) + conductor-mirror the bckp wall ghosts for the corrector
      timers.start("FieldBoundaries");
      hybrid::WallEPrime(dom, m_metadomain.mesh());
      hybrid::WallBckpFill(dom, m_metadomain.mesh());
      timers.stop("FieldBoundaries");

      // Faraday push #3 (accept B^(n+1))
      // Using: em0::345 [Ee''], em::345 [Bf^(n)] (legacy) / cur [Bf**] (subcycled)
      //
      // legacy:    Bf^(n+1) = Bf^(n) + dt * curl Ee''
      // subcycled: Bf^(n+1) = Bf** -- the corrector advance above already
      //            integrated the full interval; a third full-dt Euler push
      //            would redundantly re-integrate it (stiff-unstable)
      //
      // Now:
      //   em::345 <-- Bf^(n+1)
      timers.start("FieldSolver");
      if (subcycle) {
        hybrid::AcceptSubcycledB(dom);
      } else {
        hybrid::Faraday(dom, this->engineParams(), hybrid::faraday::push3);
      }
      timers.stop("FieldSolver");
      timers.start("Communications");
      m_metadomain.CommunicateFields(dom, ::Comm::EM_345); // Bf^(n+1) ghosts for next step
      timers.stop("Communications");
      // non-periodic (reflecting-wall) boundaries on Bf^(n+1), if any
      timers.start("FieldBoundaries");
      hybrid::FieldBoundaries<M::Dim>(dom, m_metadomain.mesh(), BC::B);
      timers.stop("FieldBoundaries");

      // Particle push #2 (corrector) — Pegasus Fig. 2 step 12.
      // Using: bckp::012 [Ec''], bckp::345 [Bc'']; accepted (store-back +
      // particle BCs). Pushes and deposits final moments: x_prtl at n+1,
      // aux::012 <-- V^(n+1), aux::3 <-- N^(n+1) (bckp ghosts already filled
      // after EMF #2; Faraday push #3 does not touch bckp)
      timers.start("ParticlePusher");
      hybrid::ParticlePush(dom, this->engineParams(), m_params, /* corrector */ true);
      timers.stop("ParticlePusher");

      timers.start("Communications");
      m_metadomain.SynchronizeFields(dom, ::Comm::AUX);
      timers.stop("Communications");
      // reflecting wall: image-plasma fold of the final moments, before the
      // ghost fill so the halo copies carry the folded values
      timers.start("FieldBoundaries");
      hybrid::MomentsWallBC(dom, m_metadomain.mesh(), /* fold */ true);
      timers.stop("FieldBoundaries");
      timers.start("Communications");
      m_metadomain.CommunicateFields(dom, ::Comm::AUX); // fill ghosts for next step's EMF
      timers.stop("Communications");
      // mirror-fill the wall ghosts again, now that the transverse halo is filled
      timers.start("FieldBoundaries");
      hybrid::MomentsWallBC(dom, m_metadomain.mesh(), /* fold */ false);
      timers.stop("FieldBoundaries");
      // smooth final N^(n+1),V^(n+1) for next step's EMF #0/#1
      timers.start("MomentFiltering");
      hybrid::MomentsFilter(m_metadomain, dom, m_params);
      timers.stop("MomentFiltering");

      timers.start("Communications");
      m_metadomain.CommunicateParticles(dom);
      timers.stop("Communications");

      /**
       * Finally:   em::012    --
       *            em::345    Bf^(n+1)
       *
       *            em0::012   --
       *            em0::345   --
       *
       *            aux::012   V^(n+1)
       *            aux::3     N^(n+1)
       *
       *            bckp::012  --
       *            bckp::345  --
       *
       *            cur::012   --
       *
       *            x_prtl   at n+1
       *            u_prtl   at n+1
       */
    }
  };
} // namespace ntt

#endif // ENGINES_HYBRID_HYBRID_HPP
