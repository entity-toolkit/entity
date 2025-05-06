#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#if defined(MPI_ENABLED)
#include <stdlib.h>
#endif //MPI_ENABLED

namespace user {
  using namespace ntt;

  // initializing guide field and curl(B) = J_ext at the initial time step
  template <Dimension D>
  struct InitFields {
    InitFields( real_t bbg ) : Bbg { bbg } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t { return Bbg; }
    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t { return ZERO; }
    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t { return ZERO; }

    private:
      const real_t Bbg;

  };

template <Dimension D>
struct ExternalCurrent {
  ExternalCurrent(real_t amplitude,
                  real_t num_waves_x,
                  real_t num_waves_y,
                  real_t num_waves_z,
                  real_t frequency,
                  real_t Lx,
                  real_t Ly,
                  real_t Lz)
    : A0 { amplitude }
    , frequency { frequency }
    , Lx { Lx }
    , Ly { Ly }
    , Lz { Lz }
    , time { "time", 1 } // size 1 by default, can be resized
    , time_index { 0 } {
    
    if constexpr (D == Dim::_2D) {
      kx = constant::TWO_PI * num_waves_x / Lx;
      ky = constant::TWO_PI * num_waves_y / Ly;
      kz = ZERO;
    }
    if constexpr (D == Dim::_3D) {
      kx = constant::TWO_PI * num_waves_x / Lx;
      ky = constant::TWO_PI * num_waves_y / Ly;
      kz = constant::TWO_PI * num_waves_z / Lz;
    }
  }

  Inline auto jx1(const coord_t<D>& x_Ph) const -> real_t {
    real_t phase = kx * x_Ph[0] + ky * x_Ph[1];
    if constexpr (D == Dim::_3D) {
      phase += kz * x_Ph[2];
    }
    return  math::cos(phase - frequency * time(time_index)) * A0;
  }

  Inline auto jx2(const coord_t<D>&) const -> real_t { return ZERO; }
  Inline auto jx3(const coord_t<D>&) const -> real_t { return ZERO; }


private:
  const real_t A0, frequency;
  const real_t Lx, Ly, Lz;
  real_t kx, ky, kz;
  int time_index;

public:
  Kokkos::View<real_t*> time;
};



  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions =
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t sx1, sx2, sx3;
    const real_t temp, amplitude;
    const real_t nwave_x, nwave_y, nwave_z, frequency;

    ExternalCurrent<D> ext_current;
    InitFields<D>      init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , sx1 { global_domain.mesh().extent(in::x1).second -
              global_domain.mesh().extent(in::x1).first }
      , sx2 { global_domain.mesh().extent(in::x2).second -
              global_domain.mesh().extent(in::x2).first }
      , sx3 { global_domain.mesh().extent(in::x3).second -
              global_domain.mesh().extent(in::x3).first }
      , temp { p.template get<real_t>("setup.temp") }
      , amplitude { p.template get<real_t>("setup.amplitude") }
      , nwave_x { p.template get<real_t>("setup.nwave_x") }
      , nwave_y { p.template get<real_t>("setup.nwave_y") }
      , nwave_z { p.template get<real_t>("setup.nwave_z") }
      , frequency { p.template get<real_t>("setup.frequency") }
      , init_flds { ONE }
      , ext_current { amplitude, nwave_x, nwave_y, nwave_z, frequency, sx1, sx2, sx3 }
      {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,local_domain.random_pool,temp);
      const auto injector = arch::UniformInjector<S, M, arch::Maxwellian>(energy_dist,{ 1, 2 });
      arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(params,local_domain,injector,HALF);
    }

    void CustomPostStep(timestep_t, simtime_t, Domain<S, M>& domain) {
        
        const auto dt = params.template get<real_t>("algorithms.timestep.dt");
        auto& ext_curr = ext_current;

        Kokkos::parallel_for(
          "Update time array",
          1,
          ClassLambda(index_t) {
            ext_curr.time(0) += dt;
          });
      
#if defined(MPI_ENABLED)
        int              rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

      auto EB          = domain.fields.em;
      auto pnt_quantity = ZERO;

      if constexpr (D == Dim::_2D) {

      #if defined(MPI_ENABLED)

        if(rank == MPI_ROOT_RANK) {

          pnt_quantity = EB(N_GHOSTS + 1, N_GHOSTS + 1, em::ex1);

        }

      #else

      pnt_quantity = EB(N_GHOSTS + 1, N_GHOSTS + 1, em::ex1);

      #endif

       }

      std::ofstream myfile1;

      #if defined(MPI_ENABLED)

        if(rank == MPI_ROOT_RANK) {

          if (time == 0) {
            myfile1.open("HF_out.txt");
          } else {
            myfile1.open("HF_out.txt", std::ios_base::app);
          }
          myfile1 << pnt_quantity << std::endl;

        }

      #else

          if (time == 0) {
            myfile1.open("HF_out.txt");
          } else {
            myfile1.open("HF_out.txt", std::ios_base::app);
          }
          myfile1 << pnt_quantity << std::endl;

      #endif

    }

  };


} // namespace user

#endif
