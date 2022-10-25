#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"
#include "pgen.h"
#include "sim_params.h"
#include "meshblock.h"
#include "particle_macros.h"

#ifdef NTTINY_ENABLED
#  include "nttiny/api.h"
#endif

#include <map>

namespace ntt {
  // template <Dimension D, SimulationType S>
  // class Injector {
  //   for
  // };

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}
    inline void UserInitParticles(const SimulationParams& params, Meshblock<D, S>& mblock) {
      if constexpr (D == Dim2) {
        auto&                electrons   = mblock.particles[0];
        auto                 random_pool = *(mblock.random_pool_ptr);
        array_t<std::size_t> nprtl("Nprtl");
        Kokkos::parallel_for(
          "test_prtl_inject", mblock.rangeActiveCells(), Lambda(index_t i1, index_t i2) {
            const real_t i1_ {static_cast<real_t>(static_cast<int>(i1) - N_GHOSTS)};
            const real_t i2_ {static_cast<real_t>(static_cast<int>(i2) - N_GHOSTS)};

            // if (i1 % 2 == 0) {
            typename RandomNumberPool_t::generator_type rand_gen = random_pool.get_state();
            std::size_t p = Kokkos::atomic_fetch_add(&nprtl(), 1);

            vec_t<Dim2> xmin_Cart {ZERO}, xmax_Cart {ZERO};
            mblock.metric.x_Code2Cart({i1_, i2_}, xmin_Cart);
            mblock.metric.x_Code2Cart({i1_ + ONE, i2_ + ONE}, xmax_Cart);
            real_t rx = rand_gen.frand(xmin_Cart[0], xmax_Cart[0]);
            real_t ry = rand_gen.frand(xmin_Cart[1], xmax_Cart[1]);
            init_prtl_2d(mblock, electrons, p, rx, ry, 0.0, 0.0, 0.0);
            random_pool.free_state(rand_gen);
            // }
          });
        auto nprtl_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nprtl);
        electrons.setNpart(nprtl_h());
        // auto        ncells      = mblock.Ni1() * mblock.Ni2() * mblock.Ni3();
        // std::size_t npart       = (std::size_t)((double)(ncells * params.ppc0() * 0.5));
        // auto&       electrons   = mblock.particles[0];
        // auto        random_pool = *(mblock.random_pool_ptr);
        // real_t      Xmin        = mblock.metric.x1_min;
        // real_t      Xmax        = mblock.metric.x1_max;
        // real_t      Ymin        = mblock.metric.x2_min;
        // real_t      Ymax        = mblock.metric.x2_max;
        // electrons.setNpart(npart);
        // Kokkos::parallel_for(
        //   "userInitPrtls", CreateRangePolicy<Dim1>({0}, {(int)npart}), Lambda(index_t p) {
        //     typename RandomNumberPool_t::generator_type rand_gen = random_pool.get_state();

        //     real_t rx {rand_gen.frand(Xmin, Xmax)};
        //     real_t ry {rand_gen.frand(Ymin, Ymax)};
        //     real_t u1 {0.01 * math::sin(2.0 * constant::TWO_PI * rx / (Xmax - Xmin))};

        //     init_prtl_2d(mblock, electrons, p, rx, ry, u1, 0.0, 0.0);
        //     random_pool.free_state(rand_gen);
        //   });
      }
    }

#ifdef NTTINY_ENABLED
    inline void
    UserInitBuffers_nttiny(const SimulationParams&,
                           const Meshblock<D, S>&,
                           std::map<std::string, nttiny::ScrollingBuffer>& buffers) {
      nttiny::ScrollingBuffer ex;
      buffers.insert({"Ex", std::move(ex)});
    }

    inline void
    UserSetBuffers_nttiny(const real_t& time,
                          const SimulationParams&,
                          const Meshblock<D, S>&                          mblock,
                          std::map<std::string, nttiny::ScrollingBuffer>& buffers) {
      if constexpr (D == Dim2) {
        buffers["Ex"].AddPoint(
          time, mblock.em_h((int)(mblock.Ni1() / 8.0), (int)(mblock.Ni2() / 2.0), em::ex1));
      }
    }

#endif
  }; // struct ProblemGenerator

} // namespace ntt

#endif
