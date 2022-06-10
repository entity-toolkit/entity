#include "global.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include "problem_generator.hpp"

#include <cmath>

namespace ntt {

  using ProblemGenerator1D = ProblemGenerator<Dimension::ONE_D, SimulationType::PIC>;
  using Meshblock1D        = Meshblock<Dimension::ONE_D, SimulationType::PIC>;

  using ProblemGenerator2D = ProblemGenerator<Dimension::TWO_D, SimulationType::PIC>;
  using Meshblock2D        = Meshblock<Dimension::TWO_D, SimulationType::PIC>;

  using ProblemGenerator3D = ProblemGenerator<Dimension::THREE_D, SimulationType::PIC>;
  using Meshblock3D        = Meshblock<Dimension::THREE_D, SimulationType::PIC>;

  template <>
  void ProblemGenerator2D::userInitFields(const SimulationParams&, Meshblock2D& mblock) {
    using index_t = std::size_t;
    Kokkos::parallel_for(
      "userInitFlds", mblock.loopActiveCells(), Lambda(index_t i, index_t j) {
        auto i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
        auto j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};
        mblock.em(i, j, em::ex1) = 1.0 * i_;
        mblock.em(i, j, em::ex2) = -2.0 * j_ - 1.5 * i_;
        mblock.em(i, j, em::ex3) = 3.0 * j_ + 5.0 * i_;
        mblock.em(i, j, em::bx1) = -4.0 * j_;
        mblock.em(i, j, em::bx2) = 5.0 * i_ + 2.12 * j_;
        mblock.em(i, j, em::bx3) = -6.0 * j_ - 3.0 * i_;
      });
  }

  template <>
  void ProblemGenerator2D::userInitParticles(const SimulationParams&, Meshblock2D&) {
    // using index_t = const std::size_t;
    // Kokkos::parallel_for(
    //   "userInitPrtls", NTTRange<Dimension::ONE_D>({0}, {1}), Lambda(index_t p) {
    //     coord_t<Dimension::TWO_D> x {0.1, 0.12}, x_CU;
    //     mblock.metric.x_Cart2Code(x, x_CU);
    //     auto [i1, dx1] = mblock.metric.CU_to_Idi(x_CU[0]);
    //     auto [i2, dx2] = mblock.metric.CU_to_Idi(x_CU[1]);
    //     // electron
    //     mblock.particles[0].i1(p)  = i1;
    //     mblock.particles[0].i2(p)  = i2;
    //     mblock.particles[0].dx1(p) = dx1;
    //     mblock.particles[0].dx2(p) = dx2;
    //     mblock.particles[0].ux1(p) = 1.0;
    //     // positron
    //     mblock.particles[1].i1(p)  = i1;
    //     mblock.particles[1].i2(p)  = i2;
    //     mblock.particles[1].dx1(p) = dx1;
    //     mblock.particles[1].dx2(p) = dx2;
    //     mblock.particles[1].ux1(p) = 1.0;
    //     // ion
    //     mblock.particles[2].i1(p)  = i1;
    //     mblock.particles[2].i2(p)  = i2;
    //     mblock.particles[2].dx1(p) = dx1;
    //     mblock.particles[2].dx2(p) = dx2;
    //     mblock.particles[2].ux1(p) = 1.0;
    //     // photon
    //     mblock.particles[3].i1(p)  = i1;
    //     mblock.particles[3].i2(p)  = i2;
    //     mblock.particles[3].dx1(p) = dx1;
    //     mblock.particles[3].dx2(p) = dx2;
    //     mblock.particles[3].ux1(p) = 1.0;
    //   });
    // mblock.particles[0].set_npart(1);
    // mblock.particles[1].set_npart(1);
    // mblock.particles[2].set_npart(1);
    // mblock.particles[3].set_npart(1);
  }

  // 1D
  template <>
  void ProblemGenerator1D::userInitFields(const SimulationParams&, Meshblock1D&) {}
  template <>
  void ProblemGenerator1D::userInitParticles(const SimulationParams&, Meshblock1D&) {}

  // 3D
  template <>
  void ProblemGenerator3D::userInitFields(const SimulationParams&, Meshblock3D&) {}
  template <>
  void ProblemGenerator3D::userInitParticles(const SimulationParams&, Meshblock3D&) {}

} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;
