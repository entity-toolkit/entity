#include "global.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include "problem_generator.hpp"

#include <cmath>

namespace ntt {

#if (METRIC == MINKOWSKI_METRIC)
  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::PIC>& mblock) {
    using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;
    Kokkos::parallel_for(
      "userInitFlds", mblock.loopActiveCells(), Lambda(index_t i, index_t j) {
        real_t i_ {(real_t)(static_cast<int>(i) - N_GHOSTS)},
          j_ {(real_t)(static_cast<int>(j) - N_GHOSTS)};
        real_t                    ex2_hat {0.1}, bx3_hat {1.0};
        vec_t<Dimension::THREE_D> e_cntrv, b_cntrv;
        mblock.metric.v_Hat2Cntrv({i_ + HALF, j_}, {ZERO, ex2_hat, ZERO}, e_cntrv);
        mblock.metric.v_Hat2Cntrv({i_ + HALF, j_ + HALF}, {ZERO, ZERO, bx3_hat}, b_cntrv);
        mblock.em(i, j, em::ex1) = ZERO;
        mblock.em(i, j, em::ex2) = e_cntrv[1];
        mblock.em(i, j, em::ex3) = ZERO;
        mblock.em(i, j, em::bx1) = ZERO;
        mblock.em(i, j, em::bx2) = ZERO;
        mblock.em(i, j, em::bx3) = b_cntrv[2];
      });
  }

  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::PIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::PIC>& mblock) {
    using index_t = const std::size_t;
    Kokkos::parallel_for(
      "userInitPrtls", NTTRange<Dimension::ONE_D>({0}, {1}), Lambda(index_t p) {
        coord_t<Dimension::TWO_D> x {0.1, 0.12}, x_CU;
        mblock.metric.x_Cart2Code(x, x_CU);
        auto [i1, dx1] = mblock.metric.CU_to_Idi(x_CU[0]);
        auto [i2, dx2] = mblock.metric.CU_to_Idi(x_CU[1]);
        // electron
        mblock.particles[0].i1(p)  = i1;
        mblock.particles[0].i2(p)  = i2;
        mblock.particles[0].dx1(p) = dx1;
        mblock.particles[0].dx2(p) = dx2;
        mblock.particles[0].ux1(p) = 1.0;
        // positron
        mblock.particles[1].i1(p)  = i1;
        mblock.particles[1].i2(p)  = i2;
        mblock.particles[1].dx1(p) = dx1;
        mblock.particles[1].dx2(p) = dx2;
        mblock.particles[1].ux1(p) = 1.0;
        // ion
        mblock.particles[2].i1(p)  = i1;
        mblock.particles[2].i2(p)  = i2;
        mblock.particles[2].dx1(p) = dx1;
        mblock.particles[2].dx2(p) = dx2;
        mblock.particles[2].ux1(p) = 1.0;
        // photon
        mblock.particles[3].i1(p)  = i1;
        mblock.particles[3].i2(p)  = i2;
        mblock.particles[3].dx1(p) = dx1;
        mblock.particles[3].dx2(p) = dx2;
        mblock.particles[3].ux1(p) = 1.0;
      });
    mblock.particles[0].set_npart(1);
    mblock.particles[1].set_npart(1);
    mblock.particles[2].set_npart(1);
    mblock.particles[3].set_npart(1);
  }
#elif (METRIC == SPHERICAL_METRIC) || (METRIC == QSPHERICAL_METRIC)
  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::PIC>& mblock) {
    using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;
    Kokkos::parallel_for(
      "userInitFlds", mblock.loopActiveCells(), Lambda(index_t i, index_t j) {
        // real_t                    i_ {(real_t)(static_cast<int>(i) - N_GHOSTS)};
        // real_t                    j_ {(real_t)(static_cast<int>(j) - N_GHOSTS)};
        // real_t                    r_min {mblock.metric.x1_min};
        // coord_t<Dimension::TWO_D> rth_;
        // // dipole
        // real_t br, btheta;
        // // Br
        // mblock.metric.x_Code2Sph({i_, j_ + HALF}, rth_);
        // br = TWO * math::cos(rth_[1]) / CUBE(rth_[0] / r_min);
        // // Btheta
        // mblock.metric.x_Code2Sph({i_ + HALF, j_}, rth_);
        // btheta = math::sin(rth_[1]) / CUBE(rth_[0] / r_min);

        // vec_t<Dimension::THREE_D> b_cntrv;
        // // @comment not quite true (need to separate for each component)
        // mblock.metric.v_Hat2Cntrv({i_ + HALF, j_ + HALF}, {br, btheta, ZERO}, b_cntrv);
        mblock.em(i, j, em::bx1) = ZERO; // b_cntrv[0];
        mblock.em(i, j, em::bx2) = ZERO; // b_cntrv[1];

        // rotating monopole
        // real_t                    br, bphi, etheta;
        //// Etheta
        // mblock.metric.x_Code2Sph({i_, j_ + HALF}, rth_);
        // etheta = -0.05 * (r_min / rth_[0]) * math::sin(rth_[1]);

        // vec_t<Dimension::THREE_D> cntrv;
        // mblock.metric.v_Hat2Cntrv({i_, j_ + HALF}, {ZERO, etheta, ZERO}, cntrv);
        // mblock.em(i, j, em::ex2) = cntrv[1];

        //// Br
        // mblock.metric.x_Code2Sph({i_, j_ + HALF}, rth_);
        // br = SQR(r_min / rth_[0]);

        // mblock.metric.x_Code2Sph({i_, j_ + HALF}, rth_);
        // bphi = -0.05 * (r_min / rth_[0]) * math::sin(rth_[1]);

        // mblock.metric.v_Hat2Cntrv({i_, j_ + HALF}, {br, ZERO, bphi}, cntrv);
        // mblock.em(i, j, em::bx1) = cntrv[0];

        //// Bphi
        // mblock.metric.x_Code2Sph({i_ + HALF, j_ + HALF}, rth_);
        // br = SQR(r_min / rth_[0]);

        // mblock.metric.x_Code2Sph({i_ + HALF, j_ + HALF}, rth_);
        // bphi = -0.05 * (r_min / rth_[0]) * math::sin(rth_[1]);

        // mblock.metric.v_Hat2Cntrv({i_ + HALF, j_ + HALF}, {br, ZERO, bphi}, cntrv);
        // mblock.em(i, j, em::bx3) = cntrv[2];
      });
  }

  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::PIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::PIC>& mblock) {
    using index_t = const std::size_t;
    Kokkos::parallel_for(
      "userInitPrtls", NTTRange<Dimension::ONE_D>({0}, {1}), Lambda(index_t p) {
        coord_t<Dimension::TWO_D> x {5.0, constant::PI * 0.5}, x_CU;
        mblock.metric.x_Sph2Code(x, x_CU);
        mblock.metric.x_Code2Sph(x_CU, x);
        auto [i1, dx1] = mblock.metric.CU_to_Idi(x_CU[0]);
        auto [i2, dx2] = mblock.metric.CU_to_Idi(x_CU[1]);
        // electron
        mblock.particles[0].i1(p)  = i1;
        mblock.particles[0].i2(p)  = i2;
        mblock.particles[0].dx1(p) = dx1;
        mblock.particles[0].dx2(p) = dx2;
        // mblock.particles[0].ux1(p) = 0.2;
        // mblock.particles[0].ux2(p) = 0.1;
        //  positron
        mblock.particles[1].i1(p)  = i1;
        mblock.particles[1].i2(p)  = i2;
        mblock.particles[1].dx1(p) = dx1;
        mblock.particles[1].dx2(p) = dx2;
        // mblock.particles[1].ux1(p) = 0.2;
        // mblock.particles[1].ux2(p) = 0.1;
        // ion
        mblock.particles[2].i1(p)  = i1;
        mblock.particles[2].i2(p)  = i2;
        mblock.particles[2].dx1(p) = dx1;
        mblock.particles[2].dx2(p) = dx2;
        // mblock.particles[2].ux1(p) = 0.2;
        // mblock.particles[2].ux2(p) = 0.1;
        // photon
        mblock.particles[3].i1(p) = i1;
        mblock.particles[3].i2(p) = i2;
        mblock.particles[3].dx1(p) = dx1;
        mblock.particles[3].dx2(p) = dx2;
        // mblock.particles[3].ux1(p) = 1.0;
        // mblock.particles[3].ux2(p) = 1.0;
        mblock.particles[3].ux3(p) = 1.0;
      });
    mblock.particles[0].set_npart(1);
    mblock.particles[1].set_npart(1);
    mblock.particles[2].set_npart(1);
    // mblock.particles[3].set_npart(1);
  }
#endif
  // 1D
  template <>
  void ProblemGenerator<Dimension::ONE_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::ONE_D, SimulationType::PIC>&) {}
  template <>
  void ProblemGenerator<Dimension::ONE_D, SimulationType::PIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dimension::ONE_D, SimulationType::PIC>&) {}

  // 3D
  template <>
  void ProblemGenerator<Dimension::THREE_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::THREE_D, SimulationType::PIC>&) {}
  template <>
  void ProblemGenerator<Dimension::THREE_D, SimulationType::PIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dimension::THREE_D, SimulationType::PIC>&) {}

} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;
