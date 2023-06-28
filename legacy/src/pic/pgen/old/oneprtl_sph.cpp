#include "wrapper.h"
#include "io/input.h"
#include "sim_params.h"
#include "meshblock/meshblock.h"
#include "particle_macros.h"

#include "problem_generator.hpp"

#include <cmath>

namespace ntt {

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserInitFields(const SimulationParams&,
                                                       Meshblock<Dim2, TypePIC>& mblock) {

    Kokkos::parallel_for(
      "UserInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::bx1) = ZERO;
        mblock.em(i, j, em::bx2) = ZERO;
      });
  }

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserBCFields(const real_t&,
                                                     const SimulationParams&,
                                                     Meshblock<Dim2, TypePIC>& mblock) {
    Kokkos::parallel_for(
      "2d_bc_rmin",
      CreateRangePolicy<Dim2>({N_GHOSTS, 0}, {N_GHOSTS + 2, mblock.i2_max() + N_GHOSTS}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::bx1) = ZERO;
        mblock.em(i, j, em::ex2) = ZERO;
        mblock.em(i, j, em::ex3) = ZERO;
      });
  }

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserInitParticles(const SimulationParams&,
                                                          Meshblock<Dim2, TypePIC>& mblock) {
    auto& electrons = mblock.particles[0];
    auto& positrons = mblock.particles[1];
    Kokkos::parallel_for(
      "UserInitPrtls", CreateRangePolicy<Dim1>({0}, {1}), Lambda(index_t p) {
        init_prtl_2d_Sph(mblock, electrons, p, 3.0, ntt::constant::PI * 0.002, 0.0, 0.0, 0.0);
        init_prtl_2d_Sph(mblock, positrons, p, 3.0, ntt::constant::PI * 0.002, 0.0, 0.0, 0.0);
      });
    mblock.particles[0].setNpart(1);
    mblock.particles[1].setNpart(1);
  }

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserDriveParticles(const real_t& t,
                                                           const SimulationParams&,
                                                           Meshblock<Dim2, TypePIC>& mblock) {
    real_t dt = mblock.timestep();
    if (t < 400 * dt) {
      auto electron = mblock.particles[0];
      Kokkos::parallel_for(
        "UserDrivePrtls", CreateRangePolicy<Dim1>({0}, {1}), Lambda(index_t p) {
          real_t vel      = 2.0 * (math::tanh((t - 350.0 * dt) / (100.0 * dt)) + 1.0) / 2.0;
          electron.ux1(p) = vel * math::sin(constant::PI * 0.25);
          electron.ux3(p) = vel * math::cos(constant::PI * 0.25);
        });
    }
  }

  // 1D
  template <>
  void ProblemGenerator<Dim1, TypePIC>::UserInitFields(const SimulationParams&,
                                                       Meshblock<Dim1, TypePIC>&) {}
  template <>
  void ProblemGenerator<Dim1, TypePIC>::UserInitParticles(const SimulationParams&,
                                                          Meshblock<Dim1, TypePIC>&) {}
  template <>
  void ProblemGenerator<Dim1, TypePIC>::UserBCFields(const real_t&,
                                                     const SimulationParams&,
                                                     Meshblock<Dim1, TypePIC>&) {}
  template <>
  void ProblemGenerator<Dim1, TypePIC>::UserDriveParticles(const real_t&,
                                                           const SimulationParams&,
                                                           Meshblock<Dim1, TypePIC>&) {}

  // 3D
  template <>
  void ProblemGenerator<Dim3, TypePIC>::UserInitFields(const SimulationParams&,
                                                       Meshblock<Dim3, TypePIC>&) {}
  template <>
  void ProblemGenerator<Dim3, TypePIC>::UserInitParticles(const SimulationParams&,
                                                          Meshblock<Dim3, TypePIC>&) {}
  template <>
  void ProblemGenerator<Dim3, TypePIC>::UserBCFields(const real_t&,
                                                     const SimulationParams&,
                                                     Meshblock<Dim3, TypePIC>&) {}
  template <>
  void ProblemGenerator<Dim3, TypePIC>::UserDriveParticles(const real_t&,
                                                           const SimulationParams&,
                                                           Meshblock<Dim3, TypePIC>&) {}

} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dim1, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim2, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim3, ntt::TypePIC>;

// real_t                    i_ {(real_t)(static_cast<int>(i) - N_GHOSTS)};
// real_t                    j_ {(real_t)(static_cast<int>(j) - N_GHOSTS)};
// real_t                    r_min {mblock.metric.x1_min};
// coord_t<Dim2> rth_;
// // dipole
// real_t br, btheta;
// // Br
// mblock.metric.x_Code2Sph({i_, j_ + HALF}, rth_);
// br = TWO * math::cos(rth_[1]) / CUBE(rth_[0] / r_min);
// // Btheta
// mblock.metric.x_Code2Sph({i_ + HALF, j_}, rth_);
// btheta = math::sin(rth_[1]) / CUBE(rth_[0] / r_min);

// vec_t<Dim3> b_cntrv;
// // @comment not quite true (need to separate for each component)
// mblock.metric.v_Hat2Cntrv({i_ + HALF, j_ + HALF}, {br, btheta, ZERO}, b_cntrv);
// mblock.em(i, j, em::bx1) = b_cntrv[0];
// mblock.em(i, j, em::bx2) = b_cntrv[1];

// rotating monopole
// real_t                    br, bphi, etheta;
//// Etheta
// mblock.metric.x_Code2Sph({i_, j_ + HALF}, rth_);
// etheta = -0.05 * (r_min / rth_[0]) * math::sin(rth_[1]);

// vec_t<Dim3> cntrv;
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