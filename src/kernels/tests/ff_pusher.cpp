#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"
#include "utils/comparators.h"

#include "metrics/flux_surface.h"

#include "kernels/particle_pusher_1D_gr.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

#define DERIVATIVE(func, x)                                               \
  ((func({ x + epsilon }) - func({ x - epsilon })) /         \
   (TWO * epsilon))

inline static constexpr auto eps = std::numeric_limits<real_t>::epsilon();

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

template <typename T>
void put_value(array_t<T*>& arr, T v, index_t p) {
  auto h = Kokkos::create_mirror_view(arr);
  Kokkos::deep_copy(h, arr);
  h(p) = v;
  Kokkos::deep_copy(arr, h);
}

template <SimEngine::type S, typename M>
void testFFPusher(const std::vector<std::size_t>&      res,
                   const boundaries_t<real_t>&          ext,
                   const real_t                         acc,
                   const std::map<std::string, real_t>& params = {}) {
  static_assert(M::Dim == 1, "Only 1D is supported");
  errorIf(res.size() != M::Dim, "res.size() != M::Dim");

  boundaries_t<real_t> extent;
  extent = ext;

  M metric { res, extent, params };
  
  static constexpr auto D = M::Dim;

  const int nx1 = res[0];

  auto coeff = real_t { 1.0 };
  auto dt    = real_t { 0.01 };

  const auto range_ext = CreateRangePolicy<Dim::_1D>(
    { 0},
    { res[0] + 2 * N_GHOSTS });

  auto efield = ndfield_t<Dim::_1D, 1> { "efield",
                                          res[0] + 2 * N_GHOSTS};

  Kokkos::parallel_for(
    "init efield",
    range_ext,
    Lambda(index_t i1) {
      efield(i1, em::ex1) = 1.92;
    });

  
  array_t<int*>      i1 { "i1", 2 };
  array_t<int*>      i1_prev { "i1_prev", 2 };
  array_t<prtldx_t*> dx1 { "dx1", 2 };
  array_t<prtldx_t*> dx1_prev { "dx1_prev", 2 };
  array_t<real_t*>   ux1 { "ux1", 2 };
  array_t<real_t*>   ux2 { "ux2", 2 };
  array_t<real_t*>   ux3 { "ux3", 2 };
  array_t<short*>    tag { "tag", 2 };

  put_value<int>(i1, 5, 0);
  put_value<prtldx_t>(dx1, (prtldx_t)(0.15), 0);
  put_value<real_t>(ux1, ZERO, 0);
  put_value<real_t>(ux3, metric.OmegaF(), 0);
  put_value<short>(tag, ParticleTag::alive, 0);

  put_value<int>(i1, 5, 1);
  put_value<prtldx_t>(dx1, (prtldx_t)(0.15), 1);
  put_value<real_t>(ux1, ZERO, 1);
  put_value<real_t>(ux3, metric.OmegaF(), 1);
  put_value<short>(tag, ParticleTag::alive, 1);


  // Particle boundaries
  auto boundaries = boundaries_t<PrtlBC> {};
  boundaries      = {
    {PrtlBC::PERIODIC, PrtlBC::PERIODIC}
  };



  Kokkos::parallel_for(
    "pusher",
    CreateRangePolicy<Dim::_1D>({ 0 }, { 1 }),
    kernel::gr::Pusher_kernel<FluxSurface<Dim::_1D>>(
                                                   efield,
                                                   i1,
                                                   i1_prev,
                                                   dx1,
                                                   dx1_prev,
                                                   ux1, ux2, ux3,
                                                   tag,
                                                   metric,
                                                   coeff, dt,
                                                   nx1,
                                                   eps * 1e3,
                                                   20,
                                                   boundaries));


  Kokkos::parallel_for(
    "pusher",
    1,
    kernel::gr::Pusher_kernel<FluxSurface<Dim::_1D>>(
                                                   efield,
                                                   i1,
                                                   i1_prev,
                                                   dx1,
                                                   dx1_prev,
                                                   ux1, ux2, ux3,
                                                   tag,
                                                   metric,
                                                   -coeff, dt,
                                                   nx1,
                                                   eps * 1e3,
                                                   20,
                                                   boundaries));
  
  auto epsilon = eps * 1e3;

  auto i1_      = Kokkos::create_mirror_view(i1);
  Kokkos::deep_copy(i1_, i1);
  auto i1_prev_     = Kokkos::create_mirror_view(i1_prev);
  Kokkos::deep_copy(i1_prev_, i1_prev);
  auto dx1_      = Kokkos::create_mirror_view(dx1);
  Kokkos::deep_copy(dx1_, dx1);
  auto dx1_prev_      = Kokkos::create_mirror_view(dx1_prev);
  Kokkos::deep_copy(dx1_prev_, dx1_prev);
  auto ux1_      = Kokkos::create_mirror_view(ux1);
  Kokkos::deep_copy(ux1_, ux1);
  auto ux2_      = Kokkos::create_mirror_view(ux2);
  Kokkos::deep_copy(ux2_, ux2);
  auto ux3_      = Kokkos::create_mirror_view(ux3);
  Kokkos::deep_copy(ux3_, ux3);
  auto efield_ = Kokkos::create_mirror_view(efield(i1(1), em::ex1));
  Kokkos::deep_copy(efield_, efield(i1(1), em::ex1));


//negative charge
  coord_t<D> xp { i_di_to_Xi(i1_(0), dx1_(0)) };
  coord_t<D> xp_prev { i_di_to_Xi(i1_prev_(0), dx1_prev_(0)) };
  vec_t<Dim::_3D> u_d {ux1_(0), ux2_(0), ux3_(0)};
  vec_t<Dim::_3D> u_u { ZERO };
  metric.template transform<Idx::D, Idx::U>(xp, u_d, u_u);

  real_t u0 { (u_u[2] - 
        u_u[0] * metric.f1(xi) * metric.template h<3, 3>(xi) / (metric.OmegaF() + metric.beta3(xi))
        ) / metric.OmegaF() };
  real_t vp { u_u[0] / u0 };

  real_t left {  u0 * (metric.f2(xp) * vp + metric.f1(xp)) - (metric.f2(xp_prev) * vp + metric.f1(xp_prev))  };
  real_t right { dt * (coeff * metric.alpha(xp_prev) * metric.template h_<1, 1>(xp_prev) * 1.92 -
                            u0 * metric.alpha(xp_prev) * DERIVATIVE(metric.alpha, xp_prev[0]) +
                            HALF * u0 * (DERIVATIVE(metric.f2, xp_prev[0]) * SQR(vp) +
                                         TWO * DERIVATIVE(metric.f1, xp_prev[0]) * vp +
                                         DERIVATIVE(metric.f0, xp_prev[0]))) };
  real_t ratio { left / right };

  unsigned int wrong { 0 };
  if (not cmp::AlmostEqual(ratio, ONE, eps * acc)) {
      printf("%.12e != 1 at negative charge\n", ratio);
      ++wrong;
    }

//positive charge
  xp[0] = i_di_to_Xi(i1_(1), dx1_(1));
  xp_prev[0] = i_di_to_Xi(i1_prev_(1), dx1_prev_(1));
  u_d[0] = ux1_(1);
  u_d[1] = ux2_(1);
  u_d[2] = ux3_(1);
  metric.template transform<Idx::D, Idx::U>(xp, u_d, u_u);

  u0 = (u_u[2] - 
        u_u[0] * metric.f1(xi) * metric.template h<3, 3>(xi) / (metric.OmegaF() + metric.beta3(xi))
        ) / metric.OmegaF() ;
  vp = u_u[0] / u0 ;

  left = u0 * (metric.f2(xp) * vp + metric.f1(xp)) - (metric.f2(xp_prev) * vp + metric.f1(xp_prev));
  right = dt * (-coeff * metric.alpha(xp) * metric.template h_<1, 1>(xp) * 1.92 -
                            u0 * metric.alpha(xp) * DERIVATIVE(metric.alpha, xp[0]) +
                            HALF * u0 * (DERIVATIVE(metric.f2, xp[0]) * SQR(vp) +
                                         TWO * DERIVATIVE(metric.f1, xp[0]) * vp +
                                         DERIVATIVE(metric.f0, xp[0])));
  ratio = left / right;

    if (not cmp::AlmostEqual(ratio, ONE, eps * acc)) {
      printf("%.12e != 1 at positive charge\n", ratio);
      ++wrong;
    }

    if (not wrong){
      throw std::runtime_error("ff_usher failed.");
    }

}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;

    testFFPusher<SimEngine::GRPIC, FluxSurface<Dim::_1D>>(
      { 128 },
      { { 2.0, 50.0 } },
      100,
      { { "a", (real_t)0.95 } , 
        { "psi0", (real_t)1.0 } , 
        { "theta0", (real_t)1.0 } , 
        { "Omega", (real_t)0.5 } ,
        { "pCur", (real_t)3.1 }  });

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}

#undef DERIVATIVE
#undef i_di_to_Xi