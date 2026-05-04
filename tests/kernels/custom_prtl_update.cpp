#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/policies.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"

#include "framework/containers/particles.h"
#include "kernels/pushers/context.h"
#include "kernels/pushers/sr.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;

void errorIf(bool condition, const std::string& message = "") {
  if (condition) {
    throw std::runtime_error(message);
  }
}

auto equal(real_t a, real_t b, const std::string& msg) -> bool {
  if (not(math::abs(a - b) < 1e-4)) {
    printf("%.12e != %.12e %s\n", a, b, msg.c_str());
    return false;
  }
  return true;
}

template <typename T>
void put_value(array_t<T*>& arr, T v, index_t p) {
  auto h = Kokkos::create_mirror_view(arr);
  Kokkos::deep_copy(h, arr);
  h(p) = v;
  Kokkos::deep_copy(arr, h);
}

template <MetricClass M>
struct TestCustomPrtlUpdate {
  Inline void operator()(index_t                          p,
                         const kernel::sr::PusherContext& ctx,
                         const kernel::sr::PusherBoundaries<M::Dim>&,
                         const ntt::ParticleArrays& particles,
                         const M& /*metric*/) const {
    if constexpr (M::Dim == Dim::_1D || M::Dim == Dim::_2D || M::Dim == Dim::_3D) {
      auto invert_vel = false;
      if (particles.i1(p) < 0) {
        particles.i1(p)  = 0;
        particles.dx1(p) = ONE - particles.dx1(p);
        invert_vel       = true;
      } else if (particles.i1(p) >= ctx.ni1) {
        particles.i1(p)  = ctx.ni1 - 1;
        particles.dx1(p) = ONE - particles.dx1(p);
        invert_vel       = true;
      }
      if (invert_vel) {
        particles.ux1(p) = -particles.ux1(p);
      }
    }
    if constexpr (M::Dim == Dim::_2D || M::Dim == Dim::_3D) {
      auto invert_vel = false;
      if (particles.i2(p) < 0) {
        particles.i2(p)  = 0;
        particles.dx2(p) = ONE - particles.dx2(p);
        invert_vel       = true;
      } else if (particles.i2(p) >= ctx.ni2) {
        particles.i2(p)  = ctx.ni2 - 1;
        particles.dx2(p) = ONE - particles.dx2(p);
        invert_vel       = true;
      }
      if (invert_vel) {
        particles.ux2(p) = -particles.ux2(p);
      }
    }
    if constexpr (M::Dim == Dim::_3D) {
      auto invert_vel = false;
      if (particles.i3(p) < 0) {
        particles.i3(p)  = 0;
        particles.dx3(p) = ONE - particles.dx3(p);
        invert_vel       = true;
      } else if (particles.i3(p) >= ctx.ni3) {
        particles.i3(p)  = ctx.ni3 - 1;
        particles.dx3(p) = ONE - particles.dx3(p);
        invert_vel       = true;
      }
      if (invert_vel) {
        particles.ux3(p) = -particles.ux3(p);
      }
    }
  }
};

template <SimEngine::type S, typename M>
void testCustomPrtlUpdate(const std::vector<std::size_t>&      res,
                          const boundaries_t<real_t>&          ext,
                          const std::map<std::string, real_t>& params = {}) {

  errorIf(res.size() != M::Dim, "res.size() != M::Dim");
  errorIf(M::CoordType != Coord::Cartesian, "M::CoordType != Coord::Cartesian");

  M metric { res, ext, params };

  int nx1 = 0, nx2 = 0, nx3 = 0;
  if (res.size() > 0) {
    nx1 = static_cast<int>(res.at(0));
  }
  if (res.size() > 1) {
    nx2 = static_cast<int>(res.at(1));
  }
  if (res.size() > 2) {
    nx3 = static_cast<int>(res.at(2));
  }

  const real_t dt = static_cast<real_t>(0.1) *
                    (ext.at(0).second - ext.at(0).first) /
                    static_cast<real_t>(nx1);

  ndfield_t<M::Dim, 6> emfield;
  if constexpr (M::Dim == Dim::_1D) {
    emfield = ndfield_t<M::Dim, 6> { "emfield", res.at(0) + 2 * N_GHOSTS };
  } else if constexpr (M::Dim == Dim::_2D) {
    emfield = ndfield_t<M::Dim, 6> { "emfield",
                                     res.at(0) + 2 * N_GHOSTS,
                                     res.at(1) + 2 * N_GHOSTS };
  } else {
    emfield = ndfield_t<M::Dim, 6> { "emfield",
                                     res.at(0) + 2 * N_GHOSTS,
                                     res.at(1) + 2 * N_GHOSTS,
                                     res.at(2) + 2 * N_GHOSTS };
  }

  // Arrays for REFLECT boundaries
  array_t<int*> r_i1 { "r_i1", 1 }, r_i2 { "r_i2", 1 }, r_i3 { "r_i3", 1 };
  array_t<int*> r_i1_prev { "r_i1_prev", 1 }, r_i2_prev { "r_i2_prev", 1 },
    r_i3_prev { "r_i3_prev", 1 };
  array_t<prtldx_t*> r_dx1 { "r_dx1", 1 }, r_dx2 { "r_dx2", 1 },
    r_dx3 { "r_dx3", 1 };
  array_t<prtldx_t*> r_dx1_prev { "r_dx1_prev", 1 },
    r_dx2_prev { "r_dx2_prev", 1 }, r_dx3_prev { "r_dx3_prev", 1 };
  array_t<real_t*> r_ux1 { "r_ux1", 1 }, r_ux2 { "r_ux2", 1 },
    r_ux3 { "r_ux3", 1 };
  array_t<short*> r_tag { "r_tag", 1 };

  // Arrays for CUSTOM boundaries
  array_t<int*> c_i1 { "c_i1", 1 }, c_i2 { "c_i2", 1 }, c_i3 { "c_i3", 1 };
  array_t<int*> c_i1_prev { "c_i1_prev", 1 }, c_i2_prev { "c_i2_prev", 1 },
    c_i3_prev { "c_i3_prev", 1 };
  array_t<prtldx_t*> c_dx1 { "c_dx1", 1 }, c_dx2 { "c_dx2", 1 },
    c_dx3 { "c_dx3", 1 };
  array_t<prtldx_t*> c_dx1_prev { "c_dx1_prev", 1 },
    c_dx2_prev { "c_dx2_prev", 1 }, c_dx3_prev { "c_dx3_prev", 1 };
  array_t<real_t*> c_ux1 { "c_ux1", 1 }, c_ux2 { "c_ux2", 1 },
    c_ux3 { "c_ux3", 1 };
  array_t<short*> c_tag { "c_tag", 1 };

  array_t<real_t*> phi;

  real_t ux_1 = 0.569197;
  real_t uy_1 = 0.716085;
  real_t uz_1 = -0.760101;
  put_value<real_t>(r_ux1, ux_1, 0);
  put_value<real_t>(c_ux1, ux_1, 0);
  put_value<real_t>(r_ux2, uy_1, 0);
  put_value<real_t>(c_ux2, uy_1, 0);
  put_value<real_t>(r_ux3, uz_1, 0);
  put_value<real_t>(c_ux3, uz_1, 0);

  put_value<short>(r_tag, ParticleTag::alive, 0);
  put_value<short>(c_tag, ParticleTag::alive, 0);

  if constexpr (M::PrtlDim >= Dim::_1D) {
    put_value<int>(r_i1, nx1 - 1, 0);
    put_value<int>(c_i1, nx1 - 1, 0);
    put_value<prtldx_t>(r_dx1, 0.9, 0);
    put_value<prtldx_t>(c_dx1, 0.9, 0);
  }
  if constexpr (M::PrtlDim >= Dim::_2D) {
    put_value<int>(r_i2, nx2 - 1, 0);
    put_value<int>(c_i2, nx2 - 1, 0);
    put_value<prtldx_t>(r_dx2, 0.9, 0);
    put_value<prtldx_t>(c_dx2, 0.9, 0);
  }
  if constexpr (M::PrtlDim == Dim::_3D) {
    put_value<int>(r_i3, 0, 0);
    put_value<int>(c_i3, 0, 0);
    put_value<prtldx_t>(r_dx3, 0.1, 0);
    put_value<prtldx_t>(c_dx3, 0.1, 0);
  }

  kernel::sr::PusherBoundaries<M::Dim> boundaries {
    { { PrtlBC::REFLECT, PrtlBC::REFLECT },
     { PrtlBC::REFLECT, PrtlBC::REFLECT },
     { PrtlBC::REFLECT, PrtlBC::REFLECT } }
  };

  ntt::ParticleArrays r_arrays { 1u };
  r_arrays.i1       = r_i1;
  r_arrays.i2       = r_i2;
  r_arrays.i3       = r_i3;
  r_arrays.i1_prev  = r_i1_prev;
  r_arrays.i2_prev  = r_i2_prev;
  r_arrays.i3_prev  = r_i3_prev;
  r_arrays.dx1      = r_dx1;
  r_arrays.dx2      = r_dx2;
  r_arrays.dx3      = r_dx3;
  r_arrays.dx1_prev = r_dx1_prev;
  r_arrays.dx2_prev = r_dx2_prev;
  r_arrays.dx3_prev = r_dx3_prev;
  r_arrays.ux1      = r_ux1;
  r_arrays.ux2      = r_ux2;
  r_arrays.ux3      = r_ux3;
  r_arrays.phi      = phi;
  r_arrays.tag      = r_tag;

  ntt::ParticleArrays c_arrays { 2u };
  c_arrays.i1       = c_i1;
  c_arrays.i2       = c_i2;
  c_arrays.i3       = c_i3;
  c_arrays.i1_prev  = c_i1_prev;
  c_arrays.i2_prev  = c_i2_prev;
  c_arrays.i3_prev  = c_i3_prev;
  c_arrays.dx1      = c_dx1;
  c_arrays.dx2      = c_dx2;
  c_arrays.dx3      = c_dx3;
  c_arrays.dx1_prev = c_dx1_prev;
  c_arrays.dx2_prev = c_dx2_prev;
  c_arrays.dx3_prev = c_dx3_prev;
  c_arrays.ux1      = c_ux1;
  c_arrays.ux2      = c_ux2;
  c_arrays.ux3      = c_ux3;
  c_arrays.phi      = phi;
  c_arrays.tag      = c_tag;

  const auto custom_update_policy =
    ::kernel::sr::PusherPolicy<M,
                               ::traits::emission::NoPolicy_t,
                               TestCustomPrtlUpdate<M>,
                               ::traits::extfields::NoPolicy_t,
                               false> {};

  static constexpr auto n_iter = 100;

  for (auto n { 0 }; n < n_iter; ++n) {
    Kokkos::parallel_for("pusher_reflect",
                         CreateRangePolicy<Dim::_1D>({ 0 }, { 1 }),
                         kernel::sr::Pusher_kernel<M>({ 1u,
                                                        ParticlePusher::BORIS,
                                                        RadiativeDrag::NONE,
                                                        1.f,
                                                        1.f,
                                                        dt * static_cast<real_t>(n),
                                                        dt,
                                                        ONE,
                                                        nx1,
                                                        nx2,
                                                        nx3 },
                                                      boundaries,
                                                      r_arrays,
                                                      emfield,
                                                      metric));

    Kokkos::parallel_for(
      "pusher_custom",
      CreateRangePolicy<Dim::_1D>({ 0 }, { 1 }),
      kernel::sr::Pusher_kernel<M, decltype(custom_update_policy)>(
        { 2u,
          ParticlePusher::BORIS,
          RadiativeDrag::NONE,
          1.f,
          1.f,
          dt * static_cast<real_t>(n),
          dt,
          ONE,
          nx1,
          nx2,
          nx3 },
        boundaries,
        c_arrays,
        emfield,
        metric,
        custom_update_policy));

    auto hr_i1 = Kokkos::create_mirror_view(r_i1);
    Kokkos::deep_copy(hr_i1, r_i1);
    auto hc_i1 = Kokkos::create_mirror_view(c_i1);
    Kokkos::deep_copy(hc_i1, c_i1);
    auto hr_dx1 = Kokkos::create_mirror_view(r_dx1);
    Kokkos::deep_copy(hr_dx1, r_dx1);
    auto hc_dx1 = Kokkos::create_mirror_view(c_dx1);
    Kokkos::deep_copy(hc_dx1, c_dx1);
    auto hr_ux1 = Kokkos::create_mirror_view(r_ux1);
    Kokkos::deep_copy(hr_ux1, r_ux1);
    auto hc_ux1 = Kokkos::create_mirror_view(c_ux1);
    Kokkos::deep_copy(hc_ux1, c_ux1);

    if constexpr (M::PrtlDim >= Dim::_1D) {
      errorIf(hr_i1(0) != hc_i1(0), "i1 mismatch");
      errorIf(not equal(hr_dx1(0), hc_dx1(0), "dx1 mismatch"));
      errorIf(not equal(hr_ux1(0), hc_ux1(0), "ux1 mismatch"));
    }

    if constexpr (M::PrtlDim >= Dim::_2D) {
      auto hr_i2 = Kokkos::create_mirror_view(r_i2);
      Kokkos::deep_copy(hr_i2, r_i2);
      auto hc_i2 = Kokkos::create_mirror_view(c_i2);
      Kokkos::deep_copy(hc_i2, c_i2);
      auto hr_dx2 = Kokkos::create_mirror_view(r_dx2);
      Kokkos::deep_copy(hr_dx2, r_dx2);
      auto hc_dx2 = Kokkos::create_mirror_view(c_dx2);
      Kokkos::deep_copy(hc_dx2, c_dx2);
      auto hr_ux2 = Kokkos::create_mirror_view(r_ux2);
      Kokkos::deep_copy(hr_ux2, r_ux2);
      auto hc_ux2 = Kokkos::create_mirror_view(c_ux2);
      Kokkos::deep_copy(hc_ux2, c_ux2);
      errorIf(hr_i2(0) != hc_i2(0), "i2 mismatch");
      errorIf(not equal(hr_dx2(0), hc_dx2(0), "dx2 mismatch"));
      errorIf(not equal(hr_ux2(0), hc_ux2(0), "ux2 mismatch"));
    }

    if constexpr (M::PrtlDim == Dim::_3D) {
      auto hr_i3 = Kokkos::create_mirror_view(r_i3);
      Kokkos::deep_copy(hr_i3, r_i3);
      auto hc_i3 = Kokkos::create_mirror_view(c_i3);
      Kokkos::deep_copy(hc_i3, c_i3);
      auto hr_dx3 = Kokkos::create_mirror_view(r_dx3);
      Kokkos::deep_copy(hr_dx3, r_dx3);
      auto hc_dx3 = Kokkos::create_mirror_view(c_dx3);
      Kokkos::deep_copy(hc_dx3, c_dx3);
      auto hr_ux3 = Kokkos::create_mirror_view(r_ux3);
      Kokkos::deep_copy(hr_ux3, r_ux3);
      auto hc_ux3 = Kokkos::create_mirror_view(c_ux3);
      Kokkos::deep_copy(hc_ux3, c_ux3);
      errorIf(hr_i3(0) != hc_i3(0), "i3 mismatch");
      errorIf(not equal(hr_dx3(0), hc_dx3(0), "dx3 mismatch"));
      errorIf(not equal(hr_ux3(0), hc_ux3(0), "ux3 mismatch"));
    }
  }
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;

    const std::vector<std::size_t> res1d { 50 };
    const boundaries_t<real_t>     ext1d {
          { 0.0, 1000.0 }
    };
    const std::vector<std::size_t> res2d { 30, 20 };
    const boundaries_t<real_t>     ext2d {
          { -15.0, 15.0 },
          { -10.0, 10.0 }
    };
    const std::vector<std::size_t> res3d { 10, 10, 10 };
    const boundaries_t<real_t>     ext3d {
          { 0.0, 1.0 },
          { 0.0, 1.0 },
          { 0.0, 1.0 }
    };

    testCustomPrtlUpdate<SimEngine::SRPIC, Minkowski<Dim::_1D>>(res1d, ext1d, {});
    testCustomPrtlUpdate<SimEngine::SRPIC, Minkowski<Dim::_2D>>(res2d, ext2d, {});
    testCustomPrtlUpdate<SimEngine::SRPIC, Minkowski<Dim::_3D>>(res3d, ext3d, {});

  } catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
