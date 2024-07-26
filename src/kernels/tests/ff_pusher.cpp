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

#define from_Xi_to_i(XI, I)                                                    \
  { I = static_cast<int>((XI)); }

#define from_Xi_to_i_di(XI, I, DI)                                             \
  {                                                                            \
    from_Xi_to_i((XI), (I));                                                   \
    DI = static_cast<prtldx_t>((XI)) - static_cast<prtldx_t>(I);               \
  }


#define DERIVATIVE(func, x)                                               \
  ((func({ x + epsilon }) - func({ x - epsilon })) /         \
   (TWO * epsilon))

inline static constexpr auto epsilon = std::numeric_limits<real_t>::epsilon();


Inline auto equal(const real_t& a,
                  const real_t& b,
                  const char*     msg,
                  const real_t    acc = ONE) -> bool {
  const auto eps = math::sqrt(epsilon) * acc;
  if (not cmp::AlmostEqual(a, b, eps)) {
    printf("%.12e != %.12e %s\n", a, b, msg);
    return false;
  }
  return true;
}


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
  
  const real_t d_eta_inv { nx1 / (metric.r2eta(extent[0].second) - metric.r2eta(extent[0].first)) };

  const auto coeff { d_eta_inv };
  const auto dt    = real_t { 0.001 };

  const auto range_ext = CreateRangePolicy<Dim::_1D>(
    { 0 },
    { res[0] + 2 * N_GHOSTS });

  auto efield = ndfield_t<Dim::_1D, 1> { "efield",
                                          res[0] + 2 * N_GHOSTS};

  
  array_t<int*>      i1 { "i1", 30 };
  array_t<int*>      i1_prev { "i1_prev", 30 };
  array_t<prtldx_t*> dx1 { "dx1", 30 };
  array_t<prtldx_t*> dx1_prev { "dx1_prev", 30 };
  array_t<real_t*>   ux1 { "ux1", 30 };
  array_t<short*>    tag { "tag", 30 };
  
  const auto sep = real_t { 0.1 * res[0] };
  real_t x1i { ZERO };
  int ii { 0 };
  prtldx_t dx1i { ZERO };


  // Particle boundaries
  auto boundaries = boundaries_t<PrtlBC> {};
  boundaries      = {
    {PrtlBC::PERIODIC, PrtlBC::PERIODIC}
  };

 // No efield
 
  Kokkos::parallel_for(
    "init efield",
    range_ext,
    Lambda(index_t i1) {
      efield(i1, em::ex1) = ZERO;
    });

  for (size_t n { 0 }; n < 30; n++){
    x1i = ZERO + (n % 10) * sep;
    from_Xi_to_i_di(x1i, ii, dx1i)
    put_value<int>(i1, ii, n);
    put_value<prtldx_t>(dx1, dx1i, n);
    put_value<real_t>(ux1, -metric.f1({ x1i }) / metric.f2({ x1i }), n);
    put_value<short>(tag, ParticleTag::alive, n);
  }
  

  // Without electric field

   Kokkos::parallel_for(
    "pusher",
    CreateRangePolicy<Dim::_1D>({ 0 }, { 10 }),
    kernel::gr::Pusher_kernel<FluxSurface<Dim::_1D>>(
                                                   efield,
                                                   i1,
                                                   i1_prev,
                                                   dx1,
                                                   dx1_prev,
                                                   ux1,
                                                   tag,
                                                   metric,
                                                   -coeff, dt,
                                                   nx1,
                                                   1e-5,
                                                   30,
                                                   boundaries));
  
  // With electric field
  // positive charge
  
  Kokkos::parallel_for(
    "init efield",
    range_ext,
    Lambda(index_t i1) {
      efield(i1, em::ex1) = 99.0;
    });

   Kokkos::parallel_for(
    "pusher",
    CreateRangePolicy<Dim::_1D>({ 10 }, { 20 }),
    kernel::gr::Pusher_kernel<FluxSurface<Dim::_1D>>(
                                                   efield,
                                                   i1,
                                                   i1_prev,
                                                   dx1,
                                                   dx1_prev,
                                                   ux1,
                                                   tag,
                                                   metric,
                                                   coeff, dt,
                                                   nx1,
                                                   1e-5,
                                                   30,
                                                   boundaries));
  //negative charge 

  Kokkos::parallel_for(
    "pusher",
    CreateRangePolicy<Dim::_1D>({ 20 }, { 30 }),
    kernel::gr::Pusher_kernel<FluxSurface<Dim::_1D>>(
                                                   efield,
                                                   i1,
                                                   i1_prev,
                                                   dx1,
                                                   dx1_prev,
                                                   ux1,
                                                   tag,
                                                   metric,
                                                   -coeff, dt,
                                                   nx1,
                                                   1e-5,
                                                   30,
                                                   boundaries));

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


  
  real_t vp_exp[10] { -0.128080544611002, -0.222571219885845, -0.301212251198582, 
                      -0.358235366733387, -0.387383121052459, -0.382799931574759, 
                      -0.340833052985762, -0.263140267573464, -0.161128753912851, 
                      -0.0604347402410836 };
  real_t vp_e_p_exp[10] { -0.122286881108314, -0.215534929107648, -0.292701135574140, 
                          -0.348028532252193, -0.375337709663405, -0.368984471614021, 
                          -0.325765909536170, -0.248149662916807, -0.148698026244964, 
                          -0.0538237879835971 };
  real_t vp_e_n_exp[10] { -0.133874207868050, -0.229607510251248, -0.309723366124719, 
                          -0.368442200032320, -0.399428530458386, -0.396615388297603, 
                          -0.355900191451829, -0.278130865433766, -0.173559474415594, 
                          -0.0670456885818427 };  

  unsigned int wrongs { 0 };
  
  for (size_t n { 0 }; n < 10; ++n){
    wrongs += !equal(ux1_(n), vp_exp[n], "no efield", acc);
  }
  
  for (size_t n { 0 }; n < 10; ++n){
    wrongs += !equal(ux1_(n + 10), vp_e_p_exp[n], "positive charge", acc);
  }
  
  for (size_t n { 0 }; n < 10; ++n){
    wrongs += !equal(ux1_(n + 20), vp_e_n_exp[n], "negative charge", acc);
  }

  if (wrongs){
      throw std::runtime_error("ff_usher failed.");
  }
  

  //with electric field, positive charge
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;

    testFFPusher<SimEngine::GRPIC, FluxSurface<Dim::_1D>>(
      { 128 },
      { { 2.0, 50.0 } },
        5,
      { { "a", (real_t)0.95 } , 
        { "psi0", (real_t)1.0 } , 
        { "theta0", (real_t)1.0 } , 
        { "Omega", (real_t)0.5 } ,
        { "pCur", (real_t)3.1 }  });
      
    testFFPusher<SimEngine::GRPIC, FluxSurface<Dim::_1D>>(
      { 512 },
      { { 2.0, 50.0 } },
        5,
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
#undef from_Xi_to_i_di
#undef from_Xi_to_i
