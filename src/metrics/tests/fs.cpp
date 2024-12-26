#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"

#include "metrics/flux_surface.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

inline static constexpr auto epsilon = std::numeric_limits<real_t>::epsilon();

template <Dimension D>
Inline auto equal(const vec_t<D>& a,
                  const vec_t<D>& b,
                  const char*     msg,
                  const real_t    acc = ONE) -> bool {
  const auto eps = epsilon * acc;
  for (unsigned short d = 0; d < D; ++d) {
    if (not cmp::AlmostEqual(a[d], b[d], eps)) {
      printf("%d : %.12e != %.12e %s\n", d, a[d], b[d], msg);
      return false;
    }
  }
  return true;
}

Inline auto equal(const real_t& a,
                  const real_t& b,
                  const char*     msg,
                  const real_t    acc = ONE) -> bool {
  const auto eps = epsilon * acc;
  if (not cmp::AlmostEqual(a, b, eps)) {
    printf("%.12e != %.12e %s\n", a, b, msg);
    return false;
  }
  return true;
}

template <Dimension D>
Inline void unravel(std::size_t                    idx,
                    tuple_t<std::size_t, D>&       ijk,
                    const tuple_t<std::size_t, D>& res) {
  for (unsigned short d = 0; d < D; ++d) {
    ijk[d]  = idx % res[d];
    idx    /= res[d];
  }
}

template <class M>
void testMetric(const std::vector<std::size_t>&      res,
                const boundaries_t<real_t>&          ext,
                const real_t                         acc    = ONE,
                const std::map<std::string, real_t>& params = {}) {
  static_assert(M::Dim == 1, "Dim != 1");
  errorIf(res.size() != (std::size_t)(M::Dim), "res.size() != M.dim");
  errorIf(ext.size() != (std::size_t)(M::Dim), "ext.size() != M.dim");
  for (const auto& e : ext) {
    errorIf(e.first >= e.second, "e.first >= e.second");
  }

  M                            metric(res, ext, params);
  tuple_t<std::size_t, M::Dim> res_tup;
  std::size_t                  npts = 1;
  for (auto d = 0; d < M::Dim; ++d) {
    res_tup[d]  = res[d];
    npts       *= res[d];
  }
  
  const auto x_min = ext[0].first;
  const auto x_max = ext[0].second;


  unsigned long all_wrongs = 0;
  const auto    rg         = metric.rg();
  const auto    rh         = metric.rhorizon();
  const auto    a          = metric.spin();
  const auto th = params.at("theta0");
  const auto psi0 = params.at("psi0");
  const auto pCur = params.at("pCur");
  const auto Omega = params.at("Omega") * a / (SQR(a) + SQR(rh));
  const auto rh_m = rh - TWO * math::sqrt(ONE - SQR(a));

  const auto eta_min = math::log((x_min - rh) / (x_min - rh_m)) / (rh - rh_m);
  const auto eta_max = math::log((x_max - rh) / (x_max - rh_m)) / (rh - rh_m);
  const auto d_eta = (eta_max - eta_min) / ((real_t)res[0]);

  Kokkos::parallel_reduce(
    "h_ij/hij",
    npts,
    Lambda(index_t n, unsigned long& wrongs) {
      tuple_t<std::size_t, M::Dim> idx;
      unravel<M::Dim>(n, idx, res_tup);
      coord_t<M::Dim> x_Code { ZERO };
      coord_t<M::Dim> x_Phys { ZERO };

      for (unsigned short d = 0; d < M::Dim; ++d) {
        x_Code[d] = (real_t)(idx[d]) + HALF;
      }

      const auto h11  = metric.template h<1, 1>(x_Code);
      const auto h22  = metric.template h<2, 2>(x_Code);
      const auto h33  = metric.template h<3, 3>(x_Code);
      const auto h_11 = metric.template h_<1, 1>(x_Code);
      const auto h_22 = metric.template h_<2, 2>(x_Code);
      const auto h_33 = metric.template h_<3, 3>(x_Code);

      metric.template convert<Crd::Cd, Crd::Ph>(x_Code, x_Phys);
      const auto r  = x_Phys[0];
    

      const auto Sigma = SQR(r) + SQR(a * math::cos(th));
      const auto Delta = SQR(r) - TWO * rg * r + SQR(a);
      const auto A     = SQR(SQR(r) + SQR(a)) - SQR(a * math::sin(th)) * Delta;
      const auto dpsi_r = 0;
      const auto dpsi_dtheta = psi0 * math::sin(th);
      const auto omega = TWO * a * r / A ;

      const auto h_11_expect  = Sigma / Delta;
      const auto h_22_expect  = Sigma ;
      const auto h_33_expect  = A * SQR(math::sin(th)) / Sigma;
      const auto h11_expect = ONE / h_11_expect;
      const auto h22_expect = ONE / h_22_expect;
      const auto h33_expect = ONE / h_33_expect;

      const auto f0_expect = h_33_expect * SQR(Omega - omega);
      const auto f1_expect = d_eta * A * pCur * math::sin(th) * (Omega - omega) / dpsi_dtheta;
      const auto f2_expect = SQR(d_eta) * Sigma * (Delta + A * SQR(pCur / dpsi_dtheta));

      const auto f0_predict = metric.f0(x_Code);
      const auto f1_predict = metric.f1(x_Code);
      const auto f2_predict = metric.f2(x_Code);


      vec_t<Dim::_3D> hij_temp { ZERO }, hij_predict { ZERO };
      metric.template transform<Idx::U, Idx::PU>(x_Code, { h11, h22, h33 }, hij_temp);
      metric.template transform<Idx::U, Idx::PU>(x_Code, hij_temp, hij_predict);


      vec_t<Dim::_3D> h_ij_temp { ZERO }, h_ij_predict { ZERO };
      metric.template transform<Idx::D, Idx::PD>(x_Code,
                                                 { h_11, h_22, h_33 },
                                                 h_ij_temp);
      metric.template transform<Idx::D, Idx::PD>(x_Code, h_ij_temp, h_ij_predict);


      vec_t<Dim::_3D> hij_expect { h11_expect, h22_expect, h33_expect };
      vec_t<Dim::_3D> h_ij_expect { h_11_expect, h_22_expect, h_33_expect };

      wrongs += not equal<Dim::_3D>(hij_expect, hij_predict, "hij", acc);
      wrongs += not equal(f0_expect, f0_predict, "f0", acc);
      wrongs += not equal(f1_expect, f1_predict, "f1", acc);
      wrongs += not equal(f2_expect, f2_predict, "f2", acc);

    },
    all_wrongs);

  errorIf(all_wrongs != 0,
          "wrong h_ij/hij for " + std::to_string(M::Dim) + "D " +
            std::string(metric.Label) + " with " + std::to_string(all_wrongs) +
            " errors");
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;
    using namespace metric;
    testMetric<FluxSurface<Dim::_1D>>(
      { 128 },
      { { 2.0, 50.0 } },
      30,
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
