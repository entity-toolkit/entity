#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t bsurf, real_t rstar) : Bsurf { bsurf }, Rstar { rstar } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return Bsurf * math::cos(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return Bsurf * HALF * math::sin(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
    }

  private:
    const real_t Bsurf, Rstar;
  };

  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(real_t time, real_t bsurf, real_t rstar, real_t omega)
      : InitFields<D> { bsurf, rstar }
      , time { time }
      , Omega { omega } {}

    using InitFields<D>::bx1;
    using InitFields<D>::bx2;

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      // return Omega * bx2(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]) * (538.1679523882938/(538.1679523882938
      // + math::cosh(48.86921905584123 - 80.*x_Ph[1])));
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      // return -Omega * bx1(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]) * (538.1679523882938/(538.1679523882938
      // + math::cosh(48.86921905584123 - 80.*x_Ph[1])));
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t time, Omega;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Spherical, Metric::QSpherical>::value
    };
    static constexpr auto dimensions { traits::compatible_with<Dim::_2D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const Metadomain<S, M>& global_domain;

    const real_t  Bsurf, Rstar, Omega;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , global_domain { m }
      , Bsurf { p.template get<real_t>("setup.Bsurf", ONE) }
      , Rstar { m.mesh().extent(in::x1).first }
      , Omega { static_cast<real_t>(constant::TWO_PI) /
                p.template get<real_t>("setup.period", ONE) }
      , init_flds { Bsurf, Rstar } {}

    inline PGen() {}

    inline void InitPrtls(Domain<S, M>& local_domain) {

      std::vector<real_t> x1s, y1s, z1s, ux1s, uy1s, uz1s;
      std::vector<real_t> x2s, y2s, z2s, ux2s, uy2s, uz2s;
      x1s.push_back(2.0);
      y1s.push_back(1.0);
      z1s.push_back(ZERO);
      ux1s.push_back(ZERO);
      uy1s.push_back(0.5);
      uz1s.push_back(ZERO);
      x2s.push_back(2.0);
      y2s.push_back(1.0);
      z2s.push_back(ZERO);
      ux2s.push_back(-ONE);
      uy2s.push_back(ONE);
      uz2s.push_back(ZERO);

      const std::map<std::string, std::vector<real_t>> data_1 {
        { "x1",  x1s},
        { "x2",  y1s},
        {"phi",  z1s},
        {"ux1", ux1s},
        {"ux2", uy1s},
        {"ux3", uz1s}
      };
      const std::map<std::string, std::vector<real_t>> data_2 {
        { "x1",  x2s},
        { "x2",  y2s},
        {"phi",  z2s},
        {"ux1", ux2s},
        {"ux2", uy2s},
        {"ux3", uz2s}
      };

      arch::InjectGlobally<S, M>(global_domain, local_domain, (arch::spidx_t)1, data_1);
      arch::InjectGlobally<S, M>(global_domain, local_domain, (arch::spidx_t)2, data_2);
    }

    auto FieldDriver(real_t time) const -> DriveFields<D> {
      return DriveFields<D> {
        time,
        Bsurf,
        Rstar,
        Omega *
          SQR(SQR(math::sin(0.25 * time * static_cast<real_t>(constant::TWO_PI))))
      };
    }
  };

} // namespace user

#endif
