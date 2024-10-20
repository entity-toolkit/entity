#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"
#include "kernels/particle_moments.hpp"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t bsurf, real_t rstar) : Bsurf { bsurf }, Rstar { rstar } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return Bsurf * SQR(Rstar / x_Ph[0]);
    }

  private:
    const real_t Bsurf, Rstar;
  };

  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(real_t time, real_t bsurf, real_t rstar, real_t omega, real_t comp)
      : InitFields<D> { bsurf, rstar }
      , time { time }
      , Omega { omega }
      , comp {comp}
      , OmegaLT { static_cast<real_t>(0.4) * Omega * comp } {}

    using InitFields<D>::bx1;

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      real_t factor;
      if(time<10){
	factor=time/10;
      }else{
	factor=1;
      }
	
      return -factor*(Omega - OmegaLT) * bx1(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t time, Omega;
    const real_t OmegaLT, comp; 
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

    const real_t  Bsurf, Rstar, Omega;
    InitFields<D> init_flds;
    const real_t  RLC;//, dt;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , Bsurf { p.template get<real_t>("setup.Bsurf", ONE) }
      , Rstar { m.mesh().extent(in::x1).first }
	//, dt { params.template get<real_t>("algorithms.timestep.dt") }
      , Omega { static_cast<real_t>(constant::TWO_PI) /
                p.template get<real_t>("setup.period", ONE) }
      , RLC {1/Omega}
      , init_flds { Bsurf, Rstar } {}

    inline PGen() {}

    void CustomFieldEvolution(std::size_t step, long double time, Domain<S, M>& domain, bool updateE, bool updateB) {
      if(updateB){
	const auto comp {params.template get<real_t>("setup.Compactness") };
	const auto _omega {static_cast<real_t>(constant::TWO_PI) / params.template get<real_t>("setup.period", ONE)};
	const auto _rstar {Rstar};
	const auto _bsurf {Bsurf};
	const auto coeff { HALF* params.template get<real_t>
	    ("algorithms.timestep.correction") * params.template get<real_t>("algorithms.timestep.dt") };
	auto& EB = domain.fields.em;
	auto metric = domain.mesh.metric;
	real_t factor;
	if(time<10){
	  factor=0;
	}else if(time>20){
	  factor=1;
	}else{
	  factor=0.1*(time-10);
	}	
        Kokkos::parallel_for(
	  "addShift",
          domain.mesh.rangeActiveCells(),
          Lambda(index_t i1, index_t i2) {
	    real_t corrEx2iP1j, corrEx2ij, corrEx1ijP1, corrEx1ij;
	    const auto i1_ = COORD(i1);
	    const auto i2_ = COORD(i2);
	    { // Etheta(i+1, j)
	      real_t shift; 
	      const coord_t<Dim::_2D> xc2d{i1_+ONE, i2_+HALF};               
	      coord_t<Dim::_2D> xPh { ZERO };
	      metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
	      shift = factor*static_cast<real_t>(0.4) * _omega * comp *
		CUBE(_rstar / xPh[0]) * xPh[0] * math::sin(xPh[1]);

	      auto Bphys = _bsurf * SQR(_rstar/ xPh[0]);
	      auto Etheta = Bphys*shift;
	      corrEx2iP1j = metric.template transform<2, Idx::T, Idx::U>( { i1_+ONE, i2_ + HALF }, Etheta);
	    }
	    { // Etheta(i, j)
	      real_t shift; 
	      const coord_t<Dim::_2D> xc2d{i1_, i2_+HALF};               
	      coord_t<Dim::_2D> xPh { ZERO };
	      metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
	      shift = factor*static_cast<real_t>(0.4) * _omega * comp *
		CUBE(_rstar / xPh[0]) * xPh[0] * math::sin(xPh[1]);

	      auto Bphys = _bsurf * SQR(_rstar / xPh[0]);
	      auto Etheta = Bphys * shift;
	      corrEx2ij = metric.template transform<2, Idx::T, Idx::U>( { i1_, i2_ + HALF },Etheta);
	    }
	    { // Er(i, j)
	      real_t shift; 
	      const coord_t<Dim::_2D> xc2d{i1_+HALF, i2_};               
	      coord_t<Dim::_2D> xPh { ZERO };
	      metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
	      shift = static_cast<real_t>(0.4) * _omega * comp *
		CUBE(_rstar / xPh[0]) * xPh[0] * math::sin(xPh[1]);

	      auto Bphys = 0.0;
	      auto Er = Bphys * shift;
	      //corrEx1ij = metric.template transform<1, Idx::T, Idx::U>({ i1_ + HALF, i2_ }, Er);
	      corrEx1ij = 0.0;
	    }
	    { // Er(i, j+1)
	      real_t shift; 
	      const coord_t<Dim::_2D> xc2d{i1_+HALF,
					   i2_+ONE};   
	      coord_t<Dim::_2D> xPh { ZERO };
	      metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
	      shift = static_cast<real_t>(0.4) * _omega * comp *
		CUBE(_rstar / xPh[0]) * xPh[0] * math::sin(xPh[1]);

	      auto Bphys = 0.0;
	      auto Er = Bphys * shift;
	      //corrEx1ijP1 = metric.template transform<1, Idx::T, Idx::U>({ i1_ + HALF, i2_ + ONE}, Er);
	      corrEx1ijP1 = 0.0;
	    }
	    const real_t inv_sqrt_detH_pHpH { ONE / metric.sqrt_det_h(
					 { i1_ + HALF, i2_ + HALF }) };
	    const real_t h1_pHp1 { metric.template h_<1, 1>({ i1_ + HALF, i2_ + ONE }) };
	    const real_t h1_pH0 { metric.template h_<1, 1>({ i1_ + HALF, i2_ }) };
	    const real_t h2_p1pH { metric.template h_<2, 2>({ i1_ + ONE, i2_ + HALF }) };
	    const real_t h2_0pH { metric.template h_<2, 2>({ i1_, i2_ + HALF }) };

	    //EB(i1, i2, em::bx2) = 0.0;
	    EB(i1, i2, em::bx3) += coeff*inv_sqrt_detH_pHpH*( h2_p1pH * corrEx2iP1j - h2_0pH * corrEx2ij +
	    h1_pHp1 * corrEx1ijP1 - h1_pH0 * corrEx1ij);
	  });
      }
    }
    
    auto FieldDriver(real_t time) const -> DriveFields<D> {
      const auto comp {params.template get<real_t>("setup.Compactness") };
      return DriveFields<D> { time, Bsurf, Rstar, Omega, comp };
    }
  };

} // namespace user

#endif
