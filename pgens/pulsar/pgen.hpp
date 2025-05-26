#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"
#include "kernels/particle_moments.hpp"

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
  struct BoundaryFields {
    BoundaryFields(real_t bsurf, real_t rstar) : Bsurf { bsurf }, Rstar { rstar } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return Bsurf * math::cos(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

  private:
    const real_t Bsurf, Rstar;
  };
  
  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(real_t time, real_t bsurf, real_t rstar, real_t omega,  real_t comp, real_t spinup)
      : InitFields<D> { bsurf, rstar }
      , time { time }
      , Omega { omega }
      , comp {comp}
      , spinup {spinup}
      , OmegaLT { static_cast<real_t>(0.4) * Omega * comp }{}

    using InitFields<D>::bx1;
    using InitFields<D>::bx2;

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      real_t factor;
      if (time>spinup){
	factor = 1.0;
      }else{
	factor = time/spinup;
      }
      return (Omega - OmegaLT) * factor * bx2(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      //real_t factor = 1.0;
      real_t factor; 
      if (time>spinup){
	factor = 1.0;
      }else{
	factor = time/spinup;
      }
      return -(Omega - OmegaLT) * factor * bx1(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t time, Omega, OmegaLT, comp, spinup;
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
    const real_t  Curv; 
    const real_t  RLC;
    const real_t  inv_n0, dt;    
    InitFields<D> init_flds;

    // these two lines are related to number density computation
    bool          is_first_step;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , Bsurf { p.template get<real_t>("setup.Bsurf", ONE) }
      , Rstar { m.mesh().extent(in::x1).first }
      , Omega { static_cast<real_t>(constant::TWO_PI) /
                p.template get<real_t>("setup.period", ONE) }
      , RLC {1/Omega}
      , inv_n0 {ONE / p.template get<real_t>("scales.n0")}
      , dt { params.template get<real_t>("algorithms.timestep.dt") }
      , Curv {p.template get<real_t>("setup.Curvature")}
      , is_first_step { true }
      , init_flds { Bsurf, Rstar } {}
    
    inline PGen() {}

    void CustomPartEvolution(std::size_t step, long double time, Domain<S, M>& domain) {
      
      const auto gcaLarmor {params.template get<real_t>("algorithms.gca.larmor_max")};
      const auto gcaEoverB_ {params.template get<real_t>("algorithms.gca.e_ovr_b_max")};
      const auto gcaEoverB {SQR(gcaEoverB_)};
      const auto larm {params.template get<real_t>("scales.larmor0")};
      const auto gamma_thres1 {params.template get<real_t>("setup.GammaThres1")};
      const auto gamma_thres2 {params.template get<real_t>("setup.GammaThres2")};
      const auto stepsEmit {1.0/params.template get<real_t>("setup.StepsEmit")};
      const auto angThres {params.template get<real_t>("setup.angThres") };
      const auto rad_radius {params.template get<real_t>("setup.rad_radius") };
      const auto PairDensity1 {params.template get<real_t>("setup.pair_dens1") };
      const auto PairDensity2 {params.template get<real_t>("setup.pair_dens2") };
      const auto gammaSec {params.template get<real_t>("setup.gammaSec") };
      const auto meanPath {params.template get<real_t>("setup.meanPath") };
      const auto FlippingFraction {params.template get<real_t>("setup.flipFrac")};
      const auto dt_ {dt};
      //const auto _Bsurf { Bsurf };
      //const auto Rstar_ { Rstar }; 

      real_t gamma_thres =  gamma_thres1;

      real_t PairDensity;
      if(time<5.0){
	PairDensity = PairDensity1;
      }else{
	PairDensity = PairDensity2;
      }
      
      auto metric = domain.mesh.metric;
      auto random_pool    = domain.random_pool;
      auto EB             = domain.fields.em; 

      //------------curvature cooling of pairs and curvature photons emission--------------
      	

      Kokkos::deep_copy(domain.fields.bckp, ZERO);
      auto scatter_bckp = Kokkos::Experimental::create_scatter_view(
							      domain.fields.bckp);
      const auto inv_n0 = ONE / params.template get<real_t>("scales.n0");
      const auto ni2         = domain.mesh.n_active(in::x2);
      const auto use_weights = M::CoordType != Coord::Cart;
        
      for( int i=0; i<4; i++){
	auto& part = domain.species[i];
	if (part.mass() == 0){
	  continue;
	}
	if (part.npart()>0){
	  Kokkos::parallel_for(
            "ComputeMoments",
	    part.rangeActiveParticles(),
	    kernel::ParticleMoments_kernel<SimEngine::SRPIC, M, FldsID::N, 6>(
              {}, scatter_bckp, 0,
	      part.i1, part.i2, part.i3,
	      part.dx1, part.dx2, part.dx3,
	      part.ux1, part.ux2, part.ux3,
	      part.phi, part.weight, part.tag,
	      part.mass(), part.charge(),
	      use_weights,
	      domain.mesh.metric, domain.mesh.flds_bc(),
	      ni2, inv_n0, 0));
	  part.set_unsorted();
	}
      }
      Kokkos::Experimental::contribute(domain.fields.bckp, scatter_bckp);
      auto BCKP = domain.fields.bckp;      

      for (std::size_t s { 0 }; s < 4; ++s) {

	if(s==1){
	  continue;
	}
	
	array_t<std::size_t> elec_ind("elec_ind");
	array_t<std::size_t> pos_ind("pos_ind");

	std::size_t elec_spec;
	elec_spec = 2;
	auto& electrons  = domain.species[elec_spec];
	auto offset_elec = electrons.npart();
	auto ux1_elec    = electrons.ux1;
	auto ux2_elec    = electrons.ux2;
	auto ux3_elec    = electrons.ux3;
	auto i1_elec     = electrons.i1;
	auto i2_elec     = electrons.i2;
	auto dx1_elec    = electrons.dx1;
	auto dx2_elec    = electrons.dx2;
	auto phi_elec    = electrons.phi;
	auto weight_elec = electrons.weight;
	auto tag_elec    = electrons.tag;
	auto pld_elec    = electrons.pld;

	std::size_t pos_spec;
	pos_spec = 3;
	auto& positrons = domain.species[pos_spec];
	auto offset_pos = positrons.npart();
	auto ux1_pos    = positrons.ux1;
	auto ux2_pos    = positrons.ux2;
	auto ux3_pos    = positrons.ux3;
	auto i1_pos     = positrons.i1;
	auto i2_pos     = positrons.i2;
	auto dx1_pos    = positrons.dx1;
	auto dx2_pos    = positrons.dx2;
	auto phi_pos    = positrons.phi;
	auto weight_pos = positrons.weight;
	auto tag_pos    = positrons.tag;
	auto pld_pos    = positrons.pld;

	auto& species = domain.species[s];
	auto ux1    = species.ux1;
	auto ux2    = species.ux2;
	auto ux3    = species.ux3;
	auto i1     = species.i1;
	auto i2     = species.i2;
	auto dx1    = species.dx1;
	auto dx2    = species.dx2;
	auto phi    = species.phi;
	auto weight = species.weight;
	auto tag    = species.tag;
	auto pld    = species.pld;

	const auto q_ovr_m = species.charge() / species.mass();
	
	Kokkos::parallel_for(
          "CurvatureCooling", species.rangeActiveParticles(), Lambda(index_t p) {
	    if (tag(p) == ParticleTag::dead) {
	      return;
	    }
              
	    auto px      = ux1(p);
	    auto py      = ux2(p);
	    auto pz      = ux3(p);
	    real_t gamma = math::sqrt(ONE + SQR(px) + SQR(py) + SQR(pz));
	    auto pld0    = pld(p,0);  
          
	    const auto   i { i1(p) + N_GHOSTS };
	    const real_t dx1_ { dx1(p) };

	    const auto   j { i2(p) + N_GHOSTS };
	    const real_t dx2_ { dx2(p) };   
	    
	    const coord_t<Dim::_2D> xc2d{static_cast<real_t>(i1(p)) + dx1(p),
					 static_cast<real_t>(i2(p)) + dx2(p)};               
	    coord_t<Dim::_2D> xPh { ZERO };
	    metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
	      
	    auto beta_x = px/gamma;
	    auto beta_y = py/gamma;
	    auto beta_z = pz/gamma;
	    
	    auto  rand_gen = random_pool.get_state();

	    //real_t densityFactor = 1.0;
	    real_t rhoMax = PairDensity/pow(xPh[0],2) * (math::tanh((rad_radius-xPh[0])/0.25)+1.0) / 2.0;
	    real_t densityFactor = math::exp(-math::pow(BCKP(i, j, 0)/rhoMax,2));
	    
	    if ((gamma > gamma_thres) && (Random<real_t>(rand_gen) < stepsEmit*densityFactor) && (pld0 > meanPath)){
	      /*ux1(p) -= 2*gammaSec*beta_x;
	      ux2(p) -= 2*gammaSec*beta_y;
	      ux3(p) -= 2*gammaSec*beta_z;*/

	      if ((xPh[1] > angThres) && (xPh[1] < (constant::PI - angThres))){
	      
		auto elec_p = Kokkos::atomic_fetch_add(&elec_ind(), 1);
		auto pos_p = Kokkos::atomic_fetch_add(&pos_ind(), 1);

		const vec_t<Dim::_3D> betaCart {beta_x, beta_y, beta_z};
		vec_t<Dim::_3D> betaTetr;
		metric.template transform<Idx::U, Idx::T>(xc2d, betaCart, betaTetr);
		
		auto mom = math::sqrt(gammaSec*gammaSec-1);
		i1_elec(elec_p + offset_elec) = i1(p);
		dx1_elec(elec_p + offset_elec) = dx1(p);
		i2_elec(elec_p + offset_elec) = i2(p);
		dx2_elec(elec_p + offset_elec) = dx2(p);
		phi_elec(elec_p + offset_elec) = phi(p);
		auto rand_gen1 = random_pool.get_state();
		auto direction_factor = Random<real_t>(rand_gen1);
		random_pool.free_state(rand_gen1);
		if ((betaTetr[0] < 0) && (direction_factor < FlippingFraction)){ 
		  ux1_elec(elec_p + offset_elec) = -1*mom * beta_x;
		  ux2_elec(elec_p + offset_elec) = -1*mom * beta_y;
		  ux3_elec(elec_p + offset_elec) = -1*mom * beta_z;
		}else{
		  ux1_elec(elec_p + offset_elec) = mom * beta_x;
		  ux2_elec(elec_p + offset_elec) = mom * beta_y;
		  ux3_elec(elec_p + offset_elec) = mom * beta_z;
		}
		weight_elec(elec_p + offset_elec) = weight(p);
		tag_elec(elec_p + offset_elec) = ParticleTag::alive;
		pld_elec(elec_p + offset_elec, 0) = 0.0;
		
		i1_pos(pos_p + offset_pos) = i1(p);
		dx1_pos(pos_p + offset_pos) = dx1(p);
		i2_pos(pos_p + offset_pos) = i2(p);
		dx2_pos(pos_p + offset_pos) = dx2(p);
		phi_pos(pos_p + offset_pos) = phi(p);
		if ((betaTetr[0] < 0) && (direction_factor < FlippingFraction)){
 		  ux1_pos(pos_p + offset_pos) = -1*mom * beta_x;
		  ux2_pos(pos_p + offset_pos) = -1*mom * beta_y;
		  ux3_pos(pos_p + offset_pos) = -1*mom * beta_z;
		}else{
 		  ux1_pos(pos_p + offset_pos) = mom * beta_x;
		  ux2_pos(pos_p + offset_pos) = mom * beta_y;
		  ux3_pos(pos_p + offset_pos) = mom * beta_z;		  
		}

		weight_pos(pos_p + offset_pos) = weight(p);
		tag_pos(pos_p + offset_pos) = ParticleTag::alive;
		pld_pos(pos_p + offset_pos, 0) = 0.0;
	      }
	    }
	    random_pool.free_state(rand_gen);
	    if (gamma>gamma_thres){
	      pld(p, 0) += dt_;
	    }else{
	      pld(p, 0) = 0.0;
	    }
	  });
        auto elec_ind_h = Kokkos::create_mirror(elec_ind);
        Kokkos::deep_copy(elec_ind_h, elec_ind);
        electrons.set_npart(offset_elec + elec_ind_h());
        
        auto pos_ind_h = Kokkos::create_mirror(pos_ind);
        Kokkos::deep_copy(pos_ind_h, pos_ind);
        positrons.set_npart(offset_pos + pos_ind_h());
      }
    }
      
    auto AtmFields(real_t time) const -> DriveFields<D> {
      const auto comp {params.template get<real_t>("setup.Compactness") };
      const auto spinup {params.template get<real_t>("setup.spinup_time") };
      return DriveFields<D> { time, Bsurf, Rstar, Omega, comp, spinup };
    }

    //auto MatchFields(real_t) const -> InitFields<D> {
    auto MatchFields(real_t) const -> BoundaryFields<D> {
      return BoundaryFields<D> { Bsurf, Rstar };  
      //return InitFields<D> { Bsurf, Rstar };
    }    
  };

} // namespace user

#endif
