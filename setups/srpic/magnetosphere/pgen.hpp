#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

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
      return Omega * bx2(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      return -Omega * bx1(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
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

    const real_t  Bsurf, Rstar, Omega;
    const real_t  Curv; 
    const real_t  RLC;
    const real_t  dt, inv_n0;    
    InitFields<D> init_flds;

    // these two lines are related to number density computation
    bool          is_first_step;
    array_t<real_t**> cbuff;

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
      , init_flds { Bsurf, Rstar } {
	Kokkos::deep_copy(cbuff, ZERO); //seems like it is duplicated in CustomPartEvolution 
      }
    
    inline PGen() {}

    //inline void Bdipolar(int component) {}

    void CustomFieldEvolution(std::size_t step, long double time, Domain<S, M>& domain, bool updateE, bool updateB) {
      if(updateB){
	const auto comp {params.template get<real_t>("setups.Compactness") };
	const auto _omega {params.template get<real_t>("setup.period", ONE)};
	const auto _rstar {Rstar};
	const auto _bsurf {Bsurf};
	const auto coeff {- HALF * params.template get<real_t>
	    ("algorithms.timestep.correction") * dt };
	auto& EB = domain.fields.em;
	auto metric = domain.mesh.metric;
        Kokkos::parallel_for(
	  "addShift",
          domain.mesh.rangeActiveCells(),
          Lambda(index_t i1, index_t i2) {
	    real_t corrEx2iP1j, corrEx2ij, corrEx1ijP1, corrEx1ij;
	    const auto i1_ = COORD(i1);
	    const auto i2_ = COORD(i2);
	    { // Etheta(i+1, j)
	      real_t shift; 
	      const coord_t<Dim::_2D> xc2d{static_cast<real_t>(i1)+ONE,
					   static_cast<real_t>(i2)+HALF};               
	      coord_t<Dim::_2D> xPh { ZERO };
	      metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
	      shift = static_cast<real_t>(0.4) * _omega * comp *
		CUBE(_rstar / xPh[0]) * xPh[0] * math::sin(xPh[1]);

	      // here the correction is computed with the actual magnetic field 
	      //corrEx2iP1j = EB(i1+1, i2, em::bx1) * shift;

	      // here the correction is computed for purely dipolar field (if the coordinate transformation is correct)
	      auto Bphys = _bsurf * math::cos(xPh[1]) / CUBE(xPh[0] / _rstar);
	      auto Btransform = metric.template transform<1, Idx::T, Idx::U>(
			        { i1_+ONE, i2_ + HALF }, Bphys);
	      corrEx2iP1j = Btransform * shift;  
	    }
	    { // Etheta(i, j)
	      real_t shift; 
	      const coord_t<Dim::_2D> xc2d{static_cast<real_t>(i1),
					   static_cast<real_t>(i2)+HALF};               
	      coord_t<Dim::_2D> xPh { ZERO };
	      metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
	      shift = static_cast<real_t>(0.4) * _omega * comp *
		CUBE(_rstar / xPh[0]) * xPh[0] * math::sin(xPh[1]);

	      // here the correction is computed with the actual magnetic field 
	      //corrEx2ij = EB(i1, i2, em::bx1) * shift;

	      // here the correction is computed for purely dipolar field (if the coordinate transformation is correct)
	      auto Bphys = _bsurf * math::cos(xPh[1]) / CUBE(xPh[0] / _rstar);
	      auto Btransform = metric.template transform<2, Idx::T, Idx::U>(
			        { i1_, i2_ + HALF }, Bphys);
	      corrEx2ij = Btransform * shift;  
	    }
	    { // Er(i, j)
	      real_t shift; 
	      const coord_t<Dim::_2D> xc2d{static_cast<real_t>(i1)+HALF,
					   static_cast<real_t>(i2)};               
	      coord_t<Dim::_2D> xPh { ZERO };
	      metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
	      shift = static_cast<real_t>(0.4) * _omega * comp *
		CUBE(_rstar / xPh[0]) * xPh[0] * math::sin(xPh[1]);

	      // here the correction is computed with the actual magnetic field 
	      //corrEx1ij = EB(i1, i2, em::bx2) * shift;

	      // here the correction is computed for purely dipolar field (if the coordinate transformation is correct)
	      auto Bphys = _bsurf * HALF * math::sin(xPh[1]) / CUBE(xPh[0] / _rstar); 
	      auto Btransform = metric.template transform<2, Idx::T, Idx::U>(
			        { i1_ + HALF, i2_ }, Bphys);
	      corrEx1ij = Btransform * shift;  
	    }
	    { // Er(i, j+1)
	      real_t shift; 
	      const coord_t<Dim::_2D> xc2d{static_cast<real_t>(i1)+HALF,
					   static_cast<real_t>(i2)+ONE};   
	      coord_t<Dim::_2D> xPh { ZERO };
	      metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
	      shift = static_cast<real_t>(0.4) * _omega * comp *
		CUBE(_rstar / xPh[0]) * xPh[0] * math::sin(xPh[1]);

	      // here the correction is computed with the actual magnetic field 
	      //corrEx1ij = EB(i1, i2+1, em::bx2) * shift;

	      // here the correction is computed for purely dipolar field (if the coordinate transformation is correct)
	      auto Bphys = _bsurf * HALF * math::sin(xPh[1]) / CUBE(xPh[0] / _rstar); 
	      auto Btransform = metric.template transform<2, Idx::T, Idx::U>(
			        { i1_ + HALF, i2_+ONE}, Bphys);
	      corrEx1ijP1 = Btransform * shift;  
	    }
	    const real_t inv_sqrt_detH_pHpH { ONE / metric.sqrt_det_h(
					 { i1_ + HALF, i2_ + HALF }) };
	    const real_t h1_pHp1 { metric.template h_<1, 1>({ i1_ + HALF, i2_ + ONE }) };
	    const real_t h1_pH0 { metric.template h_<1, 1>({ i1_ + HALF, i2_ }) };
	    const real_t h2_p1pH { metric.template h_<2, 2>({ i1_ + ONE, i2_ + HALF }) };
	    const real_t h2_0pH { metric.template h_<2, 2>({ i1_, i2_ + HALF }) };

	    //minus included into coeff 
	    EB(i1, i2, em::bx3) += coeff*( h1_pHp1 * corrEx2iP1j - h1_pH0 * corrEx2ij +
					   h2_p1pH * corrEx1ijP1 - h2_0pH * corrEx1ij);
	  });
      }
    }

    void CustomPartEvolution(std::size_t step, long double time, Domain<S, M>& domain) {

      //this part is done to compute number density of photons/pairs to limit injection 
      if (is_first_step) {
        cbuff = array_t<real_t**>("cbuff",
                                  domain.mesh.n_all(in::x1),
                                  domain.mesh.n_all(in::x2));
	is_first_step = false;
      }
      //it must be zero at the beginning of each timestep 
      Kokkos::deep_copy(cbuff, ZERO);
      
      const auto gcaLarmor {params.template get<real_t>("algorithms.gca.larmor_max")};
      const auto gcaEoverB {params.template get<real_t>("algorithms.gca.e_ovr_b_max")};
      const auto larm {params.template get<real_t>("scales.larmor0")};
      const auto CurvGammaCool {params.template get<real_t>("setup.CurvGammaCool")};
      const auto CurvGammaEmit {params.template get<real_t>("setup.CurvGammaEmit")};
      const auto angThres {params.template get<real_t>("setups.angThres") };
      const auto rad_radius {params.template get<real_t>("setups.rad_radius") };
      const auto ph_thres {params.template get<real_t>("setups.ph_thres") };
      const auto PhotonDensity {params.template get<real_t>("setups.ph_dens") };
      const auto PairDensity {params.template get<real_t>("setups.pair_dens") }; 
      const auto gammaSec {params.template get<real_t>("setups.gammaSec") };
      const auto BoverBq {params.template get<real_t>("setups.BoverBq") };
      const auto coeffCool {-2*(dt/larm) * sqrt(CUBE(Rstar/RLC)) *Bsurf* SQR(Curv) * SQR(SQR(1/CurvGammaCool))};
      const auto coeffPhoton {2*(dt/larm) * sqrt(CUBE(Rstar/RLC)) * Bsurf * Curv * CUBE(CurvGammaEmit) * SQR(SQR(1/CurvGammaCool))};
      const auto coeffAbs {static_cast<real_t>(0.23)*(27/4)*(dt/larm)*SQR(CUBE(CurvGammaEmit)/SQR(CurvGammaCool)) * Bsurf * sqrt(CUBE(Rstar/RLC)) * BoverBq };
      const auto _Bsurf = Bsurf;

      
      auto metric = domain.mesh.metric;
      auto random_pool    = domain.random_pool;
      auto EB             = domain.fields.em; 

      //------------curvature cooling of pairs and curvature photons emission--------------

      array_t<std::size_t> phot_ind("phot_ind");
      
      auto& photons = domain.species[2];
      auto offset_ph = photons.npart();
      auto ux1_ph    = photons.ux1;
      auto ux2_ph    = photons.ux2;
      auto ux3_ph    = photons.ux3;
      auto i1_ph     = photons.i1;
      auto i2_ph     = photons.i2;
      auto dx1_ph    = photons.dx1;
      auto dx2_ph    = photons.dx2;
      auto phi_ph    = photons.phi;
      auto weight_ph = photons.weight;
      auto tag_ph    = photons.tag;
      auto pld0      = photons.pld[0]; //accumulated optical depth

      //this part is done to compute number density of photons to limit their injection
      auto cbuff_sc = Kokkos::Experimental::create_scatter_view(cbuff);
      auto inv_n0_ = this->inv_n0;
      Kokkos::parallel_for(
      "PhotonDensity", photons.rangeActiveParticles(), Lambda(index_t p) {
	if (tag_ph(p) == ParticleTag::dead) {
	  return;
	}
	auto cbuff_acc     = cbuff_sc.access();
	cbuff_acc(static_cast<int>(i1_ph(p)), static_cast<int>(i2_ph(p))) += weight_ph(p) * inv_n0_ /
	  metric.sqrt_det_h({ static_cast<real_t>(i1_ph(p)) + HALF,
			      static_cast<real_t>(i2_ph(p)) + HALF });
      });
      Kokkos::Experimental::contribute(cbuff, cbuff_sc);
      //ideally, end of photon density computation 
      
      for (std::size_t s { 0 }; s < 2; ++s) {
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

	const auto q_ovr_m = species.charge() / species.mass();
	
	Kokkos::parallel_for(
        "CurvatureCooling", species.rangeActiveParticles(), Lambda(index_t p) {
          if (tag(p) == ParticleTag::dead) {
            return;
          }

	  auto px      = ux1(p);
	  auto py      = ux2(p);
	  auto pz      = ux3(p);
	  real_t gamma   = math::sqrt(ONE + SQR(px) + SQR(py) + SQR(pz));
	  
	  // here we check if the particle is in gca 
	  vec_t<Dim::_3D> b_int_Cart { ZERO };
	  vec_t<Dim::_3D> e_int_Cart { ZERO };

	  const coord_t<Dim::_3D> xc3d {static_cast<real_t>(i1(p)) + dx1(p),
					static_cast<real_t>(i2(p)) + dx2(p), phi(p)}; 

     	  const auto   i { i1(p) + N_GHOSTS };
	  const real_t dx1_ { dx1(p) };

	  const auto   j { i2(p) + N_GHOSTS };
	  const real_t dx2_ { dx2(p) };   

	  vec_t<Dim::_3D> b_int { ZERO };
	  vec_t<Dim::_3D> e_int { ZERO };
	  
	  real_t      c000, c100, c010, c110, c00, c10;
	  
	  //Bx1
	  c000  = HALF * (EB(i, j, em::bx1) + EB(i, j - 1, em::bx1));
	  c100  = HALF * (EB(i + 1, j, em::bx1) + EB(i + 1, j - 1, em::bx1));
	  c010  = HALF * (EB(i, j, em::bx1) + EB(i, j + 1, em::bx1));
	  c110  = HALF * (EB(i + 1, j, em::bx1) + EB(i + 1, j + 1, em::bx1));
	  
	  c00   = c000 * (ONE - dx1_) + c100 * dx1_;
	  c10   = c010 * (ONE - dx1_) + c110 * dx1_;
	  b_int[0] = c00 * (ONE - dx2_) + c10 * dx2_;
	  
	  // Bx2
	  c000  = HALF * (EB(i - 1, j, em::bx2) + EB(i, j, em::bx2));
	  c100  = HALF * (EB(i, j, em::bx2) + EB(i + 1, j, em::bx2));
	  c010  = HALF * (EB(i - 1, j + 1, em::bx2) + EB(i, j + 1, em::bx2));
	  c110  = HALF * (EB(i, j + 1, em::bx2) + EB(i + 1, j + 1, em::bx2));
	  c00   = c000 * (ONE - dx1_) + c100 * dx1_;
	  c10   = c010 * (ONE - dx1_) + c110 * dx1_;
	  b_int[1] = c00 * (ONE - dx2_) + c10 * dx2_;
	  // Bx3
	  c000  = INV_4 * (EB(i - 1, j - 1, em::bx3) + EB(i - 1, j, em::bx3) +
			   EB(i, j - 1, em::bx3) + EB(i, j, em::bx3));
	  c100  = INV_4 * (EB(i, j - 1, em::bx3) + EB(i, j, em::bx3) +
			   EB(i + 1, j - 1, em::bx3) + EB(i + 1, j, em::bx3));
	  c010  = INV_4 * (EB(i - 1, j, em::bx3) + EB(i - 1, j + 1, em::bx3) +
			   EB(i, j, em::bx3) + EB(i, j + 1, em::bx3));
	  c110  = INV_4 * (EB(i, j, em::bx3) + EB(i, j + 1, em::bx3) +
			   EB(i + 1, j, em::bx3) + EB(i + 1, j + 1, em::bx3));
	  c00   = c000 * (ONE - dx1_) + c100 * dx1_;
	  c10   = c010 * (ONE - dx1_) + c110 * dx1_;
	  b_int[2] = c00 * (ONE - dx2_) + c10 * dx2_;
	  
	  metric.template transform_xyz<Idx::U, Idx::XYZ>(xc3d, b_int, b_int_Cart);
	  
	  // Ex1
	  c000  = HALF * (EB(i, j, em::ex1) + EB(i - 1, j, em::ex1));
	  c100  = HALF * (EB(i, j, em::ex1) + EB(i + 1, j, em::ex1));
	  c010  = HALF * (EB(i, j + 1, em::ex1) + EB(i - 1, j + 1, em::ex1));
	  c110  = HALF * (EB(i, j + 1, em::ex1) + EB(i + 1, j + 1, em::ex1));
	  c00   = c000 * (ONE - dx1_) + c100 * dx1_;
	  c10   = c010 * (ONE - dx1_) + c110 * dx1_;
	  e_int[0] = c00 * (ONE - dx2_) + c10 * dx2_;
	  // Ex2
	  c000  = HALF * (EB(i, j, em::ex2) + EB(i, j - 1, em::ex2));
	  c100  = HALF * (EB(i + 1, j, em::ex2) + EB(i + 1, j - 1, em::ex2));
	  c010  = HALF * (EB(i, j, em::ex2) + EB(i, j + 1, em::ex2));
	  c110  = HALF * (EB(i + 1, j, em::ex2) + EB(i + 1, j + 1, em::ex2));
	  c00   = c000 * (ONE - dx1_) + c100 * dx1_;
	  c10   = c010 * (ONE - dx1_) + c110 * dx1_;
	  e_int[1] = c00 * (ONE - dx2_) + c10 * dx2_;
	  // Ex3
	  c000  = EB(i, j, em::ex3);
	  c100  = EB(i + 1, j, em::ex3);
	  c010  = EB(i, j + 1, em::ex3);
	  c110  = EB(i + 1, j + 1, em::ex3);
	  c00   = c000 * (ONE - dx1_) + c100 * dx1_;
	  c10   = c010 * (ONE - dx1_) + c110 * dx1_;
	  e_int[2] = c00 * (ONE - dx2_) + c10 * dx2_;
	  
	  metric.template transform_xyz<Idx::U, Idx::XYZ>(xc3d, e_int, e_int_Cart);
      	  
	  const auto E2 { NORM_SQR(e_int_Cart[0], e_int_Cart[1], e_int_Cart[2]) };
	  const auto B2 { NORM_SQR(b_int_Cart[0], b_int_Cart[1], b_int_Cart[2]) };
	  const auto rL { gamma * larm/ (math::abs(q_ovr_m) * math::sqrt(B2)) };

	  //the particle is in gca regime and subject to curvature cooling	  
	  if (B2 > ZERO && rL < gcaLarmor && (E2 / B2) < gcaEoverB ) {
	    auto gamma3 = CUBE(gamma);
	    auto gamma4 = SQR(SQR(gamma));
	    // Get particle coordinates for later processing
	    // Does it work with multiple blocks?
	    const coord_t<Dim::_2D> xc2d{static_cast<real_t>(i1(p)) + dx1(p),
					 static_cast<real_t>(i2(p)) + dx2(p)};               
	    coord_t<Dim::_2D> xPh { ZERO };
	    metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);

	    auto beta_x = px/gamma;
	    auto beta_y = py/gamma;
	    auto beta_z = pz/gamma;
	    
	    ux1(p) += coeffCool*gamma4*beta_x;
	    ux2(p) += coeffCool*gamma4*beta_y;
	    ux3(p) += coeffCool*gamma4*beta_z;

	    auto ph_energy { gamma3/CUBE(CurvGammaEmit)};
	    auto ph_prob { coeffPhoton * gamma};

	    auto  rand_gen = random_pool.get_state();
	    if ((Random<real_t>(rand_gen) < ph_prob) &&
		(ph_energy > ph_thres) && (xPh[0] < rad_radius) &&
		(xPh[1] > angThres) && (xPh[1] < (constant::PI - angThres))){
		//cbuff(i1(p),i2(p))<PhotonDensity) { // fix this somehow 
	       auto phot_p = Kokkos::atomic_fetch_add(&phot_ind(), 1);
	       i1_ph(phot_p + offset_ph) = i1(p);
	       dx1_ph(phot_p + offset_ph) = dx1(p);
	       i2_ph(phot_p + offset_ph) = i2(p);
	       dx2_ph(phot_p + offset_ph) = dx2(p);
	       phi_ph(phot_p + offset_ph) = phi(p);
	       ux1_ph(phot_p + offset_ph) = ph_energy * beta_x;
	       ux2_ph(phot_p + offset_ph) = ph_energy * beta_y;
	       ux3_ph(phot_p + offset_ph) = ph_energy * beta_z;
	       weight_ph(phot_p + offset_ph) = weight(p);
	       tag_ph(phot_p + offset_ph) = ParticleTag::alive;
	       pld0(phot_p + offset_ph) = 0.0;
	    }	    
	  }
	});
	auto phot_ind_h = Kokkos::create_mirror(phot_ind);
	Kokkos::deep_copy(phot_ind_h, phot_ind);
	photons.set_npart(offset_ph + phot_ind_h());
      }

      //-----------photon propagation and secondary pair emission----------

      // similarly to photon number density, this one should compute pairs number density
      Kokkos::deep_copy(cbuff, ZERO);
      cbuff_sc = Kokkos::Experimental::create_scatter_view(cbuff);
      for (std::size_t s { 0 }; s < 2; ++s) {
        auto& species = domain.species[s];
        auto i1     = species.i1;
        auto i2     = species.i2;
        auto weight = species.weight;
        auto tag    = species.tag;
	Kokkos::parallel_for(
	  "PhotonDensity", species.rangeActiveParticles(), Lambda(index_t p) {
	    if (tag(p) == ParticleTag::dead) {
	      return;
	    }
	    auto cbuff_acc     = cbuff_sc.access();
	    cbuff_acc(static_cast<int>(i1(p)), static_cast<int>(i2(p))) += weight(p) * inv_n0_ /
	      metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
				  static_cast<real_t>(i2(p)) + HALF });
	  });
      }
      Kokkos::Experimental::contribute(cbuff, cbuff_sc);
      // this should compute total number density of electrons and positrons

      array_t<std::size_t> elec_ind("elec_ind");
      array_t<std::size_t> pos_ind("pos_ind");

      auto& electrons  = domain.species[0];
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

      auto& positrons = domain.species[1];
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

      
      Kokkos::parallel_for(
       "Photons", photons.rangeActiveParticles(), Lambda(index_t p) {
	 if (tag_ph(p) == ParticleTag::dead) {
	   return;
	 }
	 
	 auto px      = ux1_ph(p);
	 auto py      = ux2_ph(p);
	 auto pz      = ux3_ph(p);
	 real_t ePh   = math::sqrt(SQR(px) + SQR(py) + SQR(pz));
	 
	 // here we check if the particle is in gca 
	 vec_t<Dim::_3D> b_int_Cart { ZERO };
	 
	 const coord_t<Dim::_3D> xc3d {static_cast<real_t>(i1_ph(p)) + dx1_ph(p),
				       static_cast<real_t>(i2_ph(p)) + dx2_ph(p), phi_ph(p)}; 
	 
	 const auto   i { i1_ph(p) + N_GHOSTS };
	 const real_t dx1_ { dx1_ph(p) };
	 
	 const auto   j { i2_ph(p) + N_GHOSTS };
	 const real_t dx2_ { dx2_ph(p) };   
	 
	 vec_t<Dim::_3D> b_int { ZERO };
	 
	 real_t      c000, c100, c010, c110, c00, c10;
	 
	 //Bx1
	 c000  = HALF * (EB(i, j, em::bx1) + EB(i, j - 1, em::bx1));
	 c100  = HALF * (EB(i + 1, j, em::bx1) + EB(i + 1, j - 1, em::bx1));
	 c010  = HALF * (EB(i, j, em::bx1) + EB(i, j + 1, em::bx1));
	 c110  = HALF * (EB(i + 1, j, em::bx1) + EB(i + 1, j + 1, em::bx1));
	 
	 c00   = c000 * (ONE - dx1_) + c100 * dx1_;
	 c10   = c010 * (ONE - dx1_) + c110 * dx1_;
	 b_int[0] = c00 * (ONE - dx2_) + c10 * dx2_;
	 
	 // Bx2
	 c000  = HALF * (EB(i - 1, j, em::bx2) + EB(i, j, em::bx2));
	 c100  = HALF * (EB(i, j, em::bx2) + EB(i + 1, j, em::bx2));
	 c010  = HALF * (EB(i - 1, j + 1, em::bx2) + EB(i, j + 1, em::bx2));
	 c110  = HALF * (EB(i, j + 1, em::bx2) + EB(i + 1, j + 1, em::bx2));
	 c00   = c000 * (ONE - dx1_) + c100 * dx1_;
	 c10   = c010 * (ONE - dx1_) + c110 * dx1_;
	 b_int[1] = c00 * (ONE - dx2_) + c10 * dx2_;
	 // Bx3
	 c000  = INV_4 * (EB(i - 1, j - 1, em::bx3) + EB(i - 1, j, em::bx3) +
			  EB(i, j - 1, em::bx3) + EB(i, j, em::bx3));
	 c100  = INV_4 * (EB(i, j - 1, em::bx3) + EB(i, j, em::bx3) +
			  EB(i + 1, j - 1, em::bx3) + EB(i + 1, j, em::bx3));
	 c010  = INV_4 * (EB(i - 1, j, em::bx3) + EB(i - 1, j + 1, em::bx3) +
			  EB(i, j, em::bx3) + EB(i, j + 1, em::bx3));
	 c110  = INV_4 * (EB(i, j, em::bx3) + EB(i, j + 1, em::bx3) +
			  EB(i + 1, j, em::bx3) + EB(i + 1, j + 1, em::bx3));
	 c00   = c000 * (ONE - dx1_) + c100 * dx1_;
	 c10   = c010 * (ONE - dx1_) + c110 * dx1_;
	 b_int[2] = c00 * (ONE - dx2_) + c10 * dx2_;
	 
	 metric.template transform_xyz<Idx::U, Idx::XYZ>(xc3d, b_int, b_int_Cart); 

	 const auto Babs { NORM(b_int_Cart[0], b_int_Cart[1], b_int_Cart[2]) };

	 const auto binitX { b_int_Cart[0]/Babs };
	 const auto binitY { b_int_Cart[1]/Babs };
	 const auto binitZ { b_int_Cart[2]/Babs };
	 
	 const auto cosAngle { DOT(binitX, binitY, binitZ,
				   px,py,pz)/ePh};
	 
	 const auto sinAngle { math::sqrt(ONE - SQR(cosAngle)) };

	 const auto increment {
	   coeffAbs * (Babs / _Bsurf) * sinAngle *
	     math::exp(-8.0 / (3.0 * BoverBq * (Babs / _Bsurf) * sinAngle * ePh))
	 };

	 pld0(p) += increment;

	 //check physical position of the photon and erase if too far
	 const coord_t<Dim::_2D> xc2d{static_cast<real_t>(i1_ph(p)) + dx1_ph(p),
				      static_cast<real_t>(i2_ph(p)) + dx2_ph(p)};
	 coord_t<Dim::_2D> xPh { ZERO };
	 metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);

	 if (xPh[0] > rad_radius){
	   tag_ph(p) = ParticleTag::dead;
	 }
	 
	 //emit pairs

	 if (pld0(p)>=1.0){
	   tag_ph(p) = ParticleTag::dead;
	   //if (cbuff(i1_ph(p),i2_ph(p)) <= PairDensity) {
	   {
	     
	     auto elec_p = Kokkos::atomic_fetch_add(&elec_ind(), 1);
	     auto pos_p = Kokkos::atomic_fetch_add(&pos_ind(), 1);

	     i1_elec(elec_p + offset_elec) = i1_ph(p);
	     dx1_elec(elec_p + offset_elec) = dx1_ph(p);
	     i2_elec(elec_p + offset_elec) = i2_ph(p);
	     dx2_elec(elec_p + offset_elec) = dx2_ph(p);
	     phi_elec(elec_p + offset_elec) = phi_ph(p);
	     ux1_elec(elec_p + offset_elec) = SIGN(cosAngle) * gammaSec * binitX;
	     ux2_elec(elec_p + offset_elec) = SIGN(cosAngle) * gammaSec * binitY;
	     ux3_elec(elec_p + offset_elec) = SIGN(cosAngle) * gammaSec * binitZ;  
	     weight_elec(elec_p + offset_elec) = weight_ph(p);
	     tag_elec(elec_p + offset_elec) = ParticleTag::alive;
	     
	     i1_pos(pos_p + offset_pos) = i1_ph(p);
	     dx1_pos(pos_p + offset_pos) = dx1_ph(p);
	     i2_pos(pos_p + offset_pos) = i2_ph(p);
	     dx2_pos(pos_p + offset_pos) = dx2_ph(p);
	     phi_pos(pos_p + offset_pos) = phi_ph(p);
	     ux1_pos(pos_p + offset_pos) = SIGN(cosAngle) * gammaSec * binitX;
	     ux2_pos(pos_p + offset_pos) = SIGN(cosAngle) * gammaSec * binitY;
	     ux3_pos(pos_p + offset_pos) = SIGN(cosAngle) * gammaSec * binitZ;
	     weight_pos(pos_p + offset_pos) = weight_ph(p);
	     tag_pos(pos_p + offset_pos) = ParticleTag::alive;
	   }
	 }
       });
      auto elec_ind_h = Kokkos::create_mirror(elec_ind);
      Kokkos::deep_copy(elec_ind_h, elec_ind);
      domain.species[0].set_npart(offset_elec + elec_ind_h());

      auto pos_ind_h = Kokkos::create_mirror(pos_ind);
      Kokkos::deep_copy(pos_ind_h, pos_ind);
      domain.species[1].set_npart(offset_pos + pos_ind_h());

    }

    auto FieldDriver(real_t time) const -> DriveFields<D> {
      return DriveFields<D> { time, Bsurf, Rstar, Omega };
    }
  };

} // namespace user

#endif
