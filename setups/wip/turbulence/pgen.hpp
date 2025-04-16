#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;
  
  //external current definition
  template <Dimension D>
  struct ExternalCurrent{
 	ExternalCurrent(real_t dB, real_t om0, real_t g0, std::vector< std::vector<real_t> >& wavenumbers, random_number_pool_t& random_pool, real_t Lx, real_t Ly, real_t Lz)
	: dB { dB }
	, omega_0 { om0 }
	, gamma_0 { g0 }
	, wavenumbers { wavenumbers }
	, n_modes {wavenumbers.size()}
	, Lx { Lx }
	, Ly { Ly }
	, Lz { Lz }
	, k { "wavevector", 2, wavenumbers.size() }
	, u_imag { "u imaginary", wavenumbers.size() }
	, u_real { "u_real", wavenumbers.size() } 
	, a_real { "a_real", wavenumbers.size() }
	, a_imag { "a_imag", wavenumbers.size() }
	, a_real_inv { "a_real", wavenumbers.size() }
        , a_imag_inv { "a_imag", wavenumbers.size() }

	, A0 {"A0", wavenumbers.size()}
	{
		// initializing wavevectors
		auto k_host = Kokkos::create_mirror_view(k); 
		for (auto i = 0; i < n_modes; i++){
			for(size_t j = 0; j < 2; j++){
				k_host(j,i) = constant::TWO_PI * wavenumbers[i][j] / Lx;
			}
			printf("k(%d) = (%f, %f)\n", i,k_host(0,i), k_host(1,i));
		}


		// initializing initial complex amplitudes
		auto a_real_host = Kokkos::create_mirror_view(a_real);
	        auto a_imag_host = Kokkos::create_mirror_view(a_imag); 	
	        auto A0_host = Kokkos::create_mirror_view(A0); 	
		for (auto i = 0; i < n_modes; i++){
				auto k_perp = math::sqrt(k_host(0,i) * k_host(0,i) + k_host(1,i) * k_host(1,i));
				auto phase = constant::TWO_PI / 6.;
				A0_host(i) =  dB / math::sqrt((real_t) n_modes) / k_perp;
				a_real_host(i) = A0_host(i) * math::cos(phase);
				a_imag_host(i) = A0_host(i) * math::sin(phase);
				printf("A0(%d) = %f\n", i,A0_host(i));
				printf("a_real(%d) = %f\n", i,a_real_host(i));
				printf("a_imag(%d) = %f\n", i,a_imag_host(i));


		}

		Kokkos::deep_copy(a_real, a_real_host);
		Kokkos::deep_copy(a_imag, a_imag_host); 
		Kokkos::deep_copy(a_real_inv, a_real_host);
                Kokkos::deep_copy(a_imag_inv, a_imag_host);

		Kokkos::deep_copy(A0, A0_host);
		Kokkos::deep_copy(k, k_host);
	//	Kokkos::parallel_for( "Generate random  ", wavenumbers.size(), Lambda (int const i){
        //                    auto generator = random_pool.get_state();
        //                    a_real(i) = generator.frand(0.0, 1.0);
        //                    a_imag(i) = generator.frand(0.0, 1.0);
        //                    printf(" Initial amplitudes (%i) a_real= %f, a_imag= %f\n",i
        //                                    , a_real(i)
        //                                    , a_imag(i));
        //                    random_pool.free_state(generator);
        //                    });
	};


	Inline auto jx3(const coord_t<D>& x_Ph) const  -> real_t {
		if constexpr(D == Dim::_2D){
			real_t jx3_ant = ZERO;
			for (size_t i=0; i < n_modes; i++){
			         //k(i,0) + k(i,1);
				auto k_perp_sq = k(0,i) * k(0,i) + k(1,i) * k(1,i);
				auto k_dot_r = k(0,i) * x_Ph[0] + k(1,i) * x_Ph[1];
				jx3_ant -= TWO * k_perp_sq * (a_real(i) * math::cos(k_dot_r)
							    - a_imag(i) * math::sin(k_dot_r));
				jx3_ant -= TWO * k_perp_sq * (a_real_inv(i) * math::cos(k_dot_r)
                                                            - a_imag_inv(i) * math::sin(k_dot_r));

			}
			return jx3_ant;
		}
		if constexpr(D == Dim::_3D){
			real_t jx3_ant = ZERO;
			for (size_t i=0; i < n_modes; i++){
//				k_perp_sq = k[i][0] + k[i][1];
//				jx3_ant -= TWO * k_perp_sq * (ONE * math::cos(k[i][0] * x_Ph[0] + k[i][1] * x_Ph[1] + k[i][2] * x_Ph[2])
//					       		    + ONE * math::sin(k[i][0] * x_Ph[0] + k[i][1] * x_Ph[1] + k[i][2] * x_Ph[2])); 
			}
			return jx3_ant;
		}
//		printf("jz_ant = %f\n", jx3_ant);
	
	}
	Inline auto jx2(const coord_t<D>& x_Ph) const -> real_t {
		if constexpr(D == Dim::_2D){
			return ZERO;
		} 
		if constexpr(D == Dim::_3D){
			return ZERO;
		}
	}
	Inline auto jx1(const coord_t<D>& x_Ph) const -> real_t {
		if constexpr(D == Dim::_2D){
			return ZERO;
		}
		if constexpr(D == Dim::_3D){
			return ZERO;
		}
	}
		
	const real_t dB, omega_0, gamma_0, Lx, Ly, Lz;
	const size_t n_modes;
	array_t<real_t**> k;
	array_t<real_t*> A0;
	  public:
	array_t<real_t*> a_real;
	array_t<real_t*> a_imag;
	array_t<real_t*> u_imag;
	array_t<real_t*> a_real_inv;
	array_t<real_t*> a_imag_inv;
	array_t<real_t*> u_real; 
	const std::vector< std::vector<real_t> > wavenumbers;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions =
      traits::compatible_with<Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t temperature, dB, omega_0, gamma_0, Lx, Ly, Lz, dt;
    std::vector< std::vector<real_t> > wavenumbers;
    random_number_pool_t random_pool;

    
    ExternalCurrent<D> ExternalCurrent;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , temperature { p.template get<real_t>("setup.temperature") }
      , dB { p.template get<real_t>("setup.dB", 1.) } 
      , omega_0 { p.template get<real_t>("setup.omega_0", 0.5) }
      , gamma_0 { p.template get<real_t>("setup.gamma_0", 0.25) }
      , wavenumbers { { { 1, 0, 1 },
			{ 0, 1, 1 },
			{ 1, 1, 1 },
			{ -1, 1, 1 }} }
      , random_pool{ 0 }
      , Lx { global_domain.mesh().extent(in::x1).second - global_domain.mesh().extent(in::x1).first }
      , Ly { global_domain.mesh().extent(in::x2).second - global_domain.mesh().extent(in::x2).first }
      , Lz { global_domain.mesh().extent(in::x3).second - global_domain.mesh().extent(in::x3).first }
      , dt { params.template get<real_t>("algorithms.timestep.dt") }
      , ExternalCurrent { dB, omega_0, gamma_0, wavenumbers, random_pool, Lx, Ly, Lz }{}

    inline PGen() {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                        local_domain.random_pool,
                                                        temperature,
                                                        ZERO);
      const auto spatial_dist = arch::UniformInjector<S, M, arch::Maxwellian>(
        energy_dist,
        { 1, 2 });
      arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
        params,
        local_domain,
        spatial_dist,
        ONE);
  
    }

    void CustomPostStep(timestep_t, simtime_t time, Domain<S, M>& domain){

	    //generate white noise 
//	    Kokkos::parallel_for( "Generate random  ", wavenumbers.size(), ClassLambda (int const i){
//			    auto generator = random_pool.get_state();
//			    this->ExternalCurrent.u_real(i) = generator.frand(-0.5, 0.5);
//			    this->ExternalCurrent.u_imag(i) = generator.frand(-0.5, 0.5);
//			    printf(" %i) u_real= %f, u_imag= %f\n",i
//					    , this->ExternalCurrent.u_real(i)
//					    , this->ExternalCurrent.u_imag(i));
//			    random_pool.free_state(generator);
//			    });

	    // update amplitudes of antenna
	    Kokkos::parallel_for( " Antenna amplitudes  ", wavenumbers.size(), Lambda (int const i){
			    auto generator = random_pool.get_state();
			    const auto u_imag = generator.frand(-0.5, 0.5);
			    const auto u_real = generator.frand(-0.5, 0.5);
			    const auto u_real_inv = generator.frand(-0.5,0.5);
			    const auto u_imag_inv = generator.frand(-0.5,0.5);
			    printf(" %i) u_real= %f, u_imag= %f, u_real_inv = %f, u_imag_inv = %f\n",i
					    , u_real
					    , u_imag
					    , u_real_inv
					    , u_imag_inv);
			    random_pool.free_state(generator);
			    random_pool.free_state(generator);
			    auto a_real_prev = this->ExternalCurrent.a_real(i);
			    auto a_imag_prev = this->ExternalCurrent.a_imag(i);
			    auto a_real_inv_prev = this->ExternalCurrent.a_real_inv(i);
                            auto a_imag_inv_prev = this->ExternalCurrent.a_imag_inv(i);

                            printf(" %i) a_real= %f, a_imag= %f\n",i
                                            , this->ExternalCurrent.a_real(i)
                                            , this->ExternalCurrent.a_imag(i));
			    this->ExternalCurrent.a_real(i) = (a_real_prev * math::cos(this->ExternalCurrent.omega_0 * time) - a_imag_prev * math::sin(this->ExternalCurrent.omega_0 * time)) * math::exp(-this->ExternalCurrent.gamma_0 * time) + this->ExternalCurrent.A0(i) * math::sqrt(12.0 * this->ExternalCurrent.gamma_0 / this->dt ) * u_real * this->dt;

			    this->ExternalCurrent.a_imag(i) = (a_imag_prev * math::cos(this->ExternalCurrent.omega_0 * time) + a_real_prev * math::sin(this->ExternalCurrent.omega_0 * time)) * math::exp(-this->ExternalCurrent.gamma_0 * time) + this->ExternalCurrent.A0(i) * math::sqrt(12.0 * this->ExternalCurrent.gamma_0 / this->dt ) * u_imag * this->dt;

			    this->ExternalCurrent.a_real_inv(i) = (a_real_inv_prev * math::cos(-this->ExternalCurrent.omega_0 * time) - a_imag_inv_prev * math::sin(-this->ExternalCurrent.omega_0 * time)) * math::exp(-this->ExternalCurrent.gamma_0 * time) + this->ExternalCurrent.A0(i) * math::sqrt(12.0 * this->ExternalCurrent.gamma_0 / this->dt ) * u_real_inv * this->dt;

                            this->ExternalCurrent.a_imag_inv(i) = (a_imag_inv_prev * math::cos(-this->ExternalCurrent.omega_0 * time) + a_real_inv_prev * math::sin(-this->ExternalCurrent.omega_0 * time)) * math::exp(-this->ExternalCurrent.gamma_0 * time) + this->ExternalCurrent.A0(i) * math::sqrt(12.0 * this->ExternalCurrent.gamma_0 / this->dt ) * u_imag_inv * this->dt;
			    });

    }
  };

} // namespace user

#endif
