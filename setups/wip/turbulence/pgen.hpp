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

#if defined(MPI_ENABLED)
#include <stdlib.h>
#endif //MPI_ENABLED

namespace user {
  using namespace ntt;

  // initializing guide field and curl(B) = J_ext at the initial time step
  template <Dimension D>
  struct InitFields {
    InitFields(array_t<real_t**>& k, array_t<real_t*>& a_real,  array_t<real_t*>& a_imag,  array_t<real_t*>& a_real_inv, array_t<real_t*>& a_imag_inv )
    : k { k }
    , a_real { a_real }
    , a_imag { a_imag }
    , a_real_inv { a_real_inv }
    , a_imag_inv { a_imag_inv }
    , n_modes {a_real.size() }   {};

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      auto bx1_0 = ZERO;
      for (auto i = 0; i < n_modes; i++){
	      auto k_dot_r = k(0,i) * x_Ph[0] + k(1,i) * x_Ph[1];
	      bx1_0 -= TWO * k(1,i) * (a_real(i) * math::sin(k_dot_r) + a_imag(i) * math::cos( k_dot_r ));
	      bx1_0 -= TWO * k(1,i) * (a_real_inv(i) * math::sin(k_dot_r) + a_imag_inv(i) * math::cos( k_dot_r ));
      
      }
      return bx1_0;
    }
    
    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      auto bx2_0 = ZERO;
      for (auto i = 0; i < n_modes; i++){
	      auto k_dot_r = k(0,i) * x_Ph[0] + k(1,i) * x_Ph[1];
              bx2_0 += TWO * k(0,i) * (a_real(i) * math::sin(k_dot_r) + a_imag(i) * math::cos( k_dot_r ));
              bx2_0 += TWO * k(0,i) * (a_real_inv(i) * math::sin(k_dot_r) + a_imag_inv(i) * math::cos( k_dot_r ));

      }
      return bx2_0;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return ONE;
    }

    array_t<real_t**> k;
    array_t<real_t*> a_real;
    array_t<real_t*> a_imag;
    array_t<real_t*> a_real_inv;
    array_t<real_t*> a_imag_inv;
    size_t n_modes;
  };
 
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
	, k { "wavevector", D, wavenumbers.size() }
	, u_imag { "u imaginary", wavenumbers.size() }
	, u_real { "u_real", wavenumbers.size() } 
	, a_real { "a_real", wavenumbers.size() }
	, a_imag { "a_imag", wavenumbers.size() }
	, a_real_inv { "a_real_inv", wavenumbers.size() }
        , a_imag_inv { "a_imag_inv", wavenumbers.size() }
	, A0 {"A0", wavenumbers.size()}
	{
		// initializing wavevectors
		auto k_host = Kokkos::create_mirror_view(k); 
		if constexpr(D == Dim::_2D){
			for (auto i = 0; i < n_modes; i++){
                          k_host(0,i) = constant::TWO_PI * wavenumbers[i][0] / Lx;
			  k_host(1,i) = constant::TWO_PI * wavenumbers[i][1] / Ly;
                          printf("k(%d) = (%f, %f)\n", i,k_host(0,i), k_host(1,i));
			}
                }
		if constexpr(D == Dim::_3D){
			for (auto i = 0; i < n_modes; i++){
                          k_host(0,i) = constant::TWO_PI * wavenumbers[i][0] / Lx;
			  k_host(1,i) = constant::TWO_PI * wavenumbers[i][1] / Ly;
			  k_host(2,i) = constant::TWO_PI * wavenumbers[i][2] / Lz;
                          printf("k(%d) = (%f, %f, %f)\n", i,k_host(0,i), k_host(1,i), k_host(2,i));
			}
		}
		// initializing initial complex amplitudes
		auto a_real_host = Kokkos::create_mirror_view(a_real);
	        auto a_imag_host = Kokkos::create_mirror_view(a_imag); 
	        auto a_real_inv_host = Kokkos::create_mirror_view(a_real_inv);
	        auto a_imag_inv_host = Kokkos::create_mirror_view(a_imag_inv);	
	        auto A0_host = Kokkos::create_mirror_view(A0); 
		real_t prefac;
		if constexpr(D == Dim::_2D){
			real_t prefac = HALF; //HALF = 1/sqrt(twice modes due to reality condition * twice the frequencies due to sign change)
		}	
		if constexpr(D == Dim::_3D){
			real_t prefac = ONE/math::sqrt(TWO); //1/sqrt(2) = 1/sqrt(twice modes due to reality condition)
		}
		for (auto i = 0; i < n_modes; i++){
				auto k_perp = math::sqrt(k_host(0,i) * k_host(0,i) + k_host(1,i) * k_host(1,i));
				auto phase = constant::TWO_PI / 6.;
				A0_host(i) =  dB / math::sqrt((real_t) n_modes) / k_perp * prefac;
				a_real_host(i) = A0_host(i) * math::cos(phase);
				a_imag_host(i) = A0_host(i) * math::sin(phase);
				phase = constant::TWO_PI / 3;
				a_imag_inv_host(i) = A0_host(i) * math::cos(phase);
				a_real_inv_host(i) = A0_host(i) * math::sin(phase);
				printf("A0(%d) = %f\n", i,A0_host(i));
				printf("a_real(%d) = %f\n", i,a_real_host(i));
				printf("a_imag(%d) = %f\n", i,a_imag_host(i));


		}

		Kokkos::deep_copy(a_real, a_real_host);
		Kokkos::deep_copy(a_imag, a_imag_host); 
		Kokkos::deep_copy(a_real_inv, a_real_inv_host);
                Kokkos::deep_copy(a_imag_inv, a_imag_inv_host);
		Kokkos::deep_copy(A0, A0_host);
		Kokkos::deep_copy(k, k_host);
	};


	Inline auto jx3(const coord_t<D>& x_Ph) const  -> real_t {
		if constexpr(D == Dim::_2D){
			real_t jx3_ant = ZERO;
			for (size_t i=0; i < n_modes; i++){
				auto k_perp_sq = k(0,i) * k(0,i) + k(1,i) * k(1,i);
				auto k_dot_r = k(0,i) * x_Ph[0] + k(1,i) * x_Ph[1];
				jx3_ant += TWO * k_perp_sq * (a_real(i) * math::cos(k_dot_r)
							    - a_imag(i) * math::sin(k_dot_r));
				jx3_ant += TWO * k_perp_sq * (a_real_inv(i) * math::cos(k_dot_r)
                                                            - a_imag_inv(i) * math::sin(k_dot_r));

			}
			return jx3_ant;
		}
		if constexpr(D == Dim::_3D){
			real_t jx3_ant = ZERO;
			for (size_t i=0; i < n_modes; i++){
				auto k_perp_sq = k(0,i) * k(0,i) + k(1,i) * k(1,i);
				auto k_dot_r = k(0,i) * x_Ph[0] + k(1,i) * x_Ph[1] + k(2,i) * x_Ph[2];
				jx3_ant += TWO * k_perp_sq * (a_real_inv(i) * math::cos(k_dot_r)
					       		    - a_imag_inv(i) * math::sin(k_dot_r)); 
			}
			return jx3_ant;
		}
	}
	Inline auto jx2(const coord_t<D>& x_Ph) const -> real_t {
		if constexpr(D == Dim::_2D){
			return ZERO;
		} 
		if constexpr(D == Dim::_3D){
			real_t jx2_ant = ZERO;
			for (size_t i = 0; i < n_modes; i++){
				auto k_dot_r = k(0,i) * x_Ph[0] + k(1,i) * x_Ph[1] + k(2,i) * x_Ph[2];
                                jx2_ant -= TWO * k(1,i) * k(2,i) * (a_real_inv(i) * math::cos(k_dot_r)
                                                            - a_imag_inv(i) * math::sin(k_dot_r));

			}
			return jx2_ant;
		}
	}
	Inline auto jx1(const coord_t<D>& x_Ph) const -> real_t {
		if constexpr(D == Dim::_2D){
			return ZERO;
		}
		if constexpr(D == Dim::_3D){
			real_t jx1_ant = ZERO;
			for (size_t i = 0; i < n_modes; i++){
				auto k_dot_r = k(0,i) * x_Ph[0] + k(1,i) * x_Ph[1] + k(2,i) * x_Ph[2];
                                jx1_ant -= TWO * k(0,i) * k(2,i) * (a_real_inv(i) * math::cos(k_dot_r)
                                                            - a_imag_inv(i) * math::sin(k_dot_r));

			}
			return jx1_ant;
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
    const int random_seed;
    std::vector< std::vector<real_t> > wavenumbers;
    random_number_pool_t random_pool;

    // debugging, will delete later
    real_t total_sum = 0.0;
    real_t total_sum_inv = 0.0;
    real_t number_of_timesteps = 0.0;

    ExternalCurrent<D> ext_current;
    InitFields<D> init_flds;

    inline static std::vector<std::vector<real_t>> init_wavenumbers(){
    	if constexpr( D == Dim::_2D){
		 return  { { 1, 0 }, { 0, 1 }, { 1, 1 }, { -1, 1 } };
	}
	if constexpr (D== Dim::_3D){
		 return  { { 1, 0, 1 }, { 0, 1, 1 }, { -1, 0, 1 }, { 0, -1, 1 } };

	}
    }

    inline static unsigned int init_pool(const int seed) {
	if (seed == -1){
#if defined(MPI_ENABLED)
		unsigned int new_seed = rand();
		MPI_Bcast(&new_seed, 1, MPI_UNSIGNED, MPI_ROOT_RANK, MPI_COMM_WORLD);
		return new_seed;
#else 
		return {};
#endif //MPI_ENABLED
	}
	else{
		return seed;
	}
    } 

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , temperature { p.template get<real_t>("setup.temperature") }
      , dB { p.template get<real_t>("setup.dB", 1.) } 
      , omega_0 { p.template get<real_t>("setup.omega_0") }
      , gamma_0 { p.template get<real_t>("setup.gamma_0") }
      , wavenumbers { init_wavenumbers() }
      , random_seed { p.template get<int>("setup.seed", -1) }
      , random_pool { init_pool(random_seed) }
      , Lx { global_domain.mesh().extent(in::x1).second - global_domain.mesh().extent(in::x1).first }
      , Ly { global_domain.mesh().extent(in::x2).second - global_domain.mesh().extent(in::x2).first }
      , Lz { global_domain.mesh().extent(in::x3).second - global_domain.mesh().extent(in::x3).first }
      , dt { params.template get<real_t>("algorithms.timestep.dt") }
      , ext_current {  dB, omega_0, gamma_0, wavenumbers, random_pool, Lx, Ly, Lz }
      , init_flds(ext_current.k, ext_current.a_real, ext_current.a_imag, ext_current.a_real_inv, ext_current.a_imag_inv) {}; 

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
  
    };

    void CustomPostStep(timestep_t, simtime_t time, Domain<S, M>& domain){

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
			    auto a_real_prev = this->ext_current.a_real(i);
			    auto a_imag_prev = this->ext_current.a_imag(i);
			    auto a_real_inv_prev = this->ext_current.a_real_inv(i);
                            auto a_imag_inv_prev = this->ext_current.a_imag_inv(i);
			    auto dt = this->dt;
                            printf(" %i) a_real= %f, a_imag= %f, a_real_inv= %f, a_imag_inv=%f \n",i
                                            , this->ext_current.a_real(i)
                                            , this->ext_current.a_imag(i)
					    , this->ext_current.a_real_inv(i)
					    , this->ext_current.a_imag_inv(i));
			    this->ext_current.a_real(i) = (a_real_prev * math::cos(this->ext_current.omega_0 * dt) + a_imag_prev * math::sin(this->ext_current.omega_0 * dt)) * math::exp(-this->ext_current.gamma_0 * dt) + this->ext_current.A0(i) * math::sqrt(12.0 * this->ext_current.gamma_0 / dt) * u_real * dt;

			    this->ext_current.a_imag(i) = (a_imag_prev * math::cos(this->ext_current.omega_0 * dt) - a_real_prev * math::sin(this->ext_current.omega_0 * dt)) * math::exp(-this->ext_current.gamma_0 * dt) + this->ext_current.A0(i) * math::sqrt(12.0 * this->ext_current.gamma_0 / dt) * u_imag * dt;

			    this->ext_current.a_real_inv(i) = (a_real_inv_prev * math::cos(-this->ext_current.omega_0 * dt) + a_imag_inv_prev * math::sin(-this->ext_current.omega_0 * dt)) * math::exp(-this->ext_current.gamma_0 * dt) + this->ext_current.A0(i) * math::sqrt(12.0 * this->ext_current.gamma_0 / dt) * u_real_inv * dt;

                            this->ext_current.a_imag_inv(i) = (a_imag_inv_prev * math::cos(-this->ext_current.omega_0 * dt) - a_real_inv_prev * math::sin(-this->ext_current.omega_0 * dt)) * math::exp(-this->ext_current.gamma_0 * dt) + this->ext_current.A0(i) * math::sqrt(12.0 * this->ext_current.gamma_0 / dt ) * u_imag_inv * dt;
			    });

//	real_t sum = 0.0;
//	// KOKKOS_LAMBDA macro includes capture-by-value specifier [=].
//	Kokkos::parallel_reduce ("Reduction", wavenumbers.size(), Lambda (const int i, real_t& update) {
//									auto a_real = this->ExternalCurrent.a_real(i);
//									auto a_imag = this->ExternalCurrent.a_imag(i);
//									auto k_perp = this->ExternalCurrent.k(0,i)*this->ExternalCurrent.k(0,i) + this->ExternalCurrent.k(1,i)*this->ExternalCurrent.k(1,i);
//								  update +=  (a_real * a_real + a_imag * a_imag) ;
//											}, sum);
//	total_sum +=sum;
//	number_of_timesteps +=1.0;
//	printf("<an^2> = %f, ", sum);
//        printf("total <a^2> = %f\n", total_sum/number_of_timesteps);
//
//	        real_t sum_inv = 0.0;
//        // KOKKOS_LAMBDA macro includes capture-by-value specifier [=].
//        Kokkos::parallel_reduce ("Reduction", wavenumbers.size(), Lambda (const int i, real_t& update) {
//                                                                        auto a_real = this->ExternalCurrent.a_real_inv(i);
//                                                                        auto a_imag = this->ExternalCurrent.a_imag_inv(i);
//                                                                        auto k_perp = this->ExternalCurrent.k(0,i)*this->ExternalCurrent.k(0,i) + this->ExternalCurrent.k(1,i)*this->ExternalCurrent.k(1,i);
//                                                                  update +=  (a_real * a_real + a_imag * a_imag) ;
//                                                                                        }, sum_inv);
//        total_sum_inv +=sum_inv;
//        printf("<an^2> = %f, ", sum_inv);
//        printf("total <a^2> = %f\n", total_sum_inv/number_of_timesteps);


    }
  };
} // namespace user

#endif
