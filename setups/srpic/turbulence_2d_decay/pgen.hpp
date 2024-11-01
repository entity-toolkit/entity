#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

#include <fstream>
#include <iostream>

enum {
  REAL = 0,
  IMAG = 1
};

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t Bnorm, array_t<real_t* [8]> amplitudes, array_t<real_t* [8]> phi )
      : Bnorm { Bnorm } 
      , B0x1 { ZERO }
      , B0x2 { ZERO }
      , B0x3 { Bnorm } 
      , amp { amplitudes}
      , phi { phi } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {

      real_t dBvec = ZERO;
      for (unsigned short k = 1; k < 9; ++k) {
        for (unsigned short l = 1; l < 9; ++l) {
          if (k == 0 && l == 0) continue;

        real_t kvec1 = constant::TWO_PI * static_cast<real_t>(k);
        real_t kvec2 = constant::TWO_PI * static_cast<real_t>(l); 
        real_t kvec3 = ZERO;

        real_t kb1 = kvec2 * B0x3 - kvec3 * B0x2;
        real_t kb2 = kvec3 * B0x1 - kvec1 * B0x3;
        real_t kb3 = kvec1 * B0x2 - kvec2 * B0x1;
        real_t kbnorm = math::sqrt(kb1*kb1 + kb2*kb2 + kb3*kb3);
        real_t kdotx = kvec1 * x_Ph[0] + kvec2 * x_Ph[1];

        dBvec -= TWO * amp(k, l) * kb1 / kbnorm * math::sin(kdotx + phi(k, l));

        }
      }

      return B0x1 + dBvec;       

    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      
      real_t dBvec = ZERO;
      for (unsigned short k = 1; k < 9; ++k) {
        for (unsigned short l = 1; l < 9; ++l) {
          if (k == 0 && l == 0) continue;

        real_t kvec1 = constant::TWO_PI * static_cast<real_t>(k);
        real_t kvec2 = constant::TWO_PI * static_cast<real_t>(l); 
        real_t kvec3 = ZERO;

        real_t kb1 = kvec2 * B0x3 - kvec3 * B0x2;
        real_t kb2 = kvec3 * B0x1 - kvec1 * B0x3;
        real_t kb3 = kvec1 * B0x2 - kvec2 * B0x1;
        real_t kbnorm = math::sqrt(kb1*kb1 + kb2*kb2 + kb3*kb3);
        real_t kdotx = kvec1 * x_Ph[0] + kvec2 * x_Ph[1];

        dBvec -= TWO * amp(k, l) * kb2 / kbnorm * math::sin(kdotx + phi(k, l));

        }
      }

      return B0x2 + dBvec;        
      }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      
      real_t dBvec = ZERO;
      for (unsigned short k = 1; k < 9; ++k) {
        for (unsigned short l = 1; l < 9; ++l) {
          if (k == 0 && l == 0) continue;

        real_t kvec1 = constant::TWO_PI * static_cast<real_t>(k);
        real_t kvec2 = constant::TWO_PI * static_cast<real_t>(l); 
        real_t kvec3 = ZERO;

        real_t kb1 = kvec2 * B0x3 - kvec3 * B0x2;
        real_t kb2 = kvec3 * B0x1 - kvec1 * B0x3;
        real_t kb3 = kvec1 * B0x2 - kvec2 * B0x1;
        real_t kbnorm = math::sqrt(kb1*kb1 + kb2*kb2 + kb3*kb3);
        real_t kdotx = kvec1 * x_Ph[0] + kvec2 * x_Ph[1];

        dBvec -= TWO * amp(k, l) * kb3 / kbnorm * math::sin(kdotx + phi(k, l));

        }
      }

      return B0x3 + dBvec;        
    }

  private:
    const real_t Bnorm;
    const real_t B0x1, B0x2, B0x3;
    array_t<real_t* [8]> amp;
    array_t<real_t* [8]> phi;
  };

  template <SimEngine::type S, class M>
  struct PowerlawDist : public arch::EnergyDistribution<S, M> {
    PowerlawDist(const M&               metric,
             random_number_pool_t&      pool,
             real_t                     g_min,
             real_t                     g_max,
             real_t                     pl_ind)
      : arch::EnergyDistribution<S, M> { metric }
      , g_min { g_min }
      , g_max { g_max }
      , random_pool { pool }
      , pl_ind { pl_ind } {}

    Inline void operator()(const coord_t<M::Dim>& x_Ph,
                           vec_t<Dim::_3D>&       v,
                           unsigned short         sp) const override {
      // if (sp == 1) {
         auto   rand_gen = random_pool.get_state();
         auto   rand_X1 = Random<real_t>(rand_gen);
         auto   rand_gam = ONE;
         if (pl_ind != -1.0) {
            rand_gam += math::pow(math::pow(g_min,ONE + pl_ind) + (-math::pow(g_min,ONE + pl_ind) + math::pow(g_max,ONE + pl_ind))*rand_X1,ONE/(ONE + pl_ind));
         } else {
            rand_gam += math::pow(g_min,ONE - rand_X1)*math::pow(g_max,rand_X1);
         }
         auto   rand_u = math::sqrt( SQR(rand_gam) - ONE );

        if constexpr (M::Dim == Dim::_1D) {
          v[0] = ZERO;
        } else if constexpr (M::Dim == Dim::_2D) {
          v[0] = ZERO;
          v[1] = ZERO;
        } else {
          auto rand_X2 = Random<real_t>(rand_gen);
          auto rand_X3 = Random<real_t>(rand_gen);
          v[0]   = rand_u * (TWO * rand_X2 - ONE);
          v[2]   = TWO * rand_u * math::sqrt(rand_X2 * (ONE - rand_X2));
          v[1]   = v[2] * math::cos(constant::TWO_PI * rand_X3);
          v[2]   = v[2] * math::sin(constant::TWO_PI * rand_X3);
        }
        random_pool.free_state(rand_gen);
      // } else {
      //   v[0] = ZERO;
      //   v[1] = ZERO;
      //   v[2] = ZERO;
      // }
    }

  private:
    const real_t g_min, g_max, pl_ind;
    random_number_pool_t random_pool;
  };


  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t         SX1, SX2, SX3;
    const real_t         temperature, machno, Bnorm;
    const unsigned int   nmodes;
    const real_t         amp0;
    const real_t        pl_gamma_min, pl_gamma_max, pl_index;
    array_t<real_t* [8]> amplitudes, phi0;
    const real_t         dt;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& params, Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { params }
      , SX1 { global_domain.mesh().extent(in::x1).second -
              global_domain.mesh().extent(in::x1).first }
      , SX2 { global_domain.mesh().extent(in::x2).second -
              global_domain.mesh().extent(in::x2).first }
      // , SX3 { global_domain.mesh().extent(in::x3).second -
      //         global_domain.mesh().extent(in::x3).first }
      , SX3 { TWO }
      , temperature { params.template get<real_t>("setup.temperature", 0.16) }
      , machno { params.template get<real_t>("setup.machno", 1.0) }
      , nmodes { params.template get<unsigned int>("setup.nmodes", 8) }
      , Bnorm { params.template get<real_t>("setup.Bnorm", 0.0) }
      , pl_gamma_min { params.template get<real_t>("setup.pl_gamma_min", 0.1) }
      , pl_gamma_max { params.template get<real_t>("setup.pl_gamma_max", 100.0) }
      , pl_index { params.template get<real_t>("setup.pl_index", -2.0) }  
      , amp0 { machno * temperature / static_cast<real_t>(8) }
      , phi0 { "DrivingPhases", 8 }
      , amplitudes { "DrivingModes", 8 }
      , init_flds { Bnorm, amplitudes, phi0 }
      , dt { params.template get<real_t>("algorithms.timestep.dt") } {
      // Initializing random phases
      auto phi0_ = Kokkos::create_mirror_view(phi0);
      auto amplitudes_ = Kokkos::create_mirror_view(amplitudes);
      srand (static_cast <unsigned> (12345));
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
          phi0_(i, j) = constant::TWO_PI * static_cast <real_t> (rand()) / static_cast <real_t> (RAND_MAX);
          amplitudes_(i, j) = amp0 * static_cast <real_t> (rand()) / static_cast <real_t> (RAND_MAX);
        }
      }
      Kokkos::deep_copy(phi0, phi0_);
      Kokkos::deep_copy(amplitudes, amplitudes_);
    }

    inline void InitPrtls(Domain<S, M>& local_domain) {
      {
        const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                        local_domain.random_pool,
                                                        temperature);
        const auto injector = arch::UniformInjector<S, M, arch::Maxwellian>(
          energy_dist,
          { 1, 2 });
        const real_t ndens = 1.0;
        arch::InjectUniform<S, M, decltype(injector)>(params,
                                                      local_domain,
                                                      injector,
                                                      ndens);
      }

      {
        // const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
        //                                                 local_domain.random_pool,
        //                                                 temperature*100);        
        // const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
        //                                                 local_domain.random_pool,
        //                                                 temperature * 2,
        //                                                 10.0,
        //                                                 1);
        // const auto injector = arch::UniformInjector<S, M, arch::Maxwellian>(
        //   energy_dist,
        //   { 1, 2 });

        const auto energy_dist = PowerlawDist<S, M>(local_domain.mesh.metric,
                                                     local_domain.random_pool,
                                                     pl_gamma_min,
                                                     pl_gamma_max,
                                                     pl_index);  

        const auto injector = arch::UniformInjector<S, M, PowerlawDist>(
          energy_dist,
          { 1, 2 });  


        const real_t ndens = 0.0;
        arch::InjectUniform<S, M, decltype(injector)>(params,
                                                      local_domain,
                                                      injector,
                                                      ndens);
      }
    }

    void CustomPostStep(std::size_t time, long double, Domain<S, M>& domain) {
      auto omega0 = 0.5*0.6 * math::sqrt(temperature * machno) * constant::TWO_PI / SX1;
      auto gamma0 = 0.5*0.5 * math::sqrt(temperature * machno) * constant::TWO_PI / SX2;
      auto sigma0 = amp0 * math::sqrt(static_cast<real_t>(nmodes) * gamma0 / dt);
      auto pool   = domain.random_pool;

      #if defined(MPI_ENABLED)
        int              rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      #endif

      Kokkos::parallel_for(
        "RandomAmplitudes",
        amplitudes.extent(0),
        ClassLambda(index_t i) {
          auto       rand_gen = pool.get_state();
          const auto unr      = Random<real_t>(rand_gen) - HALF;
          const auto uni      = Random<real_t>(rand_gen) - HALF;
          pool.free_state(rand_gen);
          const auto ampr_prev = amplitudes(i, REAL);
          const auto ampi_prev = amplitudes(i, IMAG);
          amplitudes(i, REAL)  = (ampr_prev * math::cos(omega0 * dt) +
                                 ampi_prev * math::sin(omega0 * dt)) *
                                  math::exp(-gamma0 * dt) +
                                unr * sigma0 * dt;
          amplitudes(i, IMAG) = (-ampr_prev * math::sin(omega0 * dt) +
                                 ampi_prev * math::cos(omega0 * dt)) *
                                  math::exp(-gamma0 * dt) +
                                uni * sigma0 * dt;
        });

      auto amplitudes_ = Kokkos::create_mirror_view(amplitudes);
      Kokkos::deep_copy(amplitudes_, amplitudes);
      for (int i = 0; i < nmodes; ++i) {
        printf("amplitudes_(%d, REAL) = %f\n", i, amplitudes_(i, REAL));
      }

      auto fext_en_total = ZERO;
      for (auto& species : domain.species) {
        auto fext_en_s = ZERO;
        auto pld    = species.pld[0];
        auto weight = species.weight;
        Kokkos::parallel_reduce(
          "ExtForceEnrg",
          species.rangeActiveParticles(),
          ClassLambda(index_t p, real_t & fext_en) {
            fext_en += pld(p) * weight(p);
          },
          fext_en_s);
      #if defined(MPI_ENABLED)
        auto fext_en_sg = ZERO;
        MPI_Allreduce(&fext_en_s, &fext_en_sg, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
        fext_en_total += fext_en_sg;
      #else
        fext_en_total += fext_en_s; 
      #endif
      }

      // Weight the macroparticle integral by sim parameters
      fext_en_total /= params.template get<real_t>("scales.n0");

      auto pkin_en_total = ZERO;
      for (auto& species : domain.species) {
        auto pkin_en_s = ZERO;
        auto ux1    = species.ux1;
        auto ux2    = species.ux2;
        auto ux3    = species.ux3;
        auto weight = species.weight;
        Kokkos::parallel_reduce(
          "KinEnrg",
          species.rangeActiveParticles(),
          ClassLambda(index_t p, real_t & pkin_en) {
            pkin_en += (math::sqrt(ONE + SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p))) -
                        ONE) *
                       weight(p);
          },
          pkin_en_s);
      #if defined(MPI_ENABLED)
        auto pkin_en_sg = ZERO;
        MPI_Allreduce(&pkin_en_s, &pkin_en_sg, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
        pkin_en_total += pkin_en_sg;
      #else
        pkin_en_total += pkin_en_s;
      #endif
      }

      // Weight the macroparticle integral by sim parameters
      pkin_en_total /= params.template get<real_t>("scales.n0");
        
      auto benrg_total = ZERO;
      auto eenrg_total = ZERO;

      if constexpr (D == Dim::_3D) {
        
        auto metric = domain.mesh.metric;
        
        auto benrg_s = ZERO;
        auto EB          = domain.fields.em;
        Kokkos::parallel_reduce(
          "BEnrg",
          domain.mesh.rangeActiveCells(),
          Lambda(index_t i1, index_t i2, index_t i3, real_t & benrg) {
            coord_t<Dim::_3D> x_Cd { ZERO };
            vec_t<Dim::_3D>   b_Cntrv { EB(i1, i2, i3, em::bx1),
                                      EB(i1, i2, i3, em::bx2),
                                      EB(i1, i2, i3, em::bx3) };
            vec_t<Dim::_3D>   b_XYZ;
            metric.template transform<Idx::U, Idx::T>(x_Cd,
                                                                  b_Cntrv,
                                                                  b_XYZ);
            benrg += (SQR(b_XYZ[0]) + SQR(b_XYZ[1]) + SQR(b_XYZ[2]));
          },
          benrg_s);
        #if defined(MPI_ENABLED)
          auto benrg_sg = ZERO;
          MPI_Allreduce(&benrg_s, &benrg_sg, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
          benrg_total += benrg_sg;
        #else
          benrg_total += benrg_s;
        #endif

      // Weight the field integral by sim parameters
        benrg_total *= params.template get<real_t>("scales.V0") * params.template get<real_t>("scales.sigma0") * HALF;

        auto eenrg_s = ZERO;
        Kokkos::parallel_reduce(
          "BEnrg",
          domain.mesh.rangeActiveCells(),
          Lambda(index_t i1, index_t i2, index_t i3, real_t & eenrg) {
            coord_t<Dim::_3D> x_Cd { ZERO };
            vec_t<Dim::_3D>   e_Cntrv { EB(i1, i2, i3, em::ex1),
                                      EB(i1, i2, i3, em::ex2),
                                      EB(i1, i2, i3, em::ex3) };
            vec_t<Dim::_3D>   e_XYZ;
            metric.template transform<Idx::U, Idx::T>(x_Cd, e_Cntrv, e_XYZ);            
            eenrg += (SQR(e_XYZ[0]) + SQR(e_XYZ[1]) + SQR(e_XYZ[2]));
          },
          eenrg_s);

        #if defined(MPI_ENABLED)
          auto eenrg_sg = ZERO;
          MPI_Allreduce(&eenrg_s, &eenrg_sg, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
          eenrg_total += eenrg_sg;  
        #else
          eenrg_total += eenrg_s;
        #endif

      // Weight the field integral by sim parameters
        eenrg_total *= params.template get<real_t>("scales.V0") * params.template get<real_t>("scales.sigma0") * HALF;

      }

      if constexpr (D == Dim::_2D) {
        
        auto metric = domain.mesh.metric;
        
        auto benrg_s = ZERO;
        auto EB          = domain.fields.em;
        Kokkos::parallel_reduce(
          "BEnrg",
          domain.mesh.rangeActiveCells(),
          Lambda(index_t i1, index_t i2, real_t & benrg) {
            coord_t<Dim::_2D> x_Cd { ZERO };
            vec_t<Dim::_3D>   b_Cntrv { EB(i1, i2, em::bx1),
                                      EB(i1, i2, em::bx2),
                                      EB(i1, i2, em::bx3) };
            vec_t<Dim::_3D>   b_XYZ;
            metric.template transform<Idx::U, Idx::T>(x_Cd,
                                                                  b_Cntrv,
                                                                  b_XYZ);
            benrg += (SQR(b_XYZ[0]) + SQR(b_XYZ[1]) + SQR(b_XYZ[2]));
          },
          benrg_s);
        #if defined(MPI_ENABLED)
          auto benrg_sg = ZERO;
          MPI_Allreduce(&benrg_s, &benrg_sg, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
          benrg_total += benrg_sg;
        #else
          benrg_total += benrg_s;
        #endif

      // Weight the field integral by sim parameters
        benrg_total *= params.template get<real_t>("scales.V0") * params.template get<real_t>("scales.sigma0") * HALF;

        auto eenrg_s = ZERO;
        Kokkos::parallel_reduce(
          "BEnrg",
          domain.mesh.rangeActiveCells(),
          Lambda(index_t i1, index_t i2, real_t & eenrg) {
            coord_t<Dim::_2D> x_Cd { ZERO };
            vec_t<Dim::_3D>   e_Cntrv { EB(i1, i2, em::ex1),
                                      EB(i1, i2, em::ex2),
                                      EB(i1, i2, em::ex3) };
            vec_t<Dim::_3D>   e_XYZ;
            metric.template transform<Idx::U, Idx::T>(x_Cd, e_Cntrv, e_XYZ);            
            eenrg += (SQR(e_XYZ[0]) + SQR(e_XYZ[1]) + SQR(e_XYZ[2]));
          },
          eenrg_s);

        #if defined(MPI_ENABLED)
          auto eenrg_sg = ZERO;
          MPI_Allreduce(&eenrg_s, &eenrg_sg, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
          eenrg_total += eenrg_sg;  
        #else
          eenrg_total += eenrg_s;
        #endif

      // Weight the field integral by sim parameters
        eenrg_total *= params.template get<real_t>("scales.V0") * params.template get<real_t>("scales.sigma0") * HALF;

      }

      std::ofstream myfile1;
      std::ofstream myfile2;
      std::ofstream myfile3;
      std::ofstream myfile4;

      #if defined(MPI_ENABLED)

        if(rank == MPI_ROOT_RANK) {

          printf("fext_en_total: %f, pkin_en_total: %f, benrg_total: %f, eenrg_total: %f, MPI rank %d\n", fext_en_total, pkin_en_total, benrg_total, eenrg_total, MPI_ROOT_RANK);
          
          if (time == 0) {
            myfile1.open("fextenrg.txt");
          } else {
            myfile1.open("fextenrg.txt", std::ios_base::app);
          }
          myfile1 << fext_en_total << std::endl;

          if (time == 0) {
            myfile2.open("kenrg.txt");
          } else {
            myfile2.open("kenrg.txt", std::ios_base::app);
          }
          myfile2 << pkin_en_total << std::endl;

          if (time == 0) {
            myfile3.open("bsqenrg.txt");
          } else {
            myfile3.open("bsqenrg.txt", std::ios_base::app);
          }
          myfile3 << benrg_total << std::endl;

          if (time == 0) {
            myfile4.open("esqenrg.txt");
          } else {
            myfile4.open("esqenrg.txt", std::ios_base::app);
          }
          myfile4 << eenrg_total << std::endl;
        }

      #else

          if (time == 0) {
            myfile1.open("fextenrg.txt");
          } else {
            myfile1.open("fextenrg.txt", std::ios_base::app);
          }
          myfile1 << fext_en_total << std::endl;

          if (time == 0) {
            myfile2.open("kenrg.txt");
          } else {
            myfile2.open("kenrg.txt", std::ios_base::app);
          }
          myfile2 << pkin_en_total << std::endl;

          if (time == 0) {
            myfile3.open("bsqenrg.txt");
          } else {
            myfile3.open("bsqenrg.txt", std::ios_base::app);
          }
          myfile3 << benrg_total << std::endl;

          if (time == 0) {
            myfile4.open("esqenrg.txt");
          } else {
            myfile4.open("esqenrg.txt", std::ios_base::app);
          }
          myfile4 << eenrg_total << std::endl;

      #endif
    }
  };

} // namespace user

#endif