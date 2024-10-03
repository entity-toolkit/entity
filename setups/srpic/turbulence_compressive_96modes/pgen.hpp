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
    InitFields(real_t Bnorm)
      : Bnorm { Bnorm } {
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return Bnorm;
    }

  private:
    const real_t Bnorm;
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

  template <Dimension D>
  struct ExtForce {
    ExtForce(array_t<real_t* [2]> amplitudes, real_t SX1, real_t SX2, real_t SX3)
      : amps { amplitudes }
      , sx1 { SX1 }
      , sx2 { SX2 }
      , sx3 { SX3 } 

      // modes with |K| = 1

      , k01x {  ONE * constant::TWO_PI / sx1 }
      , k01y { ZERO * constant::TWO_PI / sx2 }
      , k01z { ZERO * constant::TWO_PI / sx3 }

      , k02x { -ONE * constant::TWO_PI / sx1 }
      , k02y { ZERO * constant::TWO_PI / sx2 }
      , k02z { ZERO * constant::TWO_PI / sx3 }  

      , k03x { ZERO * constant::TWO_PI / sx1 }
      , k03y {  ONE * constant::TWO_PI / sx2 }
      , k03z { ZERO * constant::TWO_PI / sx3 }

      , k04x { ZERO * constant::TWO_PI / sx1 }
      , k04y { -ONE * constant::TWO_PI / sx2 }
      , k04z { ZERO * constant::TWO_PI / sx3 } 

      , k05x { ZERO * constant::TWO_PI / sx1 }
      , k05y { ZERO * constant::TWO_PI / sx2 }
      , k05z {  ONE * constant::TWO_PI / sx3 }

      , k06x { ZERO * constant::TWO_PI / sx1 }
      , k06y { ZERO * constant::TWO_PI / sx2 }
      , k06z { -ONE * constant::TWO_PI / sx3 }  

      // modes with |K| = sqrt(2)

      , k07x {  ONE * constant::TWO_PI / sx1 }
      , k07y {  ONE * constant::TWO_PI / sx2 }
      , k07z { ZERO * constant::TWO_PI / sx3 }

      , k08x { -ONE * constant::TWO_PI / sx1 }
      , k08y {  ONE * constant::TWO_PI / sx2 }
      , k08z { ZERO * constant::TWO_PI / sx3 }    

      , k09x {  ONE * constant::TWO_PI / sx1 }
      , k09y { -ONE * constant::TWO_PI / sx2 }
      , k09z { ZERO * constant::TWO_PI / sx3 }  

      , k10x { -ONE * constant::TWO_PI / sx1 }
      , k10y { -ONE * constant::TWO_PI / sx2 }
      , k10z { ZERO * constant::TWO_PI / sx3 }

      , k11x {  ONE * constant::TWO_PI / sx1 }
      , k11y { ZERO * constant::TWO_PI / sx2 }
      , k11z {  ONE * constant::TWO_PI / sx3 }  

      , k12x { -ONE * constant::TWO_PI / sx1 }
      , k12y { ZERO * constant::TWO_PI / sx2 }
      , k12z {  ONE * constant::TWO_PI / sx3 }

      , k13x {  ONE * constant::TWO_PI / sx1 }
      , k13y { ZERO * constant::TWO_PI / sx2 }
      , k13z { -ONE * constant::TWO_PI / sx3 }  

      , k14x { -ONE * constant::TWO_PI / sx1 }
      , k14y { ZERO * constant::TWO_PI / sx2 }
      , k14z { -ONE * constant::TWO_PI / sx3 }  

      , k15x { ZERO * constant::TWO_PI / sx1 }
      , k15y {  ONE * constant::TWO_PI / sx2 }
      , k15z {  ONE * constant::TWO_PI / sx3 }  

      , k16x { ZERO * constant::TWO_PI / sx1 }
      , k16y { -ONE * constant::TWO_PI / sx2 }
      , k16z {  ONE * constant::TWO_PI / sx3 }  

      , k17x { ZERO * constant::TWO_PI / sx1 }
      , k17y {  ONE * constant::TWO_PI / sx2 }
      , k17z { -ONE * constant::TWO_PI / sx3 }  

      , k18x { ZERO * constant::TWO_PI / sx1 }
      , k18y { -ONE * constant::TWO_PI / sx2 }
      , k18z { -ONE * constant::TWO_PI / sx3 }  

      // modes with |K| = sqrt(3)

      , k19x {  ONE * constant::TWO_PI / sx1 }
      , k19y {  ONE * constant::TWO_PI / sx2 }
      , k19z {  ONE * constant::TWO_PI / sx3 }

      , k20x { -ONE * constant::TWO_PI / sx1 }
      , k20y {  ONE * constant::TWO_PI / sx2 }
      , k20z {  ONE * constant::TWO_PI / sx3 }

      , k21x {  ONE * constant::TWO_PI / sx1 }
      , k21y { -ONE * constant::TWO_PI / sx2 }
      , k21z {  ONE * constant::TWO_PI / sx3 }

      , k22x {  ONE * constant::TWO_PI / sx1 }
      , k22y {  ONE * constant::TWO_PI / sx2 }
      , k22z { -ONE * constant::TWO_PI / sx3 }

      , k23x { -ONE * constant::TWO_PI / sx1 }
      , k23y { -ONE * constant::TWO_PI / sx2 }
      , k23z {  ONE * constant::TWO_PI / sx3 }

      , k24x { -ONE * constant::TWO_PI / sx1 }
      , k24y {  ONE * constant::TWO_PI / sx2 }
      , k24z { -ONE * constant::TWO_PI / sx3 }

      , k25x {  ONE * constant::TWO_PI / sx1 }
      , k25y { -ONE * constant::TWO_PI / sx2 }
      , k25z { -ONE * constant::TWO_PI / sx3 }

      , k26x { -ONE * constant::TWO_PI / sx1 }
      , k26y { -ONE * constant::TWO_PI / sx2 }
      , k26z { -ONE * constant::TWO_PI / sx3 }

      // modes with |K| = 2
      
      , k27x {  TWO * constant::TWO_PI / sx1 }
      , k27y { ZERO * constant::TWO_PI / sx2 }
      , k27z { ZERO * constant::TWO_PI / sx3 }

      , k28x { -TWO * constant::TWO_PI / sx1 }
      , k28y { ZERO * constant::TWO_PI / sx2 }
      , k28z { ZERO * constant::TWO_PI / sx3 }   

      , k29x { ZERO * constant::TWO_PI / sx1 }
      , k29y {  TWO * constant::TWO_PI / sx2 }
      , k29z { ZERO * constant::TWO_PI / sx3 }

      , k30x { ZERO * constant::TWO_PI / sx1 }
      , k30y { -TWO * constant::TWO_PI / sx2 }
      , k30z { ZERO * constant::TWO_PI / sx3 }  

      , k31x { ZERO * constant::TWO_PI / sx1 }
      , k31y { ZERO * constant::TWO_PI / sx2 }
      , k31z {  TWO * constant::TWO_PI / sx3 }
      
      , k32x { ZERO * constant::TWO_PI / sx1 }
      , k32y { ZERO * constant::TWO_PI / sx2 }
      , k32z { -TWO * constant::TWO_PI / sx3 }  {}

    const std::vector<unsigned short> species { 1, 2 };

    ExtForce() = default;

    Inline auto fx1(const unsigned short&,
                    const real_t&,
                    const coord_t<D>& x_Ph) const -> real_t {
      // return ZERO;
      return (k01x * amps(0,REAL) * math::cos(k01x * x_Ph[0] + k01y * x_Ph[1] + k01z * x_Ph[2]) +
              k01x * amps(0,IMAG) * math::sin(k01x * x_Ph[0] + k01y * x_Ph[1] + k01z * x_Ph[2])) +
             (k02x * amps(3,REAL) * math::cos(k02x * x_Ph[0] + k02y * x_Ph[1] + k02z * x_Ph[2]) +
              k02x * amps(3,IMAG) * math::sin(k02x * x_Ph[0] + k02y * x_Ph[1] + k02z * x_Ph[2])) +
             (amps(6,REAL) * math::cos(k03x * x_Ph[0] + k03y * x_Ph[1] + k03z * x_Ph[2]) +
              amps(6,IMAG) * math::sin(k03x * x_Ph[0] + k03y * x_Ph[1] + k03z * x_Ph[2])) +
             (amps(9,REAL) * math::cos(k04x * x_Ph[0] + k04y * x_Ph[1] + k04z * x_Ph[2]) +
              amps(9,IMAG) * math::sin(k04x * x_Ph[0] + k04y * x_Ph[1] + k04z * x_Ph[2])) +
             (amps(12,REAL) * math::cos(k05x * x_Ph[0] + k05y * x_Ph[1] + k05z * x_Ph[2]) +
              amps(12,IMAG) * math::sin(k05x * x_Ph[0] + k05y * x_Ph[1] + k05z * x_Ph[2])) +
             (amps(15,REAL) * math::cos(k06x * x_Ph[0] + k06y * x_Ph[1] + k06z * x_Ph[2]) +
              amps(15,IMAG) * math::sin(k06x * x_Ph[0] + k06y * x_Ph[1] + k06z * x_Ph[2])) +
             (k07y * amps(18,REAL) * math::cos(k07x * x_Ph[0] + k07y * x_Ph[1] + k07z * x_Ph[2]) +
              k07y * amps(18,IMAG) * math::sin(k07x * x_Ph[0] + k07y * x_Ph[1] + k07z * x_Ph[2])) +              
             (k07x * amps(20,REAL) * math::cos(k07x * x_Ph[0] + k07y * x_Ph[1] + k07z * x_Ph[2]) +
              k07x * amps(20,IMAG) * math::sin(k07x * x_Ph[0] + k07y * x_Ph[1] + k07z * x_Ph[2])) +   
             (k08y * amps(21,REAL) * math::cos(k08x * x_Ph[0] + k08y * x_Ph[1] + k08z * x_Ph[2]) +
              k08y * amps(21,IMAG) * math::sin(k08x * x_Ph[0] + k08y * x_Ph[1] + k08z * x_Ph[2])) +              
             (k08x * amps(23,REAL) * math::cos(k08x * x_Ph[0] + k08y * x_Ph[1] + k08z * x_Ph[2]) +
              k08x * amps(23,IMAG) * math::sin(k08x * x_Ph[0] + k08y * x_Ph[1] + k08z * x_Ph[2])) +   
             (k09y * amps(24,REAL) * math::cos(k09x * x_Ph[0] + k09y * x_Ph[1] + k09z * x_Ph[2]) +
              k09y * amps(24,IMAG) * math::sin(k09x * x_Ph[0] + k09y * x_Ph[1] + k09z * x_Ph[2])) +              
             (k09x * amps(26,REAL) * math::cos(k09x * x_Ph[0] + k09y * x_Ph[1] + k09z * x_Ph[2]) +
              k09x * amps(26,IMAG) * math::sin(k09x * x_Ph[0] + k09y * x_Ph[1] + k09z * x_Ph[2])) +   
             (k10y * amps(27,REAL) * math::cos(k10x * x_Ph[0] + k10y * x_Ph[1] + k10z * x_Ph[2]) +
              k10y * amps(27,IMAG) * math::sin(k10x * x_Ph[0] + k10y * x_Ph[1] + k10z * x_Ph[2])) +              
             (k10x * amps(29,REAL) * math::cos(k10x * x_Ph[0] + k10y * x_Ph[1] + k10z * x_Ph[2]) +
              k10x * amps(29,IMAG) * math::sin(k10x * x_Ph[0] + k10y * x_Ph[1] + k10z * x_Ph[2])) +   
             (k11z * amps(30,REAL) * math::cos(k11x * x_Ph[0] + k11y * x_Ph[1] + k11z * x_Ph[2]) +
              k11z * amps(30,IMAG) * math::sin(k11x * x_Ph[0] + k11y * x_Ph[1] + k11z * x_Ph[2])) +              
             (k11x * amps(32,REAL) * math::cos(k11x * x_Ph[0] + k11y * x_Ph[1] + k11z * x_Ph[2]) +
              k11x * amps(32,IMAG) * math::sin(k11x * x_Ph[0] + k11y * x_Ph[1] + k11z * x_Ph[2])) +      
             (k12z * amps(33,REAL) * math::cos(k12x * x_Ph[0] + k12y * x_Ph[1] + k12z * x_Ph[2]) +
              k12z * amps(33,IMAG) * math::sin(k12x * x_Ph[0] + k12y * x_Ph[1] + k12z * x_Ph[2])) +              
             (k12x * amps(35,REAL) * math::cos(k12x * x_Ph[0] + k12y * x_Ph[1] + k12z * x_Ph[2]) +
              k12x * amps(35,IMAG) * math::sin(k12x * x_Ph[0] + k12y * x_Ph[1] + k12z * x_Ph[2])) +        
             (k13z * amps(36,REAL) * math::cos(k13x * x_Ph[0] + k13y * x_Ph[1] + k13z * x_Ph[2]) +
              k13z * amps(36,IMAG) * math::sin(k13x * x_Ph[0] + k13y * x_Ph[1] + k13z * x_Ph[2])) +              
             (k13x * amps(38,REAL) * math::cos(k13x * x_Ph[0] + k13y * x_Ph[1] + k13z * x_Ph[2]) +
              k13x * amps(38,IMAG) * math::sin(k13x * x_Ph[0] + k13y * x_Ph[1] + k13z * x_Ph[2])) +       
             (k14z * amps(39,REAL) * math::cos(k14x * x_Ph[0] + k14y * x_Ph[1] + k14z * x_Ph[2]) +
              k14z * amps(39,IMAG) * math::sin(k14x * x_Ph[0] + k14y * x_Ph[1] + k14z * x_Ph[2])) +              
             (k14x * amps(41,REAL) * math::cos(k14x * x_Ph[0] + k14y * x_Ph[1] + k14z * x_Ph[2]) +
              k14x * amps(41,IMAG) * math::sin(k14x * x_Ph[0] + k14y * x_Ph[1] + k14z * x_Ph[2])) +    
             (amps(42,REAL) * math::cos(k15x * x_Ph[0] + k15y * x_Ph[1] + k15z * x_Ph[2]) +
              amps(42,IMAG) * math::sin(k15x * x_Ph[0] + k15y * x_Ph[1] + k15z * x_Ph[2])) +
             (amps(45,REAL) * math::cos(k16x * x_Ph[0] + k16y * x_Ph[1] + k16z * x_Ph[2]) +
              amps(45,IMAG) * math::sin(k16x * x_Ph[0] + k16y * x_Ph[1] + k16z * x_Ph[2])) +
             (amps(48,REAL) * math::cos(k17x * x_Ph[0] + k17y * x_Ph[1] + k17z * x_Ph[2]) +
              amps(48,IMAG) * math::sin(k17x * x_Ph[0] + k17y * x_Ph[1] + k17z * x_Ph[2])) +              
             (amps(51,REAL) * math::cos(k18x * x_Ph[0] + k18y * x_Ph[1] + k18z * x_Ph[2]) +
              amps(51,IMAG) * math::sin(k18x * x_Ph[0] + k18y * x_Ph[1] + k18z * x_Ph[2])) + 
// ############## possible to comment out these modes  
             (k19z * amps(54,REAL) * math::cos(k19x * x_Ph[0] + k19y * x_Ph[1] + k19z * x_Ph[2]) +
              k19z * amps(54,IMAG) * math::sin(k19x * x_Ph[0] + k19y * x_Ph[1] + k19z * x_Ph[2])) +              
             (k19x * amps(56,REAL) * math::cos(k19x * x_Ph[0] + k19y * x_Ph[1] + k19z * x_Ph[2]) +
              k19x * amps(56,IMAG) * math::sin(k19x * x_Ph[0] + k19y * x_Ph[1] + k19z * x_Ph[2])) +   
             (k20z * amps(57,REAL) * math::cos(k20x * x_Ph[0] + k20y * x_Ph[1] + k20z * x_Ph[2]) +
              k20z * amps(57,IMAG) * math::sin(k20x * x_Ph[0] + k20y * x_Ph[1] + k20z * x_Ph[2])) +              
             (k20x * amps(59,REAL) * math::cos(k20x * x_Ph[0] + k20y * x_Ph[1] + k20z * x_Ph[2]) +
              k20x * amps(59,IMAG) * math::sin(k20x * x_Ph[0] + k20y * x_Ph[1] + k20z * x_Ph[2])) +   
             (k21z * amps(60,REAL) * math::cos(k21x * x_Ph[0] + k21y * x_Ph[1] + k21z * x_Ph[2]) +
              k21z * amps(60,IMAG) * math::sin(k21x * x_Ph[0] + k21y * x_Ph[1] + k21z * x_Ph[2])) +              
             (k21x * amps(62,REAL) * math::cos(k21x * x_Ph[0] + k21y * x_Ph[1] + k21z * x_Ph[2]) +
              k21x * amps(62,IMAG) * math::sin(k21x * x_Ph[0] + k21y * x_Ph[1] + k21z * x_Ph[2])) +   
             (k22z * amps(63,REAL) * math::cos(k22x * x_Ph[0] + k22y * x_Ph[1] + k22z * x_Ph[2]) +
              k22z * amps(63,IMAG) * math::sin(k22x * x_Ph[0] + k22y * x_Ph[1] + k22z * x_Ph[2])) +              
             (k22x * amps(65,REAL) * math::cos(k22x * x_Ph[0] + k22y * x_Ph[1] + k22z * x_Ph[2]) +
              k22x * amps(65,IMAG) * math::sin(k22x * x_Ph[0] + k22y * x_Ph[1] + k22z * x_Ph[2])) +   
             (k23z * amps(66,REAL) * math::cos(k23x * x_Ph[0] + k23y * x_Ph[1] + k23z * x_Ph[2]) +
              k23z * amps(66,IMAG) * math::sin(k23x * x_Ph[0] + k23y * x_Ph[1] + k23z * x_Ph[2])) +              
             (k23x * amps(68,REAL) * math::cos(k23x * x_Ph[0] + k23y * x_Ph[1] + k23z * x_Ph[2]) +
              k23x * amps(68,IMAG) * math::sin(k23x * x_Ph[0] + k23y * x_Ph[1] + k23z * x_Ph[2])) +   
             (k24z * amps(69,REAL) * math::cos(k24x * x_Ph[0] + k24y * x_Ph[1] + k24z * x_Ph[2]) +
              k24z * amps(69,IMAG) * math::sin(k24x * x_Ph[0] + k24y * x_Ph[1] + k24z * x_Ph[2])) +              
             (k24x * amps(71,REAL) * math::cos(k24x * x_Ph[0] + k24y * x_Ph[1] + k24z * x_Ph[2]) +
              k24x * amps(71,IMAG) * math::sin(k24x * x_Ph[0] + k24y * x_Ph[1] + k24z * x_Ph[2])) +   
             (k25z * amps(72,REAL) * math::cos(k25x * x_Ph[0] + k25y * x_Ph[1] + k25z * x_Ph[2]) +
              k25z * amps(72,IMAG) * math::sin(k25x * x_Ph[0] + k25y * x_Ph[1] + k25z * x_Ph[2])) +              
             (k25x * amps(74,REAL) * math::cos(k25x * x_Ph[0] + k25y * x_Ph[1] + k25z * x_Ph[2]) +
              k25x * amps(74,IMAG) * math::sin(k25x * x_Ph[0] + k25y * x_Ph[1] + k25z * x_Ph[2])) +   
             (k26z * amps(75,REAL) * math::cos(k26x * x_Ph[0] + k26y * x_Ph[1] + k26z * x_Ph[2]) +
              k26z * amps(75,IMAG) * math::sin(k26x * x_Ph[0] + k26y * x_Ph[1] + k26z * x_Ph[2])) +              
             (k26x * amps(77,REAL) * math::cos(k26x * x_Ph[0] + k26y * x_Ph[1] + k26z * x_Ph[2]) +
              k26x * amps(77,IMAG) * math::sin(k26x * x_Ph[0] + k26y * x_Ph[1] + k26z * x_Ph[2])) +   
// ##############
             (k27x * amps(78,REAL) * math::cos(k27x * x_Ph[0] + k27y * x_Ph[1] + k27z * x_Ph[2]) +
              k27x * amps(78,IMAG) * math::sin(k27x * x_Ph[0] + k27y * x_Ph[1] + k27z * x_Ph[2])) +
             (k28x * amps(81,REAL) * math::cos(k28x * x_Ph[0] + k28y * x_Ph[1] + k28z * x_Ph[2]) +
              k28x * amps(81,IMAG) * math::sin(k28x * x_Ph[0] + k28y * x_Ph[1] + k28z * x_Ph[2])) +
             (amps(84,REAL) * math::cos(k29x * x_Ph[0] + k29y * x_Ph[1] + k29z * x_Ph[2]) +
              amps(84,IMAG) * math::sin(k29x * x_Ph[0] + k29y * x_Ph[1] + k29z * x_Ph[2])) +
             (amps(87,REAL) * math::cos(k30x * x_Ph[0] + k30y * x_Ph[1] + k30z * x_Ph[2]) +
              amps(87,IMAG) * math::sin(k30x * x_Ph[0] + k30y * x_Ph[1] + k30z * x_Ph[2])) +
             (amps(90,REAL) * math::cos(k31x * x_Ph[0] + k31y * x_Ph[1] + k31z * x_Ph[2]) +
              amps(90,IMAG) * math::sin(k31x * x_Ph[0] + k31y * x_Ph[1] + k31z * x_Ph[2])) +
             (amps(93,REAL) * math::cos(k32x * x_Ph[0] + k32y * x_Ph[1] + k32z * x_Ph[2]) +
              amps(93,IMAG) * math::sin(k32x * x_Ph[0] + k32y * x_Ph[1] + k32z * x_Ph[2])) ;
    }

    Inline auto fx2(const unsigned short&,
                    const real_t&,
                    const coord_t<D>& x_Ph) const -> real_t {
      // return ZERO;
      return (amps(1,REAL) * math::cos(k01x * x_Ph[0] + k01y * x_Ph[1] + k01z * x_Ph[2]) +
              amps(1,IMAG) * math::sin(k01x * x_Ph[0] + k01y * x_Ph[1] + k01z * x_Ph[2])) +
             (amps(4,REAL) * math::cos(k02x * x_Ph[0] + k02y * x_Ph[1] + k02z * x_Ph[2]) +
              amps(4,IMAG) * math::sin(k02x * x_Ph[0] + k02y * x_Ph[1] + k02z * x_Ph[2])) +
             (k03y * amps(7,REAL) * math::cos(k03x * x_Ph[0] + k03y * x_Ph[1] + k03z * x_Ph[2]) +
              k03y * amps(7,IMAG) * math::sin(k03x * x_Ph[0] + k03y * x_Ph[1] + k03z * x_Ph[2])) +
             (k04y * amps(10,REAL) * math::cos(k04x * x_Ph[0] + k04y * x_Ph[1] + k04z * x_Ph[2]) +
              k04y * amps(10,IMAG) * math::sin(k04x * x_Ph[0] + k04y * x_Ph[1] + k04z * x_Ph[2])) +
             (amps(13,REAL) * math::cos(k05x * x_Ph[0] + k05y * x_Ph[1] + k05z * x_Ph[2]) +
              amps(13,IMAG) * math::sin(k05x * x_Ph[0] + k05y * x_Ph[1] + k05z * x_Ph[2])) +
             (amps(16,REAL) * math::cos(k06x * x_Ph[0] + k06y * x_Ph[1] + k06z * x_Ph[2]) +
              amps(16,IMAG) * math::sin(k06x * x_Ph[0] + k06y * x_Ph[1] + k06z * x_Ph[2])) +
            (-k07x * amps(18,REAL) * math::cos(k07x * x_Ph[0] + k07y * x_Ph[1] + k07z * x_Ph[2]) +
             -k07x * amps(18,IMAG) * math::sin(k07x * x_Ph[0] + k07y * x_Ph[1] + k07z * x_Ph[2])) +   
             (k07y * amps(20,REAL) * math::cos(k07x * x_Ph[0] + k07y * x_Ph[1] + k07z * x_Ph[2]) +
              k07y * amps(20,IMAG) * math::sin(k07x * x_Ph[0] + k07y * x_Ph[1] + k07z * x_Ph[2])) +   
            (-k08x * amps(21,REAL) * math::cos(k08x * x_Ph[0] + k08y * x_Ph[1] + k08z * x_Ph[2]) +
             -k08x * amps(21,IMAG) * math::sin(k08x * x_Ph[0] + k08y * x_Ph[1] + k08z * x_Ph[2])) +   
             (k08y * amps(23,REAL) * math::cos(k08x * x_Ph[0] + k08y * x_Ph[1] + k08z * x_Ph[2]) +
              k08y * amps(23,IMAG) * math::sin(k08x * x_Ph[0] + k08y * x_Ph[1] + k08z * x_Ph[2])) +   
            (-k09x * amps(24,REAL) * math::cos(k09x * x_Ph[0] + k09y * x_Ph[1] + k09z * x_Ph[2]) +
             -k09x * amps(24,IMAG) * math::sin(k09x * x_Ph[0] + k09y * x_Ph[1] + k09z * x_Ph[2])) +   
             (k09y * amps(26,REAL) * math::cos(k09x * x_Ph[0] + k09y * x_Ph[1] + k09z * x_Ph[2]) +
              k09y * amps(26,IMAG) * math::sin(k09x * x_Ph[0] + k09y * x_Ph[1] + k09z * x_Ph[2])) +   
            (-k10x * amps(27,REAL) * math::cos(k10x * x_Ph[0] + k10y * x_Ph[1] + k10z * x_Ph[2]) +
             -k10x * amps(27,IMAG) * math::sin(k10x * x_Ph[0] + k10y * x_Ph[1] + k10z * x_Ph[2])) +   
             (k10y * amps(29,REAL) * math::cos(k10x * x_Ph[0] + k10y * x_Ph[1] + k10z * x_Ph[2]) +
              k10y * amps(29,IMAG) * math::sin(k10x * x_Ph[0] + k10y * x_Ph[1] + k10z * x_Ph[2])) +   
             (amps(31,REAL) * math::cos(k11x * x_Ph[0] + k11y * x_Ph[1] + k11z * x_Ph[2]) +
              amps(31,IMAG) * math::sin(k11x * x_Ph[0] + k11y * x_Ph[1] + k11z * x_Ph[2])) +              
             (amps(34,REAL) * math::cos(k12x * x_Ph[0] + k12y * x_Ph[1] + k12z * x_Ph[2]) +
              amps(34,IMAG) * math::sin(k12x * x_Ph[0] + k12y * x_Ph[1] + k12z * x_Ph[2])) +              
             (amps(37,REAL) * math::cos(k13x * x_Ph[0] + k13y * x_Ph[1] + k13z * x_Ph[2]) +
              amps(37,IMAG) * math::sin(k13x * x_Ph[0] + k13y * x_Ph[1] + k13z * x_Ph[2])) +              
             (amps(40,REAL) * math::cos(k14x * x_Ph[0] + k14y * x_Ph[1] + k14z * x_Ph[2]) +
              amps(40,IMAG) * math::sin(k14x * x_Ph[0] + k14y * x_Ph[1] + k14z * x_Ph[2])) +              
             (k15z * amps(43,REAL) * math::cos(k15x * x_Ph[0] + k15y * x_Ph[1] + k15z * x_Ph[2]) +
              k15z * amps(43,IMAG) * math::sin(k15x * x_Ph[0] + k15y * x_Ph[1] + k15z * x_Ph[2])) +   
             (k15y * amps(44,REAL) * math::cos(k15x * x_Ph[0] + k15y * x_Ph[1] + k15z * x_Ph[2]) +
              k15y * amps(44,IMAG) * math::sin(k15x * x_Ph[0] + k15y * x_Ph[1] + k15z * x_Ph[2])) +   
             (k16z * amps(46,REAL) * math::cos(k16x * x_Ph[0] + k16y * x_Ph[1] + k16z * x_Ph[2]) +
              k16z * amps(46,IMAG) * math::sin(k16x * x_Ph[0] + k16y * x_Ph[1] + k16z * x_Ph[2])) +   
             (k16y * amps(47,REAL) * math::cos(k16x * x_Ph[0] + k16y * x_Ph[1] + k16z * x_Ph[2]) +
              k16y * amps(47,IMAG) * math::sin(k16x * x_Ph[0] + k16y * x_Ph[1] + k16z * x_Ph[2])) +   
             (k17z * amps(49,REAL) * math::cos(k17x * x_Ph[0] + k17y * x_Ph[1] + k17z * x_Ph[2]) +
              k17z * amps(49,IMAG) * math::sin(k17x * x_Ph[0] + k17y * x_Ph[1] + k17z * x_Ph[2])) +   
             (k17y * amps(50,REAL) * math::cos(k17x * x_Ph[0] + k17y * x_Ph[1] + k17z * x_Ph[2]) +
              k17y * amps(50,IMAG) * math::sin(k17x * x_Ph[0] + k17y * x_Ph[1] + k17z * x_Ph[2])) +   
             (k18z * amps(52,REAL) * math::cos(k18x * x_Ph[0] + k18y * x_Ph[1] + k18z * x_Ph[2]) +
              k18z * amps(52,IMAG) * math::sin(k18x * x_Ph[0] + k18y * x_Ph[1] + k18z * x_Ph[2])) +   
             (k18y * amps(53,REAL) * math::cos(k18x * x_Ph[0] + k18y * x_Ph[1] + k18z * x_Ph[2]) +
              k18y * amps(53,IMAG) * math::sin(k18x * x_Ph[0] + k18y * x_Ph[1] + k18z * x_Ph[2])) + 
// ############## possible to comment out these modes  
             (k19z * amps(55,REAL) * math::cos(k19x * x_Ph[0] + k19y * x_Ph[1] + k19z * x_Ph[2]) +
              k19z * amps(55,IMAG) * math::sin(k19x * x_Ph[0] + k19y * x_Ph[1] + k19z * x_Ph[2])) +              
             (k19y * amps(56,REAL) * math::cos(k19x * x_Ph[0] + k19y * x_Ph[1] + k19z * x_Ph[2]) +
              k19y * amps(56,IMAG) * math::sin(k19x * x_Ph[0] + k19y * x_Ph[1] + k19z * x_Ph[2])) +   
             (k20z * amps(58,REAL) * math::cos(k20x * x_Ph[0] + k20y * x_Ph[1] + k20z * x_Ph[2]) +
              k20z * amps(58,IMAG) * math::sin(k20x * x_Ph[0] + k20y * x_Ph[1] + k20z * x_Ph[2])) +              
             (k20y * amps(59,REAL) * math::cos(k20x * x_Ph[0] + k20y * x_Ph[1] + k20z * x_Ph[2]) +
              k20y * amps(59,IMAG) * math::sin(k20x * x_Ph[0] + k20y * x_Ph[1] + k20z * x_Ph[2])) +   
             (k21z * amps(61,REAL) * math::cos(k21x * x_Ph[0] + k21y * x_Ph[1] + k21z * x_Ph[2]) +
              k21z * amps(61,IMAG) * math::sin(k21x * x_Ph[0] + k21y * x_Ph[1] + k21z * x_Ph[2])) +              
             (k21y * amps(62,REAL) * math::cos(k21x * x_Ph[0] + k21y * x_Ph[1] + k21z * x_Ph[2]) +
              k21y * amps(62,IMAG) * math::sin(k21x * x_Ph[0] + k21y * x_Ph[1] + k21z * x_Ph[2])) +   
             (k22z * amps(64,REAL) * math::cos(k22x * x_Ph[0] + k22y * x_Ph[1] + k22z * x_Ph[2]) +
              k22z * amps(64,IMAG) * math::sin(k22x * x_Ph[0] + k22y * x_Ph[1] + k22z * x_Ph[2])) +              
             (k22y * amps(65,REAL) * math::cos(k22x * x_Ph[0] + k22y * x_Ph[1] + k22z * x_Ph[2]) +
              k22y * amps(65,IMAG) * math::sin(k22x * x_Ph[0] + k22y * x_Ph[1] + k22z * x_Ph[2])) +   
             (k23z * amps(67,REAL) * math::cos(k23x * x_Ph[0] + k23y * x_Ph[1] + k23z * x_Ph[2]) +
              k23z * amps(67,IMAG) * math::sin(k23x * x_Ph[0] + k23y * x_Ph[1] + k23z * x_Ph[2])) +              
             (k23y * amps(68,REAL) * math::cos(k23x * x_Ph[0] + k23y * x_Ph[1] + k23z * x_Ph[2]) +
              k23y * amps(68,IMAG) * math::sin(k23x * x_Ph[0] + k23y * x_Ph[1] + k23z * x_Ph[2])) +   
             (k24z * amps(69,REAL) * math::cos(k24x * x_Ph[0] + k24y * x_Ph[1] + k24z * x_Ph[2]) +
              k24z * amps(69,IMAG) * math::sin(k24x * x_Ph[0] + k24y * x_Ph[1] + k24z * x_Ph[2])) +              
             (k24y * amps(71,REAL) * math::cos(k24x * x_Ph[0] + k24y * x_Ph[1] + k24z * x_Ph[2]) +
              k24y * amps(71,IMAG) * math::sin(k24x * x_Ph[0] + k24y * x_Ph[1] + k24z * x_Ph[2])) +   
             (k25z * amps(73,REAL) * math::cos(k25x * x_Ph[0] + k25y * x_Ph[1] + k25z * x_Ph[2]) +
              k25z * amps(73,IMAG) * math::sin(k25x * x_Ph[0] + k25y * x_Ph[1] + k25z * x_Ph[2])) +              
             (k25y * amps(74,REAL) * math::cos(k25x * x_Ph[0] + k25y * x_Ph[1] + k25z * x_Ph[2]) +
              k25y * amps(74,IMAG) * math::sin(k25x * x_Ph[0] + k25y * x_Ph[1] + k25z * x_Ph[2])) +   
             (k26z * amps(76,REAL) * math::cos(k26x * x_Ph[0] + k26y * x_Ph[1] + k26z * x_Ph[2]) +
              k26z * amps(76,IMAG) * math::sin(k26x * x_Ph[0] + k26y * x_Ph[1] + k26z * x_Ph[2])) +              
             (k26y * amps(77,REAL) * math::cos(k26x * x_Ph[0] + k26y * x_Ph[1] + k26z * x_Ph[2]) +
              k26y * amps(77,IMAG) * math::sin(k26x * x_Ph[0] + k26y * x_Ph[1] + k26z * x_Ph[2])) +   
// ############## 
             (amps(79,REAL) * math::cos(k27x * x_Ph[0] + k27y * x_Ph[1] + k27z * x_Ph[2]) +
              amps(79,IMAG) * math::sin(k27x * x_Ph[0] + k27y * x_Ph[1] + k27z * x_Ph[2])) +
             (amps(82,REAL) * math::cos(k28x * x_Ph[0] + k28y * x_Ph[1] + k28z * x_Ph[2]) +
              amps(82,IMAG) * math::sin(k28x * x_Ph[0] + k28y * x_Ph[1] + k28z * x_Ph[2])) +
             (k29y * amps(85,REAL) * math::cos(k29x * x_Ph[0] + k29y * x_Ph[1] + k29z * x_Ph[2]) +
              k29y * amps(85,IMAG) * math::sin(k29x * x_Ph[0] + k29y * x_Ph[1] + k29z * x_Ph[2])) +
             (k30y * amps(88,REAL) * math::cos(k30x * x_Ph[0] + k30y * x_Ph[1] + k30z * x_Ph[2]) +
              k30y * amps(88,IMAG) * math::sin(k30x * x_Ph[0] + k30y * x_Ph[1] + k30z * x_Ph[2])) +
             (amps(91,REAL) * math::cos(k31x * x_Ph[0] + k31y * x_Ph[1] + k31z * x_Ph[2]) +
              amps(91,IMAG) * math::sin(k31x * x_Ph[0] + k31y * x_Ph[1] + k31z * x_Ph[2])) +
             (amps(94,REAL) * math::cos(k32x * x_Ph[0] + k32y * x_Ph[1] + k32z * x_Ph[2]) +
              amps(94,IMAG) * math::sin(k32x * x_Ph[0] + k32y * x_Ph[1] + k32z * x_Ph[2])) ;
    }

    Inline auto fx3(const unsigned short&,
                    const real_t&,
                    const coord_t<D>& x_Ph) const -> real_t {

      // return ZERO;
      return (amps(2,REAL) * math::cos(k01x * x_Ph[0] + k01y * x_Ph[1] + k01z * x_Ph[2]) +
              amps(2,IMAG) * math::sin(k01x * x_Ph[0] + k01y * x_Ph[1] + k01z * x_Ph[2])) + 
             (amps(5,REAL) * math::cos(k02x * x_Ph[0] + k02y * x_Ph[1] + k02z * x_Ph[2]) +
              amps(5,IMAG) * math::sin(k02x * x_Ph[0] + k02y * x_Ph[1] + k02z * x_Ph[2])) +
             (amps(8,REAL) * math::cos(k03x * x_Ph[0] + k03y * x_Ph[1] + k03z * x_Ph[2]) +
              amps(8,IMAG) * math::sin(k03x * x_Ph[0] + k03y * x_Ph[1] + k03z * x_Ph[2])) +
             (amps(11,REAL) * math::cos(k04x * x_Ph[0] + k04y * x_Ph[1] + k04z * x_Ph[2]) +
              amps(11,IMAG) * math::sin(k04x * x_Ph[0] + k04y * x_Ph[1] + k04z * x_Ph[2])) +
             (k05z * amps(14,REAL) * math::cos(k05x * x_Ph[0] + k05y * x_Ph[1] + k05z * x_Ph[2]) +
              k05z * amps(14,IMAG) * math::sin(k05x * x_Ph[0] + k05y * x_Ph[1] + k05z * x_Ph[2])) +
             (k06z * amps(17,REAL) * math::cos(k06x * x_Ph[0] + k06y * x_Ph[1] + k06z * x_Ph[2]) +
              k06z * amps(17,IMAG) * math::sin(k06x * x_Ph[0] + k06y * x_Ph[1] + k06z * x_Ph[2])) +
             (amps(19,REAL) * math::cos(k07x * x_Ph[0] + k07y * x_Ph[1] + k07z * x_Ph[2]) +
              amps(19,IMAG) * math::sin(k07x * x_Ph[0] + k07y * x_Ph[1] + k07z * x_Ph[2])) +   
             (amps(22,REAL) * math::cos(k08x * x_Ph[0] + k08y * x_Ph[1] + k08z * x_Ph[2]) +
              amps(22,IMAG) * math::sin(k08x * x_Ph[0] + k08y * x_Ph[1] + k08z * x_Ph[2])) +   
             (amps(25,REAL) * math::cos(k09x * x_Ph[0] + k09y * x_Ph[1] + k09z * x_Ph[2]) +
              amps(25,IMAG) * math::sin(k09x * x_Ph[0] + k09y * x_Ph[1] + k09z * x_Ph[2])) +   
             (amps(28,REAL) * math::cos(k10x * x_Ph[0] + k10y * x_Ph[1] + k10z * x_Ph[2]) +
              amps(28,IMAG) * math::sin(k10x * x_Ph[0] + k10y * x_Ph[1] + k10z * x_Ph[2])) +   
            (-k11x * amps(30,REAL) * math::cos(k11x * x_Ph[0] + k11y * x_Ph[1] + k11z * x_Ph[2]) +
             -k11x * amps(30,IMAG) * math::sin(k11x * x_Ph[0] + k11y * x_Ph[1] + k11z * x_Ph[2])) +              
             (k11z * amps(32,REAL) * math::cos(k11x * x_Ph[0] + k11y * x_Ph[1] + k11z * x_Ph[2]) +
              k11z * amps(32,IMAG) * math::sin(k11x * x_Ph[0] + k11y * x_Ph[1] + k11z * x_Ph[2])) +  
            (-k12x * amps(33,REAL) * math::cos(k12x * x_Ph[0] + k12y * x_Ph[1] + k12z * x_Ph[2]） +
             -k12x * amps(33,IMAG) * math::sin(k12x * x_Ph[0] + k12y * x_Ph[1] + k12z * x_Ph[2])) +              
             (k12z * amps(35,REAL) * math::cos(k12x * x_Ph[0] + k12y * x_Ph[1] + k12z * x_Ph[2]) +
              k12z * amps(35,IMAG) * math::sin(k12x * x_Ph[0] + k12y * x_Ph[1] + k12z * x_Ph[2])) +        
            (-k13x * amps(36,REAL) * math::cos(k13x * x_Ph[0] + k13y * x_Ph[1] + k13z * x_Ph[2]) +
             -k13x * amps(36,IMAG) * math::sin(k13x * x_Ph[0] + k13y * x_Ph[1] + k13z * x_Ph[2])) +              
             (k13z * amps(38,REAL) * math::cos(k13x * x_Ph[0] + k13y * x_Ph[1] + k13z * x_Ph[2]) +
              k13z * amps(38,IMAG) * math::sin(k13x * x_Ph[0] + k13y * x_Ph[1] + k13z * x_Ph[2])) +       
            (-k14x * amps(39,REAL) * math::cos(k14x * x_Ph[0] + k14y * x_Ph[1] + k14z * x_Ph[2]) +
             -k14z * amps(39,IMAG) * math::sin(k14x * x_Ph[0] + k14y * x_Ph[1] + k14z * x_Ph[2])) +              
             (k14z * amps(41,REAL) * math::cos(k14x * x_Ph[0] + k14y * x_Ph[1] + k14z * x_Ph[2]) +
              k14z * amps(41,IMAG) * math::sin(k14x * x_Ph[0] + k14y * x_Ph[1] + k14z * x_Ph[2])) +  
            (-k15y * amps(43,REAL) * math::cos(k15x * x_Ph[0] + k15y * x_Ph[1] + k15z * x_Ph[2]) +
             -k15y * amps(43,IMAG) * math::sin(k15x * x_Ph[0] + k15y * x_Ph[1] + k15z * x_Ph[2])) +              
             (k15z * amps(44,REAL) * math::cos(k15x * x_Ph[0] + k15y * x_Ph[1] + k15z * x_Ph[2]) +
              k15z * amps(44,IMAG) * math::sin(k15x * x_Ph[0] + k15y * x_Ph[1] + k15z * x_Ph[2])) +  
            (-k16y * amps(46,REAL) * math::cos(k16x * x_Ph[0] + k16y * x_Ph[1] + k16z * x_Ph[2]) +
             -k16y * amps(46,IMAG) * math::sin(k16x * x_Ph[0] + k16y * x_Ph[1] + k16z * x_Ph[2])) +              
             (k16z * amps(47,REAL) * math::cos(k16x * x_Ph[0] + k16y * x_Ph[1] + k16z * x_Ph[2]) +
              k16z * amps(47,IMAG) * math::sin(k16x * x_Ph[0] + k16y * x_Ph[1] + k16z * x_Ph[2])) +  
            (-k17y * amps(49,REAL) * math::cos(k17x * x_Ph[0] + k17y * x_Ph[1] + k17z * x_Ph[2]) +
             -k17y * amps(49,IMAG) * math::sin(k17x * x_Ph[0] + k17y * x_Ph[1] + k17z * x_Ph[2])) +              
             (k17z * amps(50,REAL) * math::cos(k17x * x_Ph[0] + k17y * x_Ph[1] + k17z * x_Ph[2]) +
              k17z * amps(50,IMAG) * math::sin(k17x * x_Ph[0] + k17y * x_Ph[1] + k17z * x_Ph[2])) +  
            (-k18y * amps(52,REAL) * math::cos(k18x * x_Ph[0] + k18y * x_Ph[1] + k18z * x_Ph[2]) +
             -k18y * amps(52,IMAG) * math::sin(k18x * x_Ph[0] + k18y * x_Ph[1] + k18z * x_Ph[2])) +              
             (k18z * amps(53,REAL) * math::cos(k18x * x_Ph[0] + k18y * x_Ph[1] + k18z * x_Ph[2]) +
              k18z * amps(53,IMAG) * math::sin(k18x * x_Ph[0] + k18y * x_Ph[1] + k18z * x_Ph[2])) +  
// ############## possible to comment out these modes
           ((-k19x * amps(54,REAL) - k19y * amps(55,REAL)) * math::cos(k19x * x_Ph[0] + k19y * x_Ph[1] + k19z * x_Ph[2]) +
            (-k19x * amps(54,REAL) - k19y * amps(55,REAL)) * math::sin(k19x * x_Ph[0] + k19y * x_Ph[1] + k19z * x_Ph[2])) +              
             (k19z * amps(56,REAL) * math::cos(k19x * x_Ph[0] + k19y * x_Ph[1] + k19z * x_Ph[2]) +
              k19z * amps(56,IMAG) * math::sin(k19x * x_Ph[0] + k19y * x_Ph[1] + k19z * x_Ph[2])) +   
           ((-k19x * amps(57,REAL) - k19y * amps(58,REAL)) * math::cos(k20x * x_Ph[0] + k20y * x_Ph[1] + k20z * x_Ph[2]) +
            (-k19x * amps(57,REAL) - k19y * amps(58,REAL)) * math::sin(k20x * x_Ph[0] + k20y * x_Ph[1] + k20z * x_Ph[2])) +              
             (k20z * amps(59,REAL) * math::cos(k20x * x_Ph[0] + k20y * x_Ph[1] + k20z * x_Ph[2]) +
              k20z * amps(59,IMAG) * math::sin(k20x * x_Ph[0] + k20y * x_Ph[1] + k20z * x_Ph[2])) +   
           ((-k21x * amps(60,REAL) - k19y * amps(61,REAL)) * math::cos(k21x * x_Ph[0] + k21y * x_Ph[1] + k21z * x_Ph[2]) +
            (-k21x * amps(60,REAL) - k19y * amps(61,REAL)) * math::sin(k21x * x_Ph[0] + k21y * x_Ph[1] + k21z * x_Ph[2])) +              
             (k21z * amps(62,REAL) * math::cos(k21x * x_Ph[0] + k21y * x_Ph[1] + k21z * x_Ph[2]) +
              k21z * amps(62,IMAG) * math::sin(k21x * x_Ph[0] + k21y * x_Ph[1] + k21z * x_Ph[2])) +   
           ((-k22x * amps(63,REAL) - k22y * amps(64,REAL)) * math::cos(k22x * x_Ph[0] + k22y * x_Ph[1] + k22z * x_Ph[2]) +
            (-k22x * amps(63,REAL) - k22y * amps(64,REAL)) * math::sin(k22x * x_Ph[0] + k22y * x_Ph[1] + k22z * x_Ph[2])) +              
             (k22z * amps(65,REAL) * math::cos(k22x * x_Ph[0] + k22y * x_Ph[1] + k22z * x_Ph[2]) +
              k22z * amps(65,IMAG) * math::sin(k22x * x_Ph[0] + k22y * x_Ph[1] + k22z * x_Ph[2])) +   
           ((-k23x * amps(66,REAL) - k19y * amps(67,REAL)) * math::cos(k23x * x_Ph[0] + k23y * x_Ph[1] + k23z * x_Ph[2]) +
            (-k23x * amps(66,REAL) - k19y * amps(67,REAL)) * math::sin(k23x * x_Ph[0] + k23y * x_Ph[1] + k23z * x_Ph[2])) +              
             (k23z * amps(68,REAL) * math::cos(k23x * x_Ph[0] + k23y * x_Ph[1] + k23z * x_Ph[2]) +
              k23z * amps(68,IMAG) * math::sin(k23x * x_Ph[0] + k23y * x_Ph[1] + k23z * x_Ph[2])) +   
           ((-k24x * amps(69,REAL) - k24y * amps(70,REAL)) * math::cos(k24x * x_Ph[0] + k24y * x_Ph[1] + k24z * x_Ph[2]) +
            (-k24x * amps(69,REAL) - k24y * amps(70,REAL)) * math::sin(k24x * x_Ph[0] + k24y * x_Ph[1] + k24z * x_Ph[2])) +              
             (k24z * amps(71,REAL) * math::cos(k24x * x_Ph[0] + k24y * x_Ph[1] + k24z * x_Ph[2]) +
              k24z * amps(71,IMAG) * math::sin(k24x * x_Ph[0] + k24y * x_Ph[1] + k24z * x_Ph[2])) +   
           ((-k25x * amps(72,REAL) - k25y * amps(73,REAL)) * math::cos(k25x * x_Ph[0] + k25y * x_Ph[1] + k25z * x_Ph[2]) +
            (-k25x * amps(72,REAL) - k25y * amps(73,REAL)) * math::sin(k25x * x_Ph[0] + k25y * x_Ph[1] + k25z * x_Ph[2])) +              
             (k25z * amps(74,REAL) * math::cos(k25x * x_Ph[0] + k25y * x_Ph[1] + k25z * x_Ph[2]) +
              k25z * amps(74,IMAG) * math::sin(k25x * x_Ph[0] + k25y * x_Ph[1] + k25z * x_Ph[2])) +   
           ((-k26x * amps(75,REAL) - k26y * amps(76,REAL)) * math::cos(k26x * x_Ph[0] + k26y * x_Ph[1] + k26z * x_Ph[2]) +
            (-k26x * amps(75,REAL) - k26y * amps(76,REAL)) * math::sin(k26x * x_Ph[0] + k26y * x_Ph[1] + k26z * x_Ph[2])) +              
             (k26z * amps(77,REAL) * math::cos(k26x * x_Ph[0] + k26y * x_Ph[1] + k26z * x_Ph[2]) +
              k26z * amps(77,IMAG) * math::sin(k26x * x_Ph[0] + k26y * x_Ph[1] + k26z * x_Ph[2])) +   
// ############## 
             (amps(80,REAL) * math::cos(k27x * x_Ph[0] + k27y * x_Ph[1] + k27z * x_Ph[2]) +
              amps(80,IMAG) * math::sin(k27x * x_Ph[0] + k27y * x_Ph[1] + k27z * x_Ph[2])) + 
             (amps(83,REAL) * math::cos(k28x * x_Ph[0] + k28y * x_Ph[1] + k28z * x_Ph[2]) +
              amps(83,IMAG) * math::sin(k28x * x_Ph[0] + k28y * x_Ph[1] + k28z * x_Ph[2])) +
             (amps(86,REAL) * math::cos(k29x * x_Ph[0] + k29y * x_Ph[1] + k29z * x_Ph[2]) +
              amps(86,IMAG) * math::sin(k29x * x_Ph[0] + k29y * x_Ph[1] + k29z * x_Ph[2])) +
             (amps(89,REAL) * math::cos(k30x * x_Ph[0] + k30y * x_Ph[1] + k30z * x_Ph[2]) +
              amps(89,IMAG) * math::sin(k30x * x_Ph[0] + k30y * x_Ph[1] + k30z * x_Ph[2])) +
             (k31z * amps(92,REAL) * math::cos(k31x * x_Ph[0] + k31y * x_Ph[1] + k31z * x_Ph[2]) +
              k31z * amps(92,IMAG) * math::sin(k31x * x_Ph[0] + k31y * x_Ph[1] + k31z * x_Ph[2])) +
             (k32z * amps(95,REAL) * math::cos(k32x * x_Ph[0] + k32y * x_Ph[1] + k32z * x_Ph[2]) +
              k32z * amps(95,IMAG) * math::sin(k32x * x_Ph[0] + k32y * x_Ph[1] + k32z * x_Ph[2])) ;
    }

  public:

  private:
    array_t<real_t* [2]> amps;
    const real_t         sx1, sx2, sx3;
    const real_t         k01x, k01y, k01z;
    const real_t         k02x, k02y, k02z;
    const real_t         k03x, k03y, k03z;
    const real_t         k04x, k04y, k04z;
    const real_t         k05x, k05y, k05z;
    const real_t         k06x, k06y, k06z;
    const real_t         k07x, k07y, k07z;
    const real_t         k08x, k08y, k08z;
    const real_t         k09x, k09y, k09z;
    const real_t         k10x, k10y, k10z;
    const real_t         k11x, k11y, k11z;
    const real_t         k12x, k12y, k12z;
    const real_t         k13x, k13y, k13z;
    const real_t         k14x, k14y, k14z;
    const real_t         k15x, k15y, k15z;
    const real_t         k16x, k16y, k16z;
    const real_t         k17x, k17y, k17z;
    const real_t         k18x, k18y, k18z;
    const real_t         k19x, k19y, k19z;
    const real_t         k20x, k20y, k20z;
    const real_t         k21x, k21y, k21z;
    const real_t         k22x, k22y, k22z;
    const real_t         k23x, k23y, k23z;
    const real_t         k24x, k24y, k24z;
    const real_t         k25x, k25y, k25z;
    const real_t         k26x, k26y, k26z;
    const real_t         k27x, k27y, k27z;
    const real_t         k28x, k28y, k28z;
    const real_t         k29x, k29y, k29z;
    const real_t         k30x, k30y, k30z;
    const real_t         k31x, k31y, k31z;
    const real_t         k32x, k32y, k32z;
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
    array_t<real_t* [2]> amplitudes;
    array_t<real_t*> phi0, kmag;
    ExtForce<M::PrtlDim> ext_force;
    const real_t         dt;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& params, const Metadomain<S, M>& global_domain)
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
      , nmodes { params.template get<unsigned int>("setup.nmodes", 96) }
      , Bnorm { params.template get<real_t>("setup.Bnorm", 0.0) }
      , pl_gamma_min { params.template get<real_t>("setup.pl_gamma_min", 0.1) }
      , pl_gamma_max { params.template get<real_t>("setup.pl_gamma_max", 100.0) }
      , pl_index { params.template get<real_t>("setup.pl_index", -2.0) }  
      , amp0 { machno * temperature / static_cast<real_t>(nmodes) }
      , phi0 { "DrivingPhases", nmodes }
      , kmag { "DrivingKmag", nmodes }
      , amplitudes { "DrivingModes", nmodes }
      , ext_force { amplitudes, SX1, SX2, SX3 }
      , init_flds { Bnorm }
      , dt { params.template get<real_t>("algorithms.timestep.dt") } {
      // Initializing random phases
      auto phi0_ = Kokkos::create_mirror_view(phi0);
      srand (static_cast <unsigned> (12345));
      for (int i = 0; i < nmodes; ++i) {
        phi0_(i) = constant::TWO_PI * static_cast <real_t> (rand()) / static_cast <real_t> (RAND_MAX);
      }
      Kokkos::deep_copy(phi0, phi0_);
      // Initializing mode amplitudes
      auto kmag_ = Kokkos::create_mirror_view(kmag);
      auto THREE = ONE + TWO;
      kmag_(0) = ONE; kmag_(1) = ONE; kmag_(2) = ONE; kmag_(3) = ONE; kmag_(4) = ONE; kmag_(5) = ONE;    // k1,k2
      kmag_(6) = ONE; kmag_(7) = ONE; kmag_(8) = ONE; kmag_(9) = ONE; kmag_(10) = ONE; kmag_(11) = ONE;   // k3,k4
      kmag_(12) = ONE; kmag_(13) = ONE; kmag_(14) = ONE; kmag_(15) = ONE; kmag_(16) = ONE; kmag_(17) = ONE; // k5,k6 
      kmag_(18) = math::sqrt(TWO); kmag_(19) = math::sqrt(TWO); kmag_(20) = math::sqrt(TWO); kmag_(21) = math::sqrt(TWO); kmag_(22) = math::sqrt(TWO); kmag_(23) = math::sqrt(TWO);  // k7,k8
      kmag_(24) = math::sqrt(TWO); kmag_(25) = math::sqrt(TWO); kmag_(26) = math::sqrt(TWO); kmag_(27) = math::sqrt(TWO); kmag_(28) = math::sqrt(TWO); kmag_(29) = math::sqrt(TWO);  // k9,k10
      kmag_(30) = math::sqrt(TWO); kmag_(31) = math::sqrt(TWO); kmag_(32) = math::sqrt(TWO); kmag_(33) = math::sqrt(TWO); kmag_(34) = math::sqrt(TWO); kmag_(35) = math::sqrt(TWO);  // k11,k12
      kmag_(36) = math::sqrt(TWO); kmag_(37) = math::sqrt(TWO); kmag_(38) = math::sqrt(TWO); kmag_(39) = math::sqrt(TWO); kmag_(40) = math::sqrt(TWO); kmag_(41) = math::sqrt(TWO);  // k13,k14
      kmag_(42) = math::sqrt(TWO); kmag_(43) = math::sqrt(TWO); kmag_(44) = math::sqrt(TWO); kmag_(45) = math::sqrt(TWO); kmag_(46) = math::sqrt(TWO); kmag_(47) = math::sqrt(TWO);   // k15,k16
      kmag_(48) = math::sqrt(TWO); kmag_(49) = math::sqrt(TWO); kmag_(50) = math::sqrt(TWO); kmag_(51) = math::sqrt(TWO); kmag_(52) = math::sqrt(TWO); kmag_(53) = math::sqrt(TWO);   // k17,k18
      kmag_(54) = math::sqrt(THREE); kmag_(55) = math::sqrt(THREE); kmag_(56) = math::sqrt(THREE); kmag_(57) = math::sqrt(THREE); kmag_(58) = math::sqrt(THREE); kmag_(59) = math::sqrt(THREE);  // k19,k20
      kmag_(60) = math::sqrt(THREE); kmag_(61) = math::sqrt(THREE); kmag_(62) = math::sqrt(THREE); kmag_(63) = math::sqrt(THREE); kmag_(64) = math::sqrt(THREE); kmag_(65) = math::sqrt(THREE);  // k21,k22
      kmag_(66) = math::sqrt(THREE); kmag_(67) = math::sqrt(THREE); kmag_(68) = math::sqrt(THREE); kmag_(69) = math::sqrt(THREE); kmag_(70) = math::sqrt(THREE); kmag_(71) = math::sqrt(THREE);  // k23,k24
      kmag_(72) = math::sqrt(THREE); kmag_(73) = math::sqrt(THREE); kmag_(74) = math::sqrt(THREE); kmag_(75) = math::sqrt(THREE); kmag_(76) = math::sqrt(THREE); kmag_(77) = math::sqrt(THREE);   // k25,k26
      kmag_(78) = TWO; kmag_(79) = TWO; kmag_(80) = TWO; kmag_(81) = TWO; kmag_(82) = TWO; kmag_(83) = TWO;   // k27,k28
      kmag_(84) = TWO; kmag_(85) = TWO; kmag_(86) = TWO; kmag_(87) = TWO; kmag_(88) = TWO; kmag_(89) = TWO;  // k29,k30
      kmag_(90) = TWO; kmag_(91) = TWO; kmag_(92) = TWO; kmag_(93) = TWO; kmag_(94) = TWO; kmag_(95) = TWO; // k31,k32
      Kokkos::deep_copy(kmag, kmag_);
      // Initializing driving amplitudes
      Init();
    }

    void Init() {
      // initializing amplitudes
      auto       amplitudes_ = amplitudes;
      const auto amp0_       = amp0;
      const auto phi0_       = phi0;
      const auto kmag_       = kmag;
      Kokkos::parallel_for(
        "RandomAmplitudes",
        amplitudes.extent(0),
        Lambda(index_t i) {
          amplitudes_(i, REAL) = amp0_ * math::cos(phi0_(i)) * kmag_(i);
          amplitudes_(i, IMAG) = amp0_ * math::sin(phi0_(i)) * kmag_(i);
          printf("amplitudes_(%d, REAL) = %f\n", i, amplitudes_(i, REAL));
        });
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
      auto omega0 = 2.0*0.6 * math::sqrt(temperature * machno) * constant::TWO_PI / SX1;
      auto gamma0 = 2.0*0.5 * math::sqrt(temperature * machno) * constant::TWO_PI / SX2;
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
          amplitudes(i, REAL)  = (ampr_prev * math::cos(omega0 * kmag(i) * dt) +
                                 ampi_prev * math::sin(omega0 * kmag(i) * dt)) *
                                  math::exp(-gamma0 * kmag(i) * dt) +
                                unr * sigma0 * kmag(i) * dt;
          amplitudes(i, IMAG) = (-ampr_prev * math::sin(omega0 * kmag(i) * dt) +
                                 ampi_prev * math::cos(omega0 * kmag(i) * dt)) *
                                  math::exp(-gamma0 * kmag(i) * dt) +
                                uni * sigma0 * kmag(i) * dt;
        });

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