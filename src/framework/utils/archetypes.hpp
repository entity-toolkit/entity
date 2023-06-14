#ifndef FRAMEWORK_ARCHETYPES_H
#define FRAMEWORK_ARCHETYPES_H

#include "wrapper.h"

#include "sim_params.h"

#include "io/output.h"
#include "meshblock/meshblock.h"
#include "utils/qmath.h"

#ifdef NTTINY_ENABLED
#  include "nttiny/api.h"
#endif

#include <map>

namespace ntt {

  /* -------------------------------------------------------------------------- */
  /*                              Master pgen class                             */
  /* -------------------------------------------------------------------------- */
  template <Dimension D, SimulationEngine S>
  struct PGen {
    virtual inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) {}
    virtual inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) {}

    virtual inline void UserDriveFields(const real_t&,
                                        const SimulationParams&,
                                        Meshblock<D, S>&) {}
    virtual inline void UserDriveParticles(const real_t&,
                                           const SimulationParams&,
                                           Meshblock<D, S>&) {}

#ifdef NTTINY_ENABLED
    virtual inline void UserInitBuffers_nttiny(const SimulationParams&,
                                               const Meshblock<D, S>&,
                                               std::map<std::string, nttiny::ScrollingBuffer>&) {
    }
    virtual inline void UserSetBuffers_nttiny(const real_t&,
                                              const SimulationParams&,
                                              const Meshblock<D, S>&,
                                              std::map<std::string, nttiny::ScrollingBuffer>&) {
    }
#endif
  };

  /* -------------------------------------------------------------------------- */
  /*                             Target field class                             */
  /* -------------------------------------------------------------------------- */
  template <Dimension D, SimulationEngine S>
  struct TargetFields {
    TargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : m_params { params }, m_mblock { mblock } {}

    Inline virtual real_t operator()(const em&, const coord_t<D>&) const {
      return ZERO;
    }

  protected:
    SimulationParams m_params;
    Meshblock<D, S>  m_mblock;
  };

  /* -------------------------------------------------------------------------- */
  /*                             Energy distribution                            */
  /* -------------------------------------------------------------------------- */
  template <Dimension D, SimulationEngine S>
  struct EnergyDistribution {
    EnergyDistribution(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : m_params { params }, m_mblock { mblock } {}

    Inline virtual void operator()(const coord_t<D>&,
                                   vec_t<Dim3>& v,
                                   const int&   species = 0) const {
      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    }

  protected:
    SimulationParams m_params;
    Meshblock<D, S>  m_mblock;
  };

  template <Dimension D, SimulationEngine S>
  struct ColdDist : public EnergyDistribution<D, S> {
    ColdDist(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock) {}
    Inline void operator()(const coord_t<D>&,
                           vec_t<Dim3>& v,
                           const int&   species = 0) const override {
      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    }
  };

  template <Dimension D, SimulationEngine S>
  struct Maxwellian {
    Maxwellian(const Meshblock<D, S>& mblock) : pool { *(mblock.random_pool_ptr) } {}
    // Juttner-Synge distribution
    Inline void JS(vec_t<Dim3>& v, const real_t& temp) const {
      typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();
      real_t                                      u { ZERO }, eta { ZERO }, theta { ZERO };
      real_t                                      X1 { ZERO }, X2 { ZERO };
      if (temp < 0.1) {
        // Juttner-Synge distribution using the Box-Muller method - non-relativistic
        while (AlmostEqual(u, ZERO)) {
          u = rand_gen.frand();
        }
        eta = math::sqrt(-TWO * math::log(u));
        while (AlmostEqual(theta, ZERO)) {
          theta = constant::TWO_PI * rand_gen.frand();
        }
        u = eta * math::cos(theta) * math::sqrt(temp);
      } else {
        // Juttner-Synge distribution using the Sobol method - relativistic
        u = ONE;
        while (SQR(eta) <= SQR(u)) {
          while (AlmostEqual(X1, ZERO)) {
            X1 = rand_gen.frand() * rand_gen.frand() * rand_gen.frand();
          }
          u  = -temp * math::log(X1);
          X1 = rand_gen.frand();
          while (AlmostEqual(X1, 0)) {
            X1 = rand_gen.frand();
          }
          eta = u - temp * math::log(X1);
        }
      }
      X1   = rand_gen.frand();
      X2   = rand_gen.frand();
      v[0] = u * (TWO * X1 - ONE);
      v[2] = TWO * u * math::sqrt(X1 * (ONE - X1));
      v[1] = v[2] * math::cos(constant::TWO_PI * X2);
      v[2] = v[2] * math::sin(constant::TWO_PI * X2);
      pool.free_state(rand_gen);
    }
    // Boost a symmetric distribution to a relativistic speed using flipping method
    // https://arxiv.org/pdf/1504.03910.pdf
    Inline void boost(vec_t<Dim3>&  v,
                      const real_t& boost_vel,
                      const short&  boost_direction) const {
      typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();
      real_t boost_beta { boost_vel / math::sqrt(ONE + SQR(boost_vel)) };
      real_t boost_gamma { boost_vel / boost_beta };
      real_t ut { math::sqrt(ONE + SQR(v[0]) + SQR(v[1]) + SQR(v[2])) };
      if (-boost_beta * v[boost_direction] > ut * rand_gen.frand()) {
        v[boost_direction] = -v[boost_direction];
      }
      pool.free_state(rand_gen);
      v[boost_direction] = boost_gamma * (v[boost_direction] + boost_beta * ut);
    }

    Inline void operator()(vec_t<Dim3>&  v,
                           const real_t& temp,
                           const real_t& boost_vel       = 0.0,
                           const short&  boost_direction = 0) const {
      if (AlmostEqual(temp, ZERO)) {
        v[0] = ZERO;
        v[1] = ZERO;
        v[2] = ZERO;
      } else {
        JS(v, temp);
      }
      if (!AlmostEqual(boost_vel, ZERO)) {
        if (boost_direction < 0) {
          boost(v, -boost_vel, -boost_direction - 1);
        } else if (boost_direction > 0) {
          boost(v, boost_vel, boost_direction - 1);
        }
        // no boost when boost_direction == 0
      }
    }

  private:
    RandomNumberPool_t pool;
  };

  /* -------------------------------------------------------------------------- */
  /*                            Spatial distribution                            */
  /* -------------------------------------------------------------------------- */
  template <Dimension D, SimulationEngine S>
  struct SpatialDistribution {
    SpatialDistribution(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : m_params { params }, m_mblock { mblock } {}

    Inline virtual auto operator()(const coord_t<D>&) const -> real_t {
      return ONE;
    }

  private:
    SimulationParams m_params;
    Meshblock<D, S>  m_mblock;
  };

  template <Dimension D, SimulationEngine S>
  struct UniformDist : public SpatialDistribution<D, S> {
    UniformDist(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : SpatialDistribution<D, S>(params, mblock) {}
    Inline auto operator()(const coord_t<D>&) const -> real_t override {
      return ONE;
    }
  };

  /* -------------------------------------------------------------------------- */
  /*                             Injection criterion                            */
  /* -------------------------------------------------------------------------- */
  template <Dimension D, SimulationEngine S>
  struct InjectionCriterion {
    InjectionCriterion(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : m_params { params }, m_mblock { mblock } {}

    Inline virtual auto operator()(const coord_t<D>&) const -> bool {
      return true;
    }

  protected:
    SimulationParams m_params;
    Meshblock<D, S>  m_mblock;
  };

  template <Dimension D, SimulationEngine S>
  struct NoCriterion : public InjectionCriterion<D, S> {
    NoCriterion(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : InjectionCriterion<D, S>(params, mblock) {}
    Inline auto operator()(const coord_t<D>&) const -> bool {
      return true;
    }
  };

  /* -------------------------------------------------------------------------- */
  /*                              Vector potential                              */
  /* -------------------------------------------------------------------------- */
  template <Dimension D, SimulationEngine S>
  struct VectorPotential {
    VectorPotential(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : m_params { params }, m_mblock { mblock } {}

    Inline virtual auto A_x0(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
    Inline virtual auto A_x1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
    Inline virtual auto A_x2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
    Inline virtual auto A_x3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  protected:
    SimulationParams m_params;
    Meshblock<D, S>  m_mblock;
  };

}    // namespace ntt

#endif    // FRAMEWORK_ARCHETYPES_H