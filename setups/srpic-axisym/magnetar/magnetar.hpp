#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "particle_macros.h"
#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"
#include "utils/injector.hpp"

#ifdef GUI_ENABLED
  #include "nttiny/api.h"
#endif

namespace ntt {
  enum FieldMode {
    MonopoleField = 1,
    DipoleField   = 2
  };

  /**
   * Define a structure which will initialize the particle energy distribution.
   * This is used below in the UserInitParticles function.
   */
  template <Dimension D, SimulationEngine S>
  struct ThermalBackground : public EnergyDistribution<D, S> {
    ThermalBackground(const SimulationParams& params,
                      const Meshblock<D, S>&  mblock) :
      EnergyDistribution<D, S>(params, mblock),
      maxwellian { mblock },
      temperature { params.get<real_t>("problem", "atm_T") } {}

    Inline void operator()(const coord_t<PrtlCoordD>&,
                           vec_t<Dim3>& v,
                           const int&) const override {
      maxwellian(v, temperature);
      v[1] = ZERO;
      v[2] = ZERO;
    }

  private:
    const Maxwellian<D, S> maxwellian;
    const real_t           temperature;
  };

  /**
   * Main problem generator class with all the required functions to define
   * the initial/boundary conditions and the source terms.
   */
  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    real_t work_done { ZERO };

    inline ProblemGenerator(const SimulationParams& params) :
      m_atm_T { params.get<real_t>("problem", "atm_T") },
      m_atm_C { params.get<real_t>("problem", "atm_contrast") },
      m_atm_h { params.get<real_t>("problem", "atm_h") },
      m_psr_Rstar { params.get<real_t>("problem", "atm_buff") + params.extent()[0] },
      m_psr_Bsurf { params.get<real_t>("problem", "psr_Bsurf", ONE) },
      m_psr_omega { params.get<real_t>("problem", "psr_omega") },
      m_psr_spinup_time { params.get<real_t>("problem", "psr_spinup_time", ZERO) },
      m_rGJ { math::log(m_atm_C) * m_atm_h + m_psr_Rstar },
      m_gravity { m_atm_T / m_atm_h },
      m_psr_field_mode { params.get<int>("problem", "psr_field_mode", 2) == 2
                           ? DipoleField
                           : MonopoleField } {}

    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {
    }

    inline void UserInitParticles(const SimulationParams&,
                                  Meshblock<D, S>&) override {}

    inline void UserDriveFields(const real_t&,
                                const SimulationParams&,
                                Meshblock<D, S>&) override {}

    Inline auto ext_force_x1(const real_t&, const coord_t<PrtlCoordD>& x_ph) const
      -> real_t override {
      return -m_gravity * SQR(m_psr_Rstar / x_ph[0]) *
             (x_ph[0] < m_psr_Rstar + m_atm_h * TWO);
    }

    Inline auto ext_force_x2(const real_t&, const coord_t<PrtlCoordD>&) const
      -> real_t override {
      return ZERO;
    }

    Inline auto ext_force_x3(const real_t&, const coord_t<PrtlCoordD>&) const
      -> real_t override {
      return ZERO;
    }

  private:
    const real_t    m_atm_T, m_atm_C, m_atm_h;
    const real_t    m_psr_Rstar, m_psr_Bsurf, m_psr_omega, m_psr_spinup_time;
    const real_t    m_rGJ, m_gravity;
    const FieldMode m_psr_field_mode;
    ndarray_t<(short)(D)> m_ppc_per_spec;
  };

  Inline void mainBField(const coord_t<PrtlCoordD>& x_ph,
                         vec_t<Dim3>&,
                         vec_t<Dim3>& b_out,
                         real_t       _rstar,
                         real_t       _bsurf,
                         int          _mode) {
    if (_mode == 2) {
      b_out[0] = _bsurf * math::cos(x_ph[1]) / CUBE(x_ph[0] / _rstar);
      b_out[1] = _bsurf * HALF * math::sin(x_ph[1]) / CUBE(x_ph[0] / _rstar);
      b_out[2] = ZERO;
    } else {
      b_out[0] = _bsurf * SQR(_rstar / x_ph[0]);
      b_out[1] = ZERO;
      b_out[2] = ZERO;
    }
  }

  Inline void surfaceRotationField(const coord_t<PrtlCoordD>& x_ph,
                                   vec_t<Dim3>&          e_out,
                                   vec_t<Dim3>&          b_out,
                                   real_t                _rstar,
                                   real_t                _bsurf,
                                   int                   _mode,
                                   real_t                _omega) {
    mainBField(x_ph, e_out, b_out, _rstar, _bsurf, _mode);
    // e_out[0] = _omega * b_out[1] * x_ph[0] * math::sin(x_ph[1]);
    // e_out[1] = -_omega * b_out[0] * x_ph[0] * math::sin(x_ph[1]);

    // _omega   *= 0.5 * (1 - math::tanh(50 * (-0.698132 + x_ph[1])) *
    //                        math::tanh(50 * (-0.523599 + x_ph[1])));
    _omega   *= -0.5 * (math::tanh(50 * (-0.6108652381980153 + x_ph[1])) *
                      (-1 + math::tanh(50 * (-0.6981317007977319 + x_ph[1])) *
                              math::tanh(50 * (-0.5235987755982989 + x_ph[1]))));
    e_out[0]  = _omega * b_out[1] * x_ph[0] * math::sin(x_ph[1]);
    e_out[1]  = -_omega * b_out[0] * x_ph[0] * math::sin(x_ph[1]);
  }

  template <Dimension D, SimulationEngine S>
  struct PgenTargetFields : public TargetFields<D, S> {
    PgenTargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock) :
      TargetFields<D, S>(params, mblock),
      _rstar { params.get<real_t>("problem", "atm_buff") + params.extent()[0] },
      _bsurf { params.get<real_t>("problem", "psr_Bsurf", ONE) },
      _mode { params.get<int>("problem", "psr_field_mode", 2) } {}

    Inline real_t operator()(const em&             comp,
                             const coord_t<PrtlCoordD>& xi) const override {
      if ((comp == em::bx1) || (comp == em::bx2)) {
        vec_t<Dim3>    e_out { ZERO }, b_out { ZERO };
        coord_t<PrtlCoordD> x_ph { ZERO };
        (this->m_mblock).metric.x_Code2Phys(xi, x_ph);
        mainBField(x_ph, e_out, b_out, _rstar, _bsurf, _mode);
        return (comp == em::bx1) ? b_out[0] : b_out[1];
      } else {
        return ZERO;
      }
    }

  private:
    const real_t _rstar, _bsurf;
    const int    _mode;
  };

  Inline auto densityProfile(real_t r, real_t C, real_t h, real_t Rstar) -> real_t {
    return C * math::exp(-(Rstar / h) * (ONE - (Rstar / r)));
  }

  Inline auto planckSample(RandomNumberPool_t::generator_type rand_gen) -> real_t {
    real_t prob, n, rnd;
    rnd  = Random<real_t>(rand_gen);
    prob = 0.0;
    n    = 0.0;
    while (prob < rnd && n < 40.0) {
      n    += 1.0;
      prob += 1.0 / (1.20206 * n * n * n);
    }
    return -math::log(Random<real_t>(rand_gen) * Random<real_t>(rand_gen) *
                        Random<real_t>(rand_gen) +
                      1e-16) /
           n;
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
    const SimulationParams&     params,
    Meshblock<Dim2, PICEngine>& mblock) {
    // initialize buffer array
    m_ppc_per_spec = ndarray_t<2>("ppc_per_spec", mblock.Ni1(), mblock.Ni2());

    // inject stars in the atmosphere
    {
      const auto ppc0  = params.ppc0();
      const auto C     = m_atm_C;
      const auto h     = m_atm_h;
      const auto rstar = m_psr_Rstar;

      Kokkos::parallel_for(
        "ComputeDeltaNdens",
        mblock.rangeActiveCells(),
        ClassLambda(index_t i1, index_t i2) {
          const auto i1_ = static_cast<int>(i1) - N_GHOSTS;
          const auto i2_ = static_cast<int>(i2) - N_GHOSTS;
          const auto r = mblock.metric.x1_Code2Phys(static_cast<real_t>(i1_) + HALF);
          m_ppc_per_spec(i1_, i2_) = densityProfile(r, C, h, rstar) * (r > rstar) *
                                     (r < rstar + static_cast<real_t>(10) * h);
          // 2 -- for two species
          m_ppc_per_spec(i1_, i2_) *= ppc0 / TWO;
        });
      InjectNonUniform<Dim2, PICEngine, ThermalBackground>(params,
                                                           mblock,
                                                           { 1, 2 },
                                                           m_ppc_per_spec);
    }
  }

  /**
   * Field initialization for 2D:
   */

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
    const SimulationParams&     params,
    Meshblock<Dim2, PICEngine>& mblock) {
    {
      const auto rmin   = mblock.metric.x1_min;
      const auto x1_psr = mblock.metric.x1_Phys2Code(m_psr_Rstar);
      NTTHostErrorIf(rmin >= m_psr_Rstar, "rmin > r_surf");
      NTTHostErrorIf(x1_psr < params.currentFilters(), "r_surf - rmin < filters");
    }
    {
      const auto rstar = m_psr_Rstar;
      const auto bsurf = m_psr_Bsurf;
      const auto mode  = m_psr_field_mode;
      Kokkos::parallel_for(
        "UserInitFields",
        mblock.rangeActiveCells(),
        ClassLambda(index_t i, index_t j) {
          set_em_fields_2d(mblock, i, j, mainBField, rstar, bsurf, mode);
        });
    }
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveFields(
    const real_t& time,
    const SimulationParams&,
    Meshblock<Dim2, PICEngine>& mblock) {
    const unsigned int i1_surf {
      static_cast<int>(mblock.metric.x1_Phys2Code(m_psr_Rstar)) + N_GHOSTS
    };
    const auto i1_min = mblock.i1_min();

    {
      const auto rstar = m_psr_Rstar;
      const auto bsurf = m_psr_Bsurf;
      const auto mode  = m_psr_field_mode;
      const auto omega = m_psr_omega * ((time < m_psr_spinup_time)
                                          ? (time / m_psr_spinup_time)
                                          : ONE);

      Kokkos::parallel_for(
        "UserDriveFields_rmin",
        CreateRangePolicy<Dim2>({ i1_min, mblock.i2_min() },
                                { i1_surf, mblock.i2_max() }),
        ClassLambda(index_t i1, index_t i2) {
          set_ex2_2d(mblock, i1, i2, surfaceRotationField, rstar, bsurf, mode, omega);
          set_ex3_2d(mblock, i1, i2, surfaceRotationField, rstar, bsurf, mode, omega);
          set_bx1_2d(mblock, i1, i2, surfaceRotationField, rstar, bsurf, mode, omega);
          if (i1 < i1_surf - 1) {
            set_ex1_2d(mblock, i1, i2, surfaceRotationField, rstar, bsurf, mode, omega);
            set_bx2_2d(mblock, i1, i2, surfaceRotationField, rstar, bsurf, mode, omega);
            set_bx3_2d(mblock, i1, i2, surfaceRotationField, rstar, bsurf, mode, omega);
          }
        });
    }
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveParticles(
    const real_t&,
    const SimulationParams&     params,
    Meshblock<Dim2, PICEngine>& m_mblock) {
    constexpr short buff_idx = 0;
    {
      // compute the number densities
      const short smooth = 0;
      m_mblock.ComputeMoments(params, FieldID::N, {}, { 1, 2 }, buff_idx, smooth);
    }
    Kokkos::deep_copy(m_ppc_per_spec, ZERO);
    {
      const auto C     = m_atm_C;
      const auto h     = m_atm_h;
      const auto rstar = m_psr_Rstar;
      const auto ppc0  = params.ppc0();
      const auto frac  = static_cast<real_t>(0.9) * (ONE - ONE / ppc0);

      // determine how much to inject
      Kokkos::parallel_for(
        "ComputeDeltaNdens",
        m_mblock.rangeActiveCells(),
        ClassLambda(index_t i1, index_t i2) {
          const auto i1_ = i1 - static_cast<int>(N_GHOSTS),
                     i2_ = i2 - static_cast<int>(N_GHOSTS);
          const auto r   = m_mblock.metric.x1_Code2Phys(i1_ + HALF);

          m_ppc_per_spec(i1_, i2_) = densityProfile(r, C, h, rstar) *
                                     (r > rstar) * (r < rstar + h);

          const auto actual_ndens = m_mblock.buff(i1, i2, buff_idx);
          if (frac * m_ppc_per_spec(i1_, i2_) > actual_ndens) {
            m_ppc_per_spec(i1_, i2_) = m_ppc_per_spec(i1_, i2_) - actual_ndens;
          } else {
            m_ppc_per_spec(i1_, i2_) = ZERO;
          }
          // 2 -- for two species
          m_ppc_per_spec(i1_, i2_) = int(ppc0 * m_ppc_per_spec(i1_, i2_) / TWO);
        });

      // injection
      InjectNonUniform<Dim2, PICEngine, ThermalBackground>(params,
                                                           m_mblock,
                                                           { 1, 2 },
                                                           m_ppc_per_spec);
    }

    // Resonant scattering kernel
    {
      const auto angThres { params.get<real_t>("problem", "angThres") };
      const auto fid_freq { params.get<real_t>("problem", "fid_freq") };
      const auto bq { params.get<real_t>("problem", "bq") };
      auto       random_pool  = *(m_mblock.random_pool_ptr);
      auto&      photons_par  = m_mblock.particles[2];
      auto&      photons_perp = m_mblock.particles[3];
      const auto l0           = params.larmor0();
      const auto dt           = m_mblock.timestep();

      for (std::size_t s { 0 }; s < 6; ++s) {
        if ((s == 2) || (s == 3)) {
          continue;
        }
        auto&                species = m_mblock.particles[s];
        array_t<std::size_t> ph_ind_par("ph_ind");
        array_t<std::size_t> ph_ind_perp("ph_ind");
        const auto           ph_offset_par  = photons_par.npart();
        const auto           ph_offset_perp = photons_perp.npart();

        Kokkos::parallel_for(
          "ResonantScattering",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            if (species.tag(p) != ParticleTag::alive) {
              return;
            }

            // Interpolation of B to the position of the photon
            vec_t<Dim3> b_int_Cart;
            {
              vec_t<Dim3> b_int;
              real_t      c000, c100, c010, c110, c00, c10;

              const auto   i { species.i1(p) + N_GHOSTS };
              const real_t dx1 { species.dx1(p) };
              const auto   j { species.i2(p) + N_GHOSTS };
              const real_t dx2 { species.dx2(p) };
              // Bx1
              c000     = HALF * (BX1(i, j) + BX1(i, j - 1));
              c100     = HALF * (BX1(i + 1, j) + BX1(i + 1, j - 1));
              c010     = HALF * (BX1(i, j) + BX1(i, j + 1));
              c110     = HALF * (BX1(i + 1, j) + BX1(i + 1, j + 1));
              c00      = c000 * (ONE - dx1) + c100 * dx1;
              c10      = c010 * (ONE - dx1) + c110 * dx1;
              b_int[0] = c00 * (ONE - dx2) + c10 * dx2;
              // Bx2
              c000     = HALF * (BX2(i - 1, j) + BX2(i, j));
              c100     = HALF * (BX2(i, j) + BX2(i + 1, j));
              c010     = HALF * (BX2(i - 1, j + 1) + BX2(i, j + 1));
              c110     = HALF * (BX2(i, j + 1) + BX2(i + 1, j + 1));
              c00      = c000 * (ONE - dx1) + c100 * dx1;
              c10      = c010 * (ONE - dx1) + c110 * dx1;
              b_int[1] = c00 * (ONE - dx2) + c10 * dx2;
              // Bx3
              c000     = INV_4 * (BX3(i - 1, j - 1) + BX3(i - 1, j) +
                              BX3(i, j - 1) + BX3(i, j));
              c100 = INV_4 * (BX3(i, j - 1) + BX3(i, j) + BX3(i + 1, j - 1) +
                              BX3(i + 1, j));
              c010 = INV_4 * (BX3(i - 1, j) + BX3(i - 1, j + 1) + BX3(i, j) +
                              BX3(i, j + 1));
              c110 = INV_4 * (BX3(i, j) + BX3(i, j + 1) + BX3(i + 1, j) +
                              BX3(i + 1, j + 1));
              c00  = c000 * (ONE - dx1) + c100 * dx1;
              c10  = c010 * (ONE - dx1) + c110 * dx1;
              b_int[2] = c00 * (ONE - dx2) + c10 * dx2;

              const vec_t<Dim3> xp { static_cast<real_t>(species.i1(p)) +
                                       static_cast<real_t>(species.dx1(p)),
                                     static_cast<real_t>(species.i2(p)) +
                                       static_cast<real_t>(species.dx2(p)),
                                     species.phi(p) };
              m_mblock.metric.v3_Cntrv2Cart(xp, b_int, b_int_Cart);
            }

            // Interpolation of E to the position of the photon
            vec_t<Dim3> e_int_Cart;
            {
              vec_t<Dim3> e_int;
              real_t      c000, c100, c010, c110, c00, c10;

              const auto   i { species.i1(p) + N_GHOSTS };
              const real_t dx1 { species.dx1(p) };
              const auto   j { species.i2(p) + N_GHOSTS };
              const real_t dx2 { species.dx2(p) };
              // Ex1
              c000     = HALF * (EX1(i, j) + EX1(i, j - 1));
              c100     = HALF * (EX1(i + 1, j) + EX1(i + 1, j - 1));
              c010     = HALF * (EX1(i, j) + EX1(i, j + 1));
              c110     = HALF * (EX1(i + 1, j) + EX1(i + 1, j + 1));
              c00      = c000 * (ONE - dx1) + c100 * dx1;
              c10      = c010 * (ONE - dx1) + c110 * dx1;
              e_int[0] = c00 * (ONE - dx2) + c10 * dx2;
              // Ex2
              c000     = HALF * (EX2(i - 1, j) + EX2(i, j));
              c100     = HALF * (EX2(i, j) + EX2(i + 1, j));
              c010     = HALF * (EX2(i - 1, j + 1) + EX2(i, j + 1));
              c110     = HALF * (EX2(i, j + 1) + EX2(i + 1, j + 1));
              c00      = c000 * (ONE - dx1) + c100 * dx1;
              c10      = c010 * (ONE - dx1) + c110 * dx1;
              e_int[1] = c00 * (ONE - dx2) + c10 * dx2;
              // Ex3
              c000     = INV_4 * (EX3(i - 1, j - 1) + EX3(i - 1, j) +
                              EX3(i, j - 1) + EX3(i, j));
              c100 = INV_4 * (EX3(i, j - 1) + EX3(i, j) + EX3(i + 1, j - 1) +
                              EX3(i + 1, j));
              c010 = INV_4 * (EX3(i - 1, j) + EX3(i - 1, j + 1) + EX3(i, j) +
                              EX3(i, j + 1));
              c110 = INV_4 * (EX3(i, j) + EX3(i, j + 1) + EX3(i + 1, j) +
                              EX3(i + 1, j + 1));
              c00  = c000 * (ONE - dx1) + c100 * dx1;
              c10  = c010 * (ONE - dx1) + c110 * dx1;
              e_int[2] = c00 * (ONE - dx2) + c10 * dx2;

              const vec_t<Dim3> xp { static_cast<real_t>(species.i1(p)) +
                                       static_cast<real_t>(species.dx1(p)),
                                     static_cast<real_t>(species.i2(p)) +
                                       static_cast<real_t>(species.dx2(p)),
                                     species.phi(p) };
              m_mblock.metric.v3_Cntrv2Cart(xp, e_int, e_int_Cart);
            }

            auto px      = species.ux1(p);
            auto py      = species.ux2(p);
            auto pz      = species.ux3(p);
            auto gamma   = math::sqrt(ONE + px * px + py * py + pz * pz);
            auto betax   = px / gamma;
            auto betay   = py / gamma;
            auto betaz   = pz / gamma;
            auto beta_sq = betax * betax + betay * betay + betaz * betaz;

            // Boost magnetic fields to the rest frame
            auto bx0_rest = gamma * (b_int_Cart[0] - betay * e_int_Cart[2] +
                                     betaz * e_int_Cart[1]) -
                            (gamma - 1.0) *
                              (b_int_Cart[0] * betax + b_int_Cart[1] * betay +
                               b_int_Cart[2] * betaz) *
                              betax / beta_sq;
            auto by0_rest = gamma * (b_int_Cart[1] - betaz * e_int_Cart[0] +
                                     betax * e_int_Cart[2]) -
                            (gamma - 1.0) *
                              (b_int_Cart[0] * betax + b_int_Cart[1] * betay +
                               b_int_Cart[2] * betaz) *
                              betay / beta_sq;
            auto bz0_rest = gamma * (b_int_Cart[2] - betax * e_int_Cart[1] +
                                     betay * e_int_Cart[0]) -
                            (gamma - 1.0) *
                              (b_int_Cart[0] * betax + b_int_Cart[1] * betay +
                               b_int_Cart[2] * betaz) *
                              betaz / beta_sq;

            auto ex0_rest = gamma * (e_int_Cart[0] + betay * b_int_Cart[2] -
                                     betaz * b_int_Cart[1]) -
                            (gamma - 1.0) *
                              (e_int_Cart[0] * betax + e_int_Cart[1] * betay +
                               e_int_Cart[2] * betaz) *
                              betax / beta_sq;
            auto ey0_rest = gamma * (e_int_Cart[1] + betaz * b_int_Cart[0] -
                                     betax * b_int_Cart[2]) -
                            (gamma - 1.0) *
                              (e_int_Cart[0] * betax + e_int_Cart[1] * betay +
                               e_int_Cart[2] * betaz) *
                              betay / beta_sq;
            auto ez0_rest = gamma * (e_int_Cart[2] + betax * b_int_Cart[1] -
                                     betay * b_int_Cart[0]) -
                            (gamma - 1.0) *
                              (e_int_Cart[0] * betax + e_int_Cart[1] * betay +
                               e_int_Cart[2] * betaz) *
                              betaz / beta_sq;

            // Build a basis along the lab frame magnetic field
            auto norm { 1.0 / NORM(b_int_Cart[0], b_int_Cart[1], b_int_Cart[2]) };
            auto a_RF_x = b_int_Cart[0] * norm;
            auto a_RF_y = b_int_Cart[1] * norm;
            auto a_RF_z = b_int_Cart[2] * norm;
            auto b_RF_x = 1.0;
            auto b_RF_y = 0.0;
            auto b_RF_z = 0.0;
            if (a_RF_x != ZERO) {
              b_RF_x = -a_RF_y / a_RF_x;
              b_RF_y = 1.0;
              b_RF_z = 0.0;
              norm        = 1.0 / sqrt(b_RF_x * b_RF_x + b_RF_y * b_RF_y);
              b_RF_x      = b_RF_x * norm;
              b_RF_y      = b_RF_y * norm;
            }
            auto c_RF_x = b_RF_z * a_RF_y - b_RF_y * a_RF_z;
            auto c_RF_y = b_RF_x * a_RF_z - b_RF_z * a_RF_x;
            auto c_RF_z = b_RF_y * a_RF_x - b_RF_x * a_RF_y;

            // Calculate gyrofrequency in the lepton rest frame
            auto omegaB0 = 1.0 / l0;
            auto omegaB  = omegaB0 *
                          math::sqrt(bx0_rest * bx0_rest + by0_rest * by0_rest +
                                     bz0_rest * bz0_rest);

            // Monte Carlo evaluation of the cross section
            auto   tres  = 0.0;
            auto   qdist = 0.0;
            auto   vmax  = 0.0;
            auto   vmin  = 100000.0;
            auto   nmax  = 1;
            real_t eph_LF_L, eph_RF_L, u_ph_RF_L, v_ph_RF_L, w_ph_RF_L, u_ph_L,
              v_ph_L, w_ph_L;

            for (std::size_t n { 0 }; n < nmax; ++n) {

              // Calculate the photon energy and momentum
              typename RandomNumberPool_t::generator_type rand_gen =
                random_pool.get_state();
              auto eph_LF = fid_freq * omegaB0 * bq * planckSample(rand_gen);

              // Here fix the direction of photon momentum (e.g., spherical from
              // star, isotropic, etc.) Isotropic photon distribution
              // auto rand_costheta_RF = math::cos(Random<real_t>(rand_gen) * M_PI);
              // auto rand_sintheta_RF = math::sqrt(
              //   1.0 - rand_costheta_RF * rand_costheta_RF);
              // auto rand_phi_RF    = 2.0 * M_PI * Random<real_t>(rand_gen);
              // auto rand_cosphi_RF = math::cos(rand_phi_RF);
              // auto rand_sinphi_RF = math::sin(rand_phi_RF);
              // auto u_ph           = eph_LF * (rand_costheta_RF * a_RF_x +
              //                       rand_sintheta_RF * rand_cosphi_RF * b_RF_x +
              //                       rand_sintheta_RF * rand_sinphi_RF * c_RF_x);
              // auto v_ph           = eph_LF * (rand_costheta_RF * a_RF_y +
              //                       rand_sintheta_RF * rand_cosphi_RF * b_RF_y +
              //                       rand_sintheta_RF * rand_sinphi_RF * c_RF_y);
              // auto w_ph           = eph_LF * (rand_costheta_RF * a_RF_z +
              //                       rand_sintheta_RF * rand_cosphi_RF * b_RF_z +
              //                       rand_sintheta_RF * rand_sinphi_RF * c_RF_z);

              // Radial photon distribution
              const vec_t<Dim3> xi { i_di_to_Xi(species.i1(p), species.dx1(p)),
                                     i_di_to_Xi(species.i2(p), species.dx2(p)),
                                     species.phi(p) };
              coord_t<Dim3>     x_ph { ZERO };
              m_mblock.metric.x_Code2Cart(xi, x_ph);
              auto xnorm { 1.0 / NORM(x_ph[0], x_ph[1], x_ph[2]) };
              auto x1norm = x_ph[0] * xnorm;
              auto x2norm = x_ph[1] * xnorm;
              auto x3norm = x_ph[2] * xnorm;
              auto u_ph   = eph_LF * x1norm;
              auto v_ph   = eph_LF * x2norm;
              auto w_ph   = eph_LF * x3norm;
              auto rand_costheta_RF {
                DOT(px, py, pz, x1norm, x2norm, x3norm) / NORM(px, py, pz)
              };

              // Boost photon momentum to the lepton rest frame
              auto pdotk   = u_ph * px + v_ph * py + w_ph * pz;
              auto eph_RF  = gamma * eph_LF - pdotk;
              auto u_ph_RF = u_ph + (pdotk / (1.0 + gamma) - eph_LF) * px;
              auto v_ph_RF = v_ph + (pdotk / (1.0 + gamma) - eph_LF) * py;
              auto w_ph_RF = w_ph + (pdotk / (1.0 + gamma) - eph_LF) * pz;

              // Boost resonant frequency back to the lab frame
              auto omegaBL = (omegaB + px * u_ph + py * v_ph + pz * w_ph) / gamma;

              // Calculate the resonance quality factor
              auto xres = (1.0 - omegaBL / eph_LF);
              auto qres = math::exp(-0.5 * xres * xres /
                                    ((0.1 / gamma) * (0.1 / gamma))) /
                          sqrt(2.0 * M_PI * ((0.1 / gamma) * (0.1 / gamma))) *
                          (1.0 - sqrt(beta_sq) * rand_costheta_RF) / eph_LF;
              tres += qres;

              if (qres > qdist) {
                eph_LF_L  = eph_RF;
                u_ph_RF_L = u_ph_RF;
                v_ph_RF_L = v_ph_RF;
                w_ph_RF_L = w_ph_RF;
                qdist     = qres;
                u_ph_L    = u_ph;
                v_ph_L    = v_ph;
                w_ph_L    = w_ph;
              }

              if (xres > vmax) {
                vmax = xres;
              }

              if (xres < vmin) {
                vmin = xres;
              }

              random_pool.free_state(rand_gen);
            }

            // Calculate cross section and trigger scattering event
            auto p_scatter = tres / static_cast<real_t>(nmax) * 1.0 / (vmax - vmin);
            typename RandomNumberPool_t::generator_type rand_gen =
              random_pool.get_state();
            if (Random<real_t>(rand_gen) < p_scatter) {

              // Calculate energy and momentum of the excited lepton
              auto gammaeb     = 1.0 + math::sqrt(u_ph_RF_L * u_ph_RF_L +
                                              v_ph_RF_L * v_ph_RF_L +
                                              w_ph_RF_L * w_ph_RF_L);
              auto betax_ex    = u_ph_RF_L / gammaeb;
              auto betay_ex    = v_ph_RF_L / gammaeb;
              auto betaz_ex    = w_ph_RF_L / gammaeb;
              auto gamma_ex    = 1.0 / math::sqrt(1.0 - betax_ex * betax_ex -
                                               betay_ex * betay_ex -
                                               betaz_ex * betaz_ex);
              auto eb          = gammaeb / gamma_ex;
              auto pel_ex_x    = gamma_ex * betax_ex;
              auto pel_ex_y    = gamma_ex * betay_ex;
              auto pel_ex_z    = gamma_ex * betaz_ex;
              auto betax_ex_sq = betax_ex * betax_ex + betay_ex * betay_ex +
                                 betaz_ex * betaz_ex;

              // Boost fields into the de-excitation rest frame
              auto bx0_drest = gamma_ex * (bx0_rest - betay_ex * ez0_rest +
                                           betaz_ex * ey0_rest) -
                               (gamma_ex - 1.0) *
                                 (bx0_rest * betax_ex + by0_rest * betay_ex +
                                  bz0_rest * betaz_ex) *
                                 betax_ex / betax_ex_sq;
              auto by0_drest = gamma_ex * (by0_rest - betaz_ex * ex0_rest +
                                           betax_ex * ez0_rest) -
                               (gamma_ex - 1.0) *
                                 (bx0_rest * betax_ex + by0_rest * betay_ex +
                                  bz0_rest * betaz_ex) *
                                 betay_ex / betax_ex_sq;
              auto bz0_drest = gamma_ex * (bz0_rest - betax_ex * ey0_rest +
                                           betay_ex * ex0_rest) -
                               (gamma_ex - 1.0) *
                                 (bx0_rest * betax_ex + by0_rest * betay_ex +
                                  bz0_rest * betaz_ex) *
                                 betaz_ex / betax_ex_sq;

              // Prescribe the fraction of parallel polarization
              bool pol_par = false;
              if (Random<real_t>(rand_gen) < 0.25) {
                pol_par = true;
              }

              // Draw scattering angles (depending on polarization)
              auto mudash = Random<real_t>(rand_gen);
              if (pol_par) {
                if (mudash <= 0.5) {
                  mudash = -math::pow(math::abs(-1.0 + 2.0 * mudash),
                                      0.3333333333333333);
                } else {
                  mudash = math::pow(math::abs(-1.0 + 2.0 * mudash),
                                     0.3333333333333333);
                }
              } else {
                mudash = (2.0 * mudash - 1.0);
              }

              if(fabs(mudash) >= 1.0) {
                printf("mudash = %e\n", mudash);
              }

              // Build a basis along the rest frame magnetic field
              auto norm { 1.0 / NORM(bx0_drest, by0_drest, bz0_drest) };
              auto a_RF_x = bx0_drest * norm;
              auto a_RF_y = by0_drest * norm;
              auto a_RF_z = bz0_drest * norm;
              auto b_RF_x = 1.0;
              auto b_RF_y = 0.0;
              auto b_RF_z = 0.0;
              if (a_RF_x != ZERO) {
                b_RF_x = -a_RF_y / a_RF_x;
                b_RF_y = 1.0;
                b_RF_z = 0.0;
                norm   = 1.0 / math::sqrt(b_RF_x * b_RF_x + b_RF_y * b_RF_y);
                b_RF_x = b_RF_x * norm;
                b_RF_y = b_RF_y * norm;
              }
              auto c_RF_x = b_RF_z * a_RF_y - b_RF_y * a_RF_z;
              auto c_RF_y = b_RF_x * a_RF_z - b_RF_z * a_RF_x;
              auto c_RF_z = b_RF_y * a_RF_x - b_RF_x * a_RF_y;

              // Calculate emission vector for photon
              auto rand_phi_RF      = 2.0 * M_PI * Random<real_t>(rand_gen);
              auto rand_costheta_RF = mudash;
              auto rand_sintheta_RF = math::sqrt(1.0 - mudash * mudash);
              auto rand_cosphi_RF   = math::cos(rand_phi_RF);
              auto rand_sinphi_RF   = math::sin(rand_phi_RF);

              // Calculate rest frame energy and momentum of emitted photon
              auto eph_RF = eb / (rand_sintheta_RF * rand_sintheta_RF) *
                            (1.0 -
                             math::sqrt((rand_costheta_RF * rand_costheta_RF) +
                                        (1.0 / (eb * eb)) *
                                          (rand_sintheta_RF * rand_sintheta_RF)));
              
              if(fabs(mudash) >= 1.0) {
                eph_RF = (-1 + pow(eb,2))/(2.*eb);
              }

              auto kph_RF_x = eph_RF *
                              (rand_costheta_RF * a_RF_x +
                               rand_sintheta_RF * rand_cosphi_RF * b_RF_x +
                               rand_sintheta_RF * rand_sinphi_RF * c_RF_x);
              auto kph_RF_y = eph_RF *
                              (rand_costheta_RF * a_RF_y +
                               rand_sintheta_RF * rand_cosphi_RF * b_RF_y +
                               rand_sintheta_RF * rand_sinphi_RF * c_RF_y);
              auto kph_RF_z = eph_RF *
                              (rand_costheta_RF * a_RF_z +
                               rand_sintheta_RF * rand_cosphi_RF * b_RF_z +
                               rand_sintheta_RF * rand_sinphi_RF * c_RF_z);

              // Calculate the lepton energy and momentum after scattering
              auto el_RF   = eb - eph_RF;
              auto kl_RF_x = - eph_RF * rand_costheta_RF * a_RF_x;
              auto kl_RF_y = - eph_RF * rand_costheta_RF * a_RF_y;
              auto kl_RF_z = - eph_RF * rand_costheta_RF * a_RF_z;

              // Boost lepton momentum to excitation frame
              auto el_EX = -(-pel_ex_x) * kl_RF_x - (-pel_ex_y) * kl_RF_y -
                           (-pel_ex_z) * kl_RF_z + el_RF * gamma_ex;
              auto kl_EX_x =
                (-(kl_RF_z * (-pel_ex_x) * (-pel_ex_z)) +
                 kl_RF_x * (math::pow(-pel_ex_y, 2) + math::pow(-pel_ex_z, 2)) -
                 el_RF * (-pel_ex_x) *
                   (math::pow(-pel_ex_x, 2) + math::pow(-pel_ex_y, 2) +
                    math::pow(-pel_ex_z, 2)) +
                 kl_RF_y * (-pel_ex_x) * (-pel_ex_y) * (-1 + gamma_ex) +
                 (-pel_ex_x) * (kl_RF_x * (-pel_ex_x) + kl_RF_z * (-pel_ex_z)) *
                   gamma_ex) /
                (math::pow(-pel_ex_x, 2) + math::pow(-pel_ex_y, 2) +
                 math::pow(-pel_ex_z, 2));
              auto kl_EX_y =
                (-((-pel_ex_y) *
                   (kl_RF_x * (-pel_ex_x) + kl_RF_z * (-pel_ex_z) +
                    el_RF * (math::pow(-pel_ex_x, 2) + math::pow(-pel_ex_y, 2) +
                             math::pow(-pel_ex_z, 2)))) +
                 (-pel_ex_y) * (kl_RF_x * (-pel_ex_x) + kl_RF_z * (-pel_ex_z)) *
                   gamma_ex +
                 kl_RF_y * (math::pow(-pel_ex_x, 2) + math::pow(-pel_ex_z, 2) +
                            math::pow(-pel_ex_y, 2) * gamma_ex)) /
                (math::pow(-pel_ex_x, 2) + math::pow(-pel_ex_y, 2) +
                 math::pow(-pel_ex_z, 2));
              auto kl_EX_z =
                (-((-pel_ex_z) *
                   (kl_RF_x * (-pel_ex_x) + kl_RF_y * (-pel_ex_y) +
                    el_RF * (math::pow(-pel_ex_x, 2) + math::pow(-pel_ex_y, 2) +
                             math::pow(-pel_ex_z, 2)))) +
                 (kl_RF_x * (-pel_ex_x) + kl_RF_y * (-pel_ex_y)) * (-pel_ex_z) *
                   gamma_ex +
                 kl_RF_z * (math::pow(-pel_ex_x, 2) + math::pow(-pel_ex_y, 2) +
                            math::pow(-pel_ex_z, 2) * gamma_ex)) /
                (math::pow(-pel_ex_x, 2) + math::pow(-pel_ex_y, 2) +
                 math::pow(-pel_ex_z, 2));

              // Boost lepton momentum to lab frame
              auto gamma_el_new = -(-px) * kl_EX_x - (-py) * kl_EX_y -
                                  (-pz) * kl_EX_z + el_EX * gamma;
              auto u_el_new =
                (-(kl_EX_z * (-px) * (-pz)) +
                 kl_EX_x * (math::pow(-py, 2) + math::pow(-pz, 2)) -
                 el_EX * (-px) *
                   (math::pow(-px, 2) + math::pow(-py, 2) + math::pow(-pz, 2)) +
                 kl_EX_y * (-px) * (-py) * (-1 + gamma) +
                 (-px) * (kl_EX_x * (-px) + kl_EX_z * (-pz)) * gamma) /
                (math::pow(-px, 2) + math::pow(-py, 2) + math::pow(-pz, 2));
              auto v_el_new =
                (-((-py) * (kl_EX_x * (-px) + kl_EX_z * (-pz) +
                            el_EX * (math::pow(-px, 2) + math::pow(-py, 2) +
                                     math::pow(-pz, 2)))) +
                 (-py) * (kl_EX_x * (-px) + kl_EX_z * (-pz)) * gamma +
                 kl_EX_y * (math::pow(-px, 2) + math::pow(-pz, 2) +
                            math::pow(-py, 2) * gamma)) /
                (math::pow(-px, 2) + math::pow(-py, 2) + math::pow(-pz, 2));
              auto w_el_new =
                (-((-pz) * (kl_EX_x * (-px) + kl_EX_y * (-py) +
                            el_EX * (math::pow(-px, 2) + math::pow(-py, 2) +
                                     math::pow(-pz, 2)))) +
                 (kl_EX_x * (-px) + kl_EX_y * (-py)) * (-pz) * gamma +
                 kl_EX_z * (math::pow(-px, 2) + math::pow(-py, 2) +
                            math::pow(-pz, 2) * gamma)) /
                (math::pow(-px, 2) + math::pow(-py, 2) + math::pow(-pz, 2));

              species.ux1(p) = u_el_new;
              species.ux2(p) = v_el_new;
              species.ux3(p) = w_el_new;

              // Boost photon momentum to the excitation frame
              auto pdotk = kph_RF_x * (-pel_ex_x) + kph_RF_y * (-pel_ex_y) +
                           kph_RF_z * (-pel_ex_z);
              auto eph_EX   = gamma_ex * eph_RF - pdotk;
              auto kph_EX_x = kph_RF_x +
                              (pdotk / (1.0 + gamma_ex) - eph_RF) * (-pel_ex_x);
              auto kph_EX_y = kph_RF_y +
                              (pdotk / (1.0 + gamma_ex) - eph_RF) * (-pel_ex_y);
              auto kph_EX_z = kph_RF_z +
                              (pdotk / (1.0 + gamma_ex) - eph_RF) * (-pel_ex_z);

              // Boost photon momentum to the lab frame
              pdotk    = kph_EX_x * (-px) + kph_EX_y * (-py) + kph_EX_z * (-pz);
              auto eph = gamma * eph_EX - pdotk;
              auto kph_x = kph_EX_x + (pdotk / (1.0 + gamma) - eph_EX) * (-px);
              auto kph_y = kph_EX_y + (pdotk / (1.0 + gamma) - eph_EX) * (-py);
              auto kph_z = kph_EX_z + (pdotk / (1.0 + gamma) - eph_EX) * (-pz);

              const vec_t<Dim3> xi { i_di_to_Xi(species.i1(p), species.dx1(p)),
                                     i_di_to_Xi(species.i2(p), species.dx2(p)),
                                     species.phi(p) };
              coord_t<Dim3>     x_ph { ZERO };
              m_mblock.metric.x_Code2Sph(xi, x_ph);

              // Inject the scattered photon
              if ((eph > 2.1) && (x_ph[1] > angThres) && (x_ph[1] < (constant::PI - angThres))) {
                if (pol_par) {
                  auto ph_p = Kokkos::atomic_fetch_add(&ph_ind_par(), 1);
                  init_prtl_2d_i_di(photons_par,
                                    ph_offset_par + ph_p,
                                    species.i1(p),
                                    species.i2(p),
                                    species.dx1(p),
                                    species.dx2(p),
                                    kph_x,
                                    kph_y,
                                    kph_z,
                                    species.weight(p));
                } else {
                  auto ph_p = Kokkos::atomic_fetch_add(&ph_ind_perp(), 1);
                  init_prtl_2d_i_di(photons_perp,
                                    ph_offset_perp + ph_p,
                                    species.i1(p),
                                    species.i2(p),
                                    species.dx1(p),
                                    species.dx2(p),
                                    kph_x,
                                    kph_y,
                                    kph_z,
                                    species.weight(p));
                }
              }
            }

            random_pool.free_state(rand_gen);
          });

        auto ph_ind_h = Kokkos::create_mirror(ph_ind_par);
        Kokkos::deep_copy(ph_ind_h, ph_ind_par);
        photons_par.setNpart(photons_par.npart() + ph_ind_h());

        ph_ind_h = Kokkos::create_mirror(ph_ind_perp);
        Kokkos::deep_copy(ph_ind_h, ph_ind_perp);
        photons_perp.setNpart(photons_perp.npart() + ph_ind_h());
      }
    }

    // Gamma-B pair production
    {
      const auto angThres { params.get<real_t>("problem", "angThres") };
      const auto bsurf { params.get<real_t>("problem", "psr_Bsurf", ONE) };
      const auto gamma_curv { params.get<real_t>("problem", "gamma_curv") };
      const auto gamma_curv_ph { params.get<real_t>("problem", "gamma_curv_ph") };
      const auto m_psr_omega { params.get<real_t>("problem", "psr_omega") };
      const auto           bq { params.get<real_t>("problem", "bq") };
      const auto           AbsCoeff { 0.23 * (27.0 / 8.0) * bsurf / bq *
                            (m_mblock.timestep() / params.skindepth0()) *
                            (m_psr_omega * 0.1) * math::sqrt(params.sigma0()) 
                            * SQR(CUBE(gamma_curv_ph)) / SQR(SQR(gamma_curv)) };
      auto&                electrons = m_mblock.particles[4];
      auto&                positrons = m_mblock.particles[5];
      const auto           minRad    = m_psr_Rstar + m_atm_h;

      const short buff_idx = 0;
      const short smooth   = 0;
      m_mblock.ComputeMoments(params, FieldID::N, {}, { 1, 2, 5, 6 }, buff_idx, smooth);

      for (std::size_t s { 0 }; s < 6; ++s) {
        if ((s == 0) || (s == 1) || (s == 4) || (s == 5)) {
          continue;
        }
        array_t<std::size_t> elec_ind("elec_ind");
        array_t<std::size_t> pos_ind("pos_ind");
        auto&      photons     = m_mblock.particles[s];
        auto&      payload1    = photons.pld[0];
        const auto elec_offset = electrons.npart();
        const auto pos_offset  = positrons.npart();

        Kokkos::parallel_for(
          "Photons",
          photons.rangeActiveParticles(),
          Lambda(index_t p) {
            if (photons.tag(p) == ParticleTag::alive) {
              const vec_t<Dim3> xi { i_di_to_Xi(photons.i1(p), photons.dx1(p)),
                                     i_di_to_Xi(photons.i2(p), photons.dx2(p)),
                                     photons.phi(p) };
              coord_t<Dim3>     x_ph { ZERO };
              m_mblock.metric.x_Code2Sph(xi, x_ph);
              if (x_ph[0] < minRad) {
                photons.tag(p) = ParticleTag::dead;
              } else if (photons.tag(p) != ParticleTag::dead) {
                // Interpolate magnetic field to the photon position
                vec_t<Dim3> b_int_Cart;
                {
                  vec_t<Dim3> b_int;
                  real_t      c000, c100, c010, c110, c00, c10;

                  const auto   i { photons.i1(p) + N_GHOSTS };
                  const real_t dx1 { photons.dx1(p) };
                  const auto   j { photons.i2(p) + N_GHOSTS };
                  const real_t dx2 { photons.dx2(p) };
                  // Bx1
                  c000     = HALF * (BX1(i, j) + BX1(i, j - 1));
                  c100     = HALF * (BX1(i + 1, j) + BX1(i + 1, j - 1));
                  c010     = HALF * (BX1(i, j) + BX1(i, j + 1));
                  c110     = HALF * (BX1(i + 1, j) + BX1(i + 1, j + 1));
                  c00      = c000 * (ONE - dx1) + c100 * dx1;
                  c10      = c010 * (ONE - dx1) + c110 * dx1;
                  b_int[0] = c00 * (ONE - dx2) + c10 * dx2;
                  // Bx2
                  c000     = HALF * (BX2(i - 1, j) + BX2(i, j));
                  c100     = HALF * (BX2(i, j) + BX2(i + 1, j));
                  c010     = HALF * (BX2(i - 1, j + 1) + BX2(i, j + 1));
                  c110     = HALF * (BX2(i, j + 1) + BX2(i + 1, j + 1));
                  c00      = c000 * (ONE - dx1) + c100 * dx1;
                  c10      = c010 * (ONE - dx1) + c110 * dx1;
                  b_int[1] = c00 * (ONE - dx2) + c10 * dx2;
                  // Bx3
                  c000     = INV_4 * (BX3(i - 1, j - 1) + BX3(i - 1, j) +
                                  BX3(i, j - 1) + BX3(i, j));
                  c100     = INV_4 * (BX3(i, j - 1) + BX3(i, j) +
                                  BX3(i + 1, j - 1) + BX3(i + 1, j));
                  c010     = INV_4 * (BX3(i - 1, j) + BX3(i - 1, j + 1) +
                                  BX3(i, j) + BX3(i, j + 1));
                  c110 = INV_4 * (BX3(i, j) + BX3(i, j + 1) + BX3(i + 1, j) +
                                  BX3(i + 1, j + 1));
                  c00  = c000 * (ONE - dx1) + c100 * dx1;
                  c10  = c010 * (ONE - dx1) + c110 * dx1;
                  b_int[2] = c00 * (ONE - dx2) + c10 * dx2;

                  const vec_t<Dim3> xp { static_cast<real_t>(photons.i1(p)) +
                                           static_cast<real_t>(photons.dx1(p)),
                                         static_cast<real_t>(photons.i2(p)) +
                                           static_cast<real_t>(photons.dx2(p)),
                                         photons.phi(p) };
                  m_mblock.metric.v3_Cntrv2Cart(xp, b_int, b_int_Cart);
                }

                //  Check for the angle of photon propagation with magnetic field
                auto babs { NORM(b_int_Cart[0], b_int_Cart[1], b_int_Cart[2]) };
                b_int_Cart[0] /= (babs + 1e-12);
                b_int_Cart[1] /= (babs + 1e-12);
                b_int_Cart[2] /= (babs + 1e-12);
                auto ePh { NORM(photons.ux1(p), photons.ux2(p), photons.ux3(p)) };
                auto       cosAngle { DOT(b_int_Cart[0],
                                    b_int_Cart[1],
                                    b_int_Cart[2],
                                    photons.ux1(p),
                                    photons.ux2(p),
                                    photons.ux3(p)) /
                                ePh };

                if (ePh <= 2){
                printf("ePh = %e\n", ePh);
                }

                auto       sinAngle { math::sqrt(ONE - SQR(cosAngle)) };
                const auto increment { AbsCoeff * babs / bq * sinAngle *
                                       math::exp(-8.0 / (3.0 * babs / bq *
                                                         sinAngle * ePh)) };
                payload1(p) += increment;

                // Check for pair production trigger
                if (payload1(p) >= ONE) {
                  photons.tag(p) = ParticleTag::dead;
                  const auto actual_ndens = m_mblock.buff(photons.i1(p) + N_GHOSTS,
                                                          photons.i2(p) + N_GHOSTS,
                                                          buff_idx);
                  auto upar {
                    math::abs(cosAngle) * math::sqrt(SQR(ePh) - FOUR) /
                    math::sqrt(SQR(ePh * sinAngle) + FOUR * SQR(cosAngle))
                  };
                  // const auto upar { 5.0 };

                  if((x_ph[1] > angThres) && (x_ph[1] < (constant::PI - angThres)) && actual_ndens < 0.1) {
                    auto elec_p = Kokkos::atomic_fetch_add(&elec_ind(), 1);
                    auto pos_p  = Kokkos::atomic_fetch_add(&pos_ind(), 1);
                    init_prtl_2d_i_di(electrons,
                                      elec_offset + elec_p,
                                      photons.i1(p),
                                      photons.i2(p),
                                      photons.dx1(p),
                                      photons.dx2(p),
                                      SIGN(cosAngle) * upar * b_int_Cart[0],
                                      SIGN(cosAngle) * upar * b_int_Cart[1],
                                      SIGN(cosAngle) * upar * b_int_Cart[2],
                                      photons.weight(p));

                    init_prtl_2d_i_di(positrons,
                                      pos_offset + pos_p,
                                      photons.i1(p),
                                      photons.i2(p),
                                      photons.dx1(p),
                                      photons.dx2(p),
                                      SIGN(cosAngle) * upar * b_int_Cart[0],
                                      SIGN(cosAngle) * upar * b_int_Cart[1],
                                      SIGN(cosAngle) * upar * b_int_Cart[2],
                                      photons.weight(p));
                  }
                }
              }
            }
          });

        auto elec_ind_h = Kokkos::create_mirror(elec_ind);
        Kokkos::deep_copy(elec_ind_h, elec_ind);
        electrons.setNpart(electrons.npart() + elec_ind_h());

        auto pos_ind_h = Kokkos::create_mirror(pos_ind);
        Kokkos::deep_copy(pos_ind_h, pos_ind);
        positrons.setNpart(positrons.npart() + pos_ind_h());
      }
    }
  }
} // namespace ntt

#endif
