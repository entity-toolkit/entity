#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"
#include "utils/injector.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {

    inline ProblemGenerator(const SimulationParams& params) :
      m_psr_Rstar { params.get<real_t>("problem", "rstar") } {}

    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {
    }

    inline void UserInitParticles(const SimulationParams&,
                                  Meshblock<D, S>&) override {}


  private:
    const real_t    m_psr_Rstar;
  };

Inline void boost_photon(const real_t el,
                             const real_t ux,
                             const real_t uy,
                             const real_t uz,
                             const real_t gammab,
                             const real_t uxb,
                             const real_t uyb,
                             const real_t uzb,
                             real_t&      el1,
                             real_t&      ux1,
                             real_t&      uy1,
                             real_t&      uz1) {

      auto psq = uxb * uxb + uyb * uyb + uzb * uzb;
      el1      = -ux * uxb - uy * uyb - uz * uzb + el * gammab;
      ux1      = (-(uz * uxb * uzb) + ux * (pow(uyb, 2) + pow(uzb, 2)) -
             el * uxb * psq + uy * uxb * uyb * (-1 + gammab) +
             uxb * (ux * uxb + uz * uzb) * gammab) /
            psq;
      uy1 = (-(uyb * (ux * uxb + uz * uzb + el * psq)) +
             uyb * (ux * uxb + uz * uzb) * gammab +
             uy * (pow(uxb, 2) + pow(uzb, 2) + pow(uyb, 2) * gammab)) /
            psq;
      uz1 = (-(uzb * (ux * uxb + uy * uyb + el * psq)) +
             (ux * uxb + uy * uyb) * uzb * gammab +
             uz * (pow(uxb, 2) + pow(uyb, 2) + pow(uzb, 2) * gammab)) /
            psq;
    }

    Inline void boost_lepton(const real_t el,
                             const real_t ux,
                             const real_t uy,
                             const real_t uz,
                             const real_t gammab,
                             const real_t uxb,
                             const real_t uyb,
                             const real_t uzb,
                             real_t&      el1,
                             real_t&      ux1,
                             real_t&      uy1,
                             real_t&      uz1) {

      auto psq = uxb * uxb + uyb * uyb + uzb * uzb;
      el1      = -ux * uxb - uy * uyb - uz * uzb + el * gammab;
      ux1      = (-(uz * uxb * uzb) + ux * (pow(uyb, 2) + pow(uzb, 2)) -
             el * uxb * psq + uy * uxb * uyb * (-1 + gammab) +
             uxb * (ux * uxb + uz * uzb) * gammab) /
            psq;
      uy1 = (-(uyb * (ux * uxb + uz * uzb + el * psq)) +
             uyb * (ux * uxb + uz * uzb) * gammab +
             uy * (pow(uxb, 2) + pow(uzb, 2) + pow(uyb, 2) * gammab)) /
            psq;
      uz1 = (-(uzb * (ux * uxb + uy * uyb + el * psq)) +
             (ux * uxb + uy * uyb) * uzb * gammab +
             uz * (pow(uxb, 2) + pow(uyb, 2) + pow(uzb, 2) * gammab)) /
            psq;
    }

    Inline auto planckSample(RandomNumberPool_t::generator_type rand_gen)->real_t {
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

  Inline void mainBField(const coord_t<Dim1>& x_ph,
                         vec_t<Dim3>&,
                         vec_t<Dim3>& b_out,
                         real_t       _rstar,
                         real_t       _bsurf) {
      b_out[0] = _bsurf / CUBE(x_ph[0]/_rstar);
      b_out[1] = ZERO;
      b_out[2] = ZERO;
  }


  template <>
  inline void ProblemGenerator<Dim1, PICEngine>::UserInitFields(
          const SimulationParams&     params,
      Meshblock<Dim1, PICEngine>& mblock) {

      Kokkos::parallel_for(
        "UserInitFields",
        mblock.rangeActiveCells(),
        ClassLambda(index_t i) {
          set_em_fields_1d(mblock, i, mainBField, m_psr_Rstar, ONE);
        });
  }

    template <>
    inline void ProblemGenerator<Dim1, PICEngine>::UserInitParticles(
      const SimulationParams&     params,
      Meshblock<Dim1, PICEngine>& mblock) {
      auto&        prtls = mblock.particles[0];
      auto prtls_idx    = array_t<std::size_t>("lecs_idx");
      auto prtls_offset = prtls.npart();
      const auto m_psr_Rstar { params.get<real_t>("problem", "rstar") };

      Kokkos::parallel_for(
        "UserInitParticles",
        1,
        Lambda(index_t) {
          InjectParticle_1D(mblock, prtls, prtls_idx, prtls_offset, m_psr_Rstar, 1000.0, ZERO, ZERO, ONE);
        });

    auto prtls_idx_h = Kokkos::create_mirror_view(prtls_idx);
    Kokkos::deep_copy(prtls_idx_h, prtls_idx);
    prtls.setNpart(prtls.npart() + prtls_idx_h());

  }

    template <>
    inline void ProblemGenerator<Dim1, PICEngine>::UserDriveParticles(
      const real_t& time,
      const SimulationParams&     params,
      Meshblock<Dim1, PICEngine>& m_mblock) {

      // Resonant scattering kernel
      {
        const auto fid_freq { params.get<real_t>("problem", "fid_freq") };
        const auto bq { params.get<real_t>("problem", "bq") };
        const auto m_psr_Rstar { params.get<real_t>("problem", "rstar") };

        auto       random_pool  = *(m_mblock.random_pool_ptr);

        for (std::size_t s { 0 }; s < 1; ++s) {
          auto&                species = m_mblock.particles[s];
          const auto           dt = m_mblock.timestep();

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
                real_t      c0, c1;

                const auto   i { species.i1(p) + N_GHOSTS };
                const real_t dx1 { species.dx1(p) };

                // Bx1
                c0    = BX1(i);
                c1    = BX1(i + 1);
                b_int[0] = c0 * (ONE - dx1) + c1 * dx1;
                // Bx2
                c0    = HALF * (BX2(i - 1) + BX2(i));
                c1    = HALF * (BX2(i) + BX2(i + 1));
                b_int[1] = c0 * (ONE - dx1) + c1 * dx1;
                // Bx3
                c0    = HALF * (BX3(i - 1) + BX3(i));
                c1    = HALF * (BX3(i) + BX3(i + 1));
                b_int[2] = c0 * (ONE - dx1) + c1 * dx1;

                const vec_t<Dim1> xp { static_cast<real_t>(species.i1(p)) +
                                         static_cast<real_t>(species.dx1(p)) };
                m_mblock.metric.v3_Cntrv2Cart(xp, b_int, b_int_Cart);
              }

              // Interpolation of E to the position of the photon
              vec_t<Dim3> e_int_Cart;
              {
                vec_t<Dim3> e_int;
                real_t      c0, c1;

                const auto   i { species.i1(p) + N_GHOSTS };
                const real_t dx1 { species.dx1(p) };

                // Ex1
                // interpolate to nodes
                c0    = HALF * (EX1(i) + EX1(i - 1));
                c1    = HALF * (EX1(i) + EX1(i + 1));
                // interpolate from nodes to the particle position
                e_int[0] = c0 * (ONE - dx1) + c1 * dx1;
                // Ex2
                c0    = EX2(i);
                c1    = EX2(i + 1);
                e_int[1] = c0 * (ONE - dx1) + c1 * dx1;
                // Ex3
                c0    = EX3(i);
                c1    = EX3(i + 1);
                e_int[2] = c0 * (ONE - dx1) + c1 * dx1;

                const vec_t<Dim1> xp { static_cast<real_t>(species.i1(p)) +
                                         static_cast<real_t>(species.dx1(p))};
                m_mblock.metric.v3_Cntrv2Cart(xp, e_int, e_int_Cart);
              }

              // Check lepton position and exclude those inside the atmosphere
              const vec_t<Dim1> xi { i_di_to_Xi(species.i1(p), species.dx1(p)) };
              coord_t<Dim1>     x_ph;
              m_mblock.metric.x_Code2Cart(xi, x_ph);

              // Define lepton properties for evaluation
              auto px      = species.ux1(p);
              auto py      = species.ux2(p);
              auto pz      = species.ux3(p);
              auto gamma   = math::sqrt(ONE + SQR(px) + SQR(py) + SQR(pz));
              auto betax   = px / gamma;
              auto betay   = py / gamma;
              auto betaz   = pz / gamma;
              auto beta_sq = SQR(betax) + SQR(betay) + SQR(betaz);

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
              auto norm { 1.0 / NORM(b_int_Cart[0], b_int_Cart[1],
              b_int_Cart[2]) }; auto a_RF_x = b_int_Cart[0] * norm; auto
              a_RF_y = b_int_Cart[1] * norm; auto a_RF_z = b_int_Cart[2] *
              norm; auto b_RF_x = 1.0; auto b_RF_y = 0.0; auto b_RF_z = 0.0;
              if (a_RF_x != ZERO) {
                b_RF_x = -a_RF_y / a_RF_x;
                b_RF_y = 1.0;
                b_RF_z = 0.0;
                norm   = 1.0 / sqrt(b_RF_x * b_RF_x + b_RF_y * b_RF_y);
                b_RF_x = b_RF_x * norm;
                b_RF_y = b_RF_y * norm;
              }
              auto c_RF_x = b_RF_z * a_RF_y - b_RF_y * a_RF_z;
              auto c_RF_y = b_RF_x * a_RF_z - b_RF_z * a_RF_x;
              auto c_RF_z = b_RF_y * a_RF_x - b_RF_x * a_RF_y;

              // Monte Carlo evaluation of the cross section
              auto   tres  = 0.0;
              auto   qdist = 0.0;
              auto   vmax  = 0.0;
              auto   vmin  = 100000.0;
              auto   nmax  = 1000;
              real_t eph_LF_L, eph_RF_L, u_ph_RF_L, v_ph_RF_L, w_ph_RF_L, u_ph_L,
                v_ph_L, w_ph_L, boostfactor;

              // for (std::size_t n { 0 }; n < nmax; ++n) {

              //   // Calculate the photon energy and momentum
                typename RandomNumberPool_t::generator_type rand_gen =
                  random_pool.get_state();
              //   auto eph_LF = fid_freq * planckSample(rand_gen);
                // auto eph_LF = fid_freq;

                // // Here fix the direction of photon momentum (e.g., spherical from
                // // star, isotropic, etc.) Isotropic photon distribution
                // auto rand_costheta_RF = math::cos(120./180. * Random<real_t>(rand_gen) * M_PI);
                // // auto rand_costheta_RF = math::cos(120./180. * M_PI);
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

                // // Boost photon into the lepton rest frame
                // real_t eph_RF, u_ph_RF, v_ph_RF, w_ph_RF;
                // boost_photon(eph_LF,
                //              u_ph,
                //              v_ph,
                //              w_ph,
                //              gamma,
                //              px,
                //              py,
                //              pz,
                //              eph_RF,
                //              u_ph_RF,
                //              v_ph_RF,
                //              w_ph_RF);

                // auto bbq = math::sqrt(bx0_rest * bx0_rest + by0_rest * by0_rest + bz0_rest * bz0_rest) / bq; 
                // auto omegares = bbq / (gamma * (1.0 - sqrt(beta_sq) * rand_costheta_RF));

                // // Calculate the resonance quality factor
                // auto xres = (eph_LF / omegares - 1.0);

                // auto qres = 0.0;
                // auto gfac = 0.001;
                // auto tpeak = fid_freq / 2.821;
                // if (fabs(xres) < 10.0 * gfac) {
                //   qres = SQR(eph_LF) / omegares * math::exp(-0.5 * xres * xres / ((gfac) * (gfac))) /
                //          sqrt(2.0 * M_PI * ((gfac) * (gfac))) * 1.0 / (math::exp(eph_LF/tpeak) - 1.0);
                // }

                // tres += qres;

                // if (qres > qdist) {
                //   eph_LF_L    = eph_LF;
                //   eph_RF_L    = eph_RF;
                //   u_ph_RF_L   = u_ph_RF;
                //   v_ph_RF_L   = v_ph_RF;
                //   w_ph_RF_L   = w_ph_RF;
                //   qdist       = qres;
                //   u_ph_L      = u_ph;
                //   v_ph_L      = v_ph;
                //   w_ph_L      = w_ph;
                //   boostfactor = (1.0 - sqrt(beta_sq) * rand_costheta_RF);
                // }

                // if (xres > vmax) {
                //   vmax = xres;
                // }

                // if (xres < vmin) {
                //   vmin = xres;
                // }

              //   random_pool.free_state(rand_gen);
              // }  
              

              // Calculate cross section: probability for scattering event (TODO: add dt dependence)
              // auto p_scatter = tres / static_cast<real_t>(nmax) * dt * 1000000000000.0 * 
              //                  (vmax - vmin) * sqrt(beta_sq) / gamma * SQR(m_psr_Rstar/x_ph[0]);

                // auto rand_costheta_RF = math::cos(120./180. * Random<real_t>(rand_gen) * M_PI);
                auto rand_costheta_RF = math::cos(120./180. * M_PI);
                auto bbq = math::sqrt(bx0_rest * bx0_rest + by0_rest * by0_rest + bz0_rest * bz0_rest) / bq; 
                auto eph_LF = bbq / (gamma * (1.0 - sqrt(beta_sq) * rand_costheta_RF));

                auto rand_sintheta_RF = math::sqrt(
                  1.0 - rand_costheta_RF * rand_costheta_RF);
                auto rand_phi_RF    = 2.0 * M_PI * Random<real_t>(rand_gen);
                auto rand_cosphi_RF = math::cos(rand_phi_RF);
                auto rand_sinphi_RF = math::sin(rand_phi_RF);
                auto u_ph           = eph_LF * (rand_costheta_RF * a_RF_x +
                                      rand_sintheta_RF * rand_cosphi_RF * b_RF_x +
                                      rand_sintheta_RF * rand_sinphi_RF * c_RF_x);
                auto v_ph           = eph_LF * (rand_costheta_RF * a_RF_y +
                                      rand_sintheta_RF * rand_cosphi_RF * b_RF_y +
                                      rand_sintheta_RF * rand_sinphi_RF * c_RF_y);
                auto w_ph           = eph_LF * (rand_costheta_RF * a_RF_z +
                                      rand_sintheta_RF * rand_cosphi_RF * b_RF_z +
                                      rand_sintheta_RF * rand_sinphi_RF * c_RF_z);

              real_t eph_RF, u_ph_RF, v_ph_RF, w_ph_RF;
                boost_photon(eph_LF,
                             u_ph,
                             v_ph,
                             w_ph,
                             gamma,
                             px,
                             py,
                             pz,
                             eph_RF,
                             u_ph_RF,
                             v_ph_RF,
                             w_ph_RF);

                  eph_LF_L    = eph_LF;
                  eph_RF_L    = eph_RF;
                  u_ph_RF_L   = u_ph_RF;
                  v_ph_RF_L   = v_ph_RF;
                  w_ph_RF_L   = w_ph_RF;
                  u_ph_L      = u_ph;
                  v_ph_L      = v_ph;
                  w_ph_L      = w_ph;

              auto tpeak = fid_freq / 2.821;
              auto ndot = 10000000000000.0 * SQR(m_psr_Rstar/x_ph[0]) * sqrt(beta_sq) / gamma
                            * SQR(eph_LF) / (math::exp(eph_LF/tpeak) - 1.0);
              auto p_scatter = dt * ndot;

              // Check if the photon scatters
              // typename RandomNumberPool_t::generator_type rand_gen =
              //   random_pool.get_state();
              if (Random<real_t>(rand_gen) < p_scatter) {

              // Make sure the photon has exact resonance energy (momentum according to the 'most' resonant one above)
              u_ph_RF_L = u_ph_RF_L / eph_RF_L *
                          math::sqrt(bx0_rest * bx0_rest + by0_rest * by0_rest +
                                     bz0_rest * bz0_rest) /
                          bq;
              v_ph_RF_L = v_ph_RF_L / eph_RF_L *
                          math::sqrt(bx0_rest * bx0_rest + by0_rest * by0_rest +
                                     bz0_rest * bz0_rest) /
                          bq;
              w_ph_RF_L = w_ph_RF_L / eph_RF_L *
                          math::sqrt(bx0_rest * bx0_rest + by0_rest * by0_rest +
                                     bz0_rest * bz0_rest) /
                          bq;

                // Calculate lepton properties after collision in excitation rest frame
                auto eb = math::sqrt(
                  1.0 + 2.0 *
                          math::sqrt(bx0_rest * bx0_rest + by0_rest * by0_rest +
                                     bz0_rest * bz0_rest) /
                          bq);
                auto gammaeb     = 1.0 + math::sqrt(u_ph_RF_L * u_ph_RF_L +
                                                v_ph_RF_L * v_ph_RF_L +
                                                w_ph_RF_L * w_ph_RF_L);
                auto gamma_ex    = gammaeb / eb;
                auto betax_ex    = u_ph_RF_L * 1.0 / gammaeb;
                auto betay_ex    = v_ph_RF_L * 1.0 / gammaeb;
                auto betaz_ex    = w_ph_RF_L * 1.0 / gammaeb;
                auto pel_ex_x    = gamma_ex * betax_ex;
                auto pel_ex_y    = gamma_ex * betay_ex;
                auto pel_ex_z    = gamma_ex * betaz_ex;
                auto betax_ex_sq = SQR(betax_ex) + SQR(betay_ex) + SQR(betaz_ex);

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
                  norm   = 1.0 / math::sqrt(b_RF_x * b_RF_x + b_RF_y *
                  b_RF_y); b_RF_x = b_RF_x * norm; b_RF_y = b_RF_y * norm;
                }
                auto c_RF_x = b_RF_z * a_RF_y - b_RF_y * a_RF_z;
                auto c_RF_y = b_RF_x * a_RF_z - b_RF_z * a_RF_x;
                auto c_RF_z = b_RF_y * a_RF_x - b_RF_x * a_RF_y;

                // Calculate emission vector for photon
                auto rand_phi_RF      = 2.0 * M_PI * Random<real_t>(rand_gen);
                auto rand_costheta_RF = mudash;
                auto rand_sintheta_RF = math::sqrt(1.0 - SQR(mudash));
                auto rand_cosphi_RF   = math::cos(rand_phi_RF);
                auto rand_sinphi_RF   = math::sin(rand_phi_RF);

                // Calculate rest frame energy and momentum of emitted photon
                auto eph_RFS = eb / (rand_sintheta_RF * rand_sintheta_RF) *
                               (1.0 - math::sqrt(
                                        (rand_costheta_RF * rand_costheta_RF) +
                                        (1.0 / (eb * eb)) *
                                          (rand_sintheta_RF * rand_sintheta_RF)));

                if (fabs(mudash) >= 1.0) {
                  eph_RFS = (-1 + SQR(eb)) / (2. * eb);
                }

                auto kph_RFS_x = eph_RFS *
                                 (rand_costheta_RF * a_RF_x +
                                  rand_sintheta_RF * rand_cosphi_RF * b_RF_x +
                                  rand_sintheta_RF * rand_sinphi_RF * c_RF_x);
                auto kph_RFS_y = eph_RFS *
                                 (rand_costheta_RF * a_RF_y +
                                  rand_sintheta_RF * rand_cosphi_RF * b_RF_y +
                                  rand_sintheta_RF * rand_sinphi_RF * c_RF_y);
                auto kph_RFS_z = eph_RFS *
                                 (rand_costheta_RF * a_RF_z +
                                  rand_sintheta_RF * rand_cosphi_RF * b_RF_z +
                                  rand_sintheta_RF * rand_sinphi_RF * c_RF_z);

                // Calculate the lepton energy and momentum after scattering
                auto el_RF   = eb - eph_RFS;
                auto kl_RF_x = -eph_RFS * rand_costheta_RF * a_RF_x;
                auto kl_RF_y = -eph_RFS * rand_costheta_RF * a_RF_y;
                auto kl_RF_z = -eph_RFS * rand_costheta_RF * a_RF_z;

                real_t el_EX, kl_EX_x, kl_EX_y, kl_EX_z;
                boost_lepton(el_RF,
                             kl_RF_x,
                             kl_RF_y,
                             kl_RF_z,
                             gamma_ex,
                             -pel_ex_x,
                             -pel_ex_y,
                             -pel_ex_z,
                             el_EX,
                             kl_EX_x,
                             kl_EX_y,
                             kl_EX_z);

                real_t gamma_el_new, u_el_new, v_el_new, w_el_new;
                boost_lepton(el_EX,
                             kl_EX_x,
                             kl_EX_y,
                             kl_EX_z,
                             gamma,
                             -px,
                             -py,
                             -pz,
                             gamma_el_new,
                             u_el_new,
                             v_el_new,
                             w_el_new);

                species.ux1(p) = u_el_new;
                species.ux2(p) = v_el_new;
                species.ux3(p) = w_el_new;

              }

              random_pool.free_state(rand_gen);
            });

        }
      }
      }

} // namespace ntt

#endif