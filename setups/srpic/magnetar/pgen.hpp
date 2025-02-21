#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t bsurf, real_t rstar) : Bsurf { bsurf }, Rstar { rstar } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return Bsurf * math::cos(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
      // return Bsurf * SQR(Rstar / x_Ph[0]);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return Bsurf * HALF * math::sin(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
      // return ZERO;
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
      auto sigma = (x_Ph[1] - HALF * constant::PI) /
                   (static_cast<real_t>(0.2) * constant::PI);
      return Omega * bx2(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]) * sigma *
             math::exp((ONE - SQR(SQR(sigma))) * INV_4);
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      auto sigma = (x_Ph[1] - HALF * constant::PI) /
                   (static_cast<real_t>(0.2) * constant::PI);
      return -Omega * bx1(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]) * sigma *
             math::exp((ONE - SQR(SQR(sigma))) * INV_4);
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t time, Omega;
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
    ux1 = (-(uz * uxb * uzb) + ux * (pow(uyb, 2) + pow(uzb, 2)) - el * uxb * psq +
           uy * uxb * uyb * (-1 + gammab) + uxb * (ux * uxb + uz * uzb) * gammab) /
          psq;
    uy1 = (-(uyb * (ux * uxb + uz * uzb + el * psq)) +
           uyb * (ux * uxb + uz * uzb) * gammab +
           uy * (pow(uxb, 2) + pow(uzb, 2) + pow(uyb, 2) * gammab)) /
          psq;
    uz1 = (-(uzb * (ux * uxb + uy * uyb + el * psq)) +
           (ux * uxb + uy * uyb) * uzb * gammab +
           uz * (pow(uxb, 2) + pow(uyb, 2) + pow(uzb, 2) * gammab)) /
          psq;
    // el1 = math::sqrt(ZERO + ux1 * ux1 + uy1 * uy1 + uz1 * uz1);

    // if (fabs(1 - el1 / math::sqrt(ZERO + ux1 * ux1 + uy1 * uy1 + uz1 * uz1)) > 1e-3) {
    //   printf("WARNING: Lepton energy in boost is not accurate.\n");
    // }
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
    ux1 = (-(uz * uxb * uzb) + ux * (pow(uyb, 2) + pow(uzb, 2)) - el * uxb * psq +
           uy * uxb * uyb * (-1 + gammab) + uxb * (ux * uxb + uz * uzb) * gammab) /
          psq;
    uy1 = (-(uyb * (ux * uxb + uz * uzb + el * psq)) +
           uyb * (ux * uxb + uz * uzb) * gammab +
           uy * (pow(uxb, 2) + pow(uzb, 2) + pow(uyb, 2) * gammab)) /
          psq;
    uz1 = (-(uzb * (ux * uxb + uy * uyb + el * psq)) +
           (ux * uxb + uy * uyb) * uzb * gammab +
           uz * (pow(uxb, 2) + pow(uyb, 2) + pow(uzb, 2) * gammab)) /
          psq;
    // el1 = math::sqrt(ONE + ux1 * ux1 + uy1 * uy1 + uz1 * uz1);

    // if (fabs(1.0 - el1 / math::sqrt(ONE + ux1 * ux1 + uy1 * uy1 + uz1 * uz1)) >
    //     1e-3) {
    //   printf("WARNING: Lepton energy in boost is not accurate.\n");
    // }
  }


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

    const Metadomain<S, M>& global_domain;

    const real_t  Bsurf, Rstar, Omega, fid_freq, bq, dt, inv_n0, gamma_pairs, pp_thres;
    InitFields<D> init_flds;
    bool          is_first_step;
    const bool    threshold_pp, rics_pp;
    
    array_t<real_t**> cbuff, cbuff2;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , is_first_step { true }
      , global_domain { m }
      , Bsurf { p.template get<real_t>("setup.Bsurf", ONE) }
      , Rstar { m.mesh().extent(in::x1).first }
      , Omega { p.template get<real_t>("setup.omega") }
      , inv_n0 {ONE / p.template get<real_t>("scales.n0")}
      , fid_freq { p.template get<real_t>("setup.fid_freq", ZERO) }
      , bq { p.template get<real_t>("setup.bq", ONE) }
      , pp_thres { p.template get<real_t>("setup.pp_thres") }
      , gamma_pairs { p.template get<real_t>("setup.gamma_pairs") }
      , dt { params.template get<real_t>("algorithms.timestep.dt") }
      , threshold_pp { p.template get<bool>("setup.threshold_pp") }
      , rics_pp { p.template get<bool>("setup.rics_pp") }
      , init_flds { Bsurf, Rstar } {
        Kokkos::deep_copy(cbuff, ZERO);
        Kokkos::deep_copy(cbuff2, ZERO);
      }

    inline PGen() {}

    auto FieldDriver(real_t time) const -> DriveFields<D> {
      const real_t omega_t =
        Omega *
        ((ONE - math::tanh((static_cast<real_t>(5.0) - time) * HALF)) *
         (ONE + (-ONE + math::tanh((static_cast<real_t>(45.0) - time) * HALF)) *
                  HALF)) *
        HALF;
        // Omega *
        // ((ONE + (-ONE + math::tanh((static_cast<real_t>(40.0) - time) * HALF)) *
        //           HALF));
      return DriveFields<D> { time, Bsurf, Rstar, omega_t };
    }

        void CustomPostStep(std::size_t time, long double, Domain<S, M>& domain) {
      
      if (is_first_step) {
        cbuff = array_t<real_t**>("cbuff",
                                  domain.mesh.n_all(in::x1),
                                  domain.mesh.n_all(in::x2));
        Kokkos::deep_copy(cbuff, ZERO);
        cbuff2 = array_t<real_t**>("cbuff2",
                                  domain.mesh.n_all(in::x1),
                                  domain.mesh.n_all(in::x2));
        Kokkos::deep_copy(cbuff2, ZERO);
      }

    // Ad-hoc PP kernel
    if (threshold_pp) {

      // auto& species2_e = domain.species[2];
      // auto& species2_p = domain.species[3];
        auto& species2_e = domain.species[4];
        auto& species2_p = domain.species[5];
        auto& species3_e = domain.species[4];
        auto& species3_p = domain.species[5];
        auto metric = domain.mesh.metric;
        auto cbuff_sc = Kokkos::Experimental::create_scatter_view(cbuff);
        auto cbuff2_sc = Kokkos::Experimental::create_scatter_view(cbuff2);
        auto pp_thres_ = this->pp_thres;
        auto gamma_pairs_ = this->gamma_pairs;
        auto inv_n0_ = this->inv_n0;

         for (std::size_t s { 0 }; s < 6; ++s) {
          // if (s == 1) {
            if (s == 1 || s == 2 || s == 3) {
              continue;
            }

            array_t<std::size_t> elec_ind("elec_ind");
            array_t<std::size_t> pos_ind("pos_ind");
              
            auto offset_e = species3_e.npart();
            auto offset_p = species3_p.npart();

            auto ux1_e    = species3_e.ux1;
            auto ux2_e    = species3_e.ux2;
            auto ux3_e    = species3_e.ux3;
            auto i1_e     = species3_e.i1;
            auto i2_e     = species3_e.i2;
            auto dx1_e    = species3_e.dx1;
            auto dx2_e    = species3_e.dx2;
            auto phi_e    = species3_e.phi;
            auto weight_e = species3_e.weight;
            auto tag_e    = species3_e.tag;

            auto ux1_p    = species3_p.ux1;
            auto ux2_p    = species3_p.ux2;
            auto ux3_p    = species3_p.ux3;
            auto i1_p     = species3_p.i1;
            auto i2_p     = species3_p.i2;
            auto dx1_p    = species3_p.dx1;
            auto dx2_p    = species3_p.dx2;
            auto phi_p    = species3_p.phi;
            auto weight_p = species3_p.weight;
            auto tag_p    = species3_p.tag;

            if ((s == 0) || (s == 1)) {

              offset_e = species2_e.npart();
              offset_p = species2_p.npart();

              ux1_e    = species2_e.ux1;
              ux2_e    = species2_e.ux2;
              ux3_e    = species2_e.ux3;
              i1_e     = species2_e.i1;
              i2_e     = species2_e.i2;
              dx1_e    = species2_e.dx1;
              dx2_e    = species2_e.dx2;
              phi_e    = species2_e.phi;
              weight_e = species2_e.weight;
              tag_e    = species2_e.tag;

              ux1_p    = species2_p.ux1;
              ux2_p    = species2_p.ux2;
              ux3_p    = species2_p.ux3;
              i1_p     = species2_p.i1;
              i2_p     = species2_p.i2;
              dx1_p    = species2_p.dx1;
              dx2_p    = species2_p.dx2;
              phi_p    = species2_p.phi;
              weight_p = species2_p.weight;
              tag_p    = species2_p.tag;

            }

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

    Kokkos::parallel_for(
        "InjectPairs", species.rangeActiveParticles(), Lambda(index_t p) {
          if (tag(p) == ParticleTag::dead) {
            return;
          }

            auto px      = ux1(p);
            auto py      = ux2(p);
            auto pz      = ux3(p);
            auto gamma   = math::sqrt(ONE + SQR(px) + SQR(py) + SQR(pz));

            const coord_t<D> xCd{
                static_cast<real_t>(i1(p)) + dx1(p),
                static_cast<real_t>(i2(p)) + dx2(p)};

            coord_t<D> xPh { ZERO };
            metric.template convert<Crd::Cd, Crd::Ph>(xCd, xPh);

          if ((gamma > pp_thres_) && (math::sin(xPh[1]) > 0.1)
          && (xPh[0] < 15.0)) {

            auto new_gamma = gamma - 2.0 * gamma_pairs_;
            auto new_fac = math::sqrt(SQR(new_gamma) - 1.0) / math::sqrt(SQR(gamma) - 1.0);
            auto pair_fac = math::sqrt(SQR(gamma_pairs_) - 1.0) / math::sqrt(SQR(gamma) - 1.0);

            auto elec_p = Kokkos::atomic_fetch_add(&elec_ind(), 1);
            auto pos_p  = Kokkos::atomic_fetch_add(&pos_ind(), 1);

              i1_e(elec_p + offset_e) = i1(p);
              dx1_e(elec_p + offset_e) = dx1(p);
              i2_e(elec_p + offset_e) = i2(p);
              dx2_e(elec_p + offset_e) = dx2(p);
              phi_e(elec_p + offset_e) = phi(p);
              ux1_e(elec_p + offset_e) = px * pair_fac;
              ux2_e(elec_p + offset_e) = py * pair_fac;
              ux3_e(elec_p + offset_e) = pz * pair_fac;
              weight_e(elec_p + offset_e) = weight(p);
              tag_e(elec_p + offset_e) = ParticleTag::alive;

              i1_p(pos_p + offset_p) = i1(p);
              dx1_p(pos_p + offset_p) = dx1(p);
              i2_p(pos_p + offset_p) = i2(p);
              dx2_p(pos_p + offset_p) = dx2(p);
              phi_p(pos_p + offset_p) = phi(p);
              ux1_p(pos_p + offset_p) = px * pair_fac;
              ux2_p(pos_p + offset_p) = py * pair_fac;
              ux3_p(pos_p + offset_p) = pz * pair_fac;
              weight_p(pos_p + offset_p) = weight(p);
              tag_p(pos_p + offset_p) = ParticleTag::alive;

              ux1(p) *= new_fac;
              ux2(p) *= new_fac;
              ux3(p) *= new_fac;

              if ((s == 0) || (s == 1)) {
                auto cbuff_acc     = cbuff_sc.access();
                cbuff_acc(static_cast<int>(i1(p)), static_cast<int>(i2(p))) += weight(p) * inv_n0_ /
                    metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                                        static_cast<real_t>(i2(p)) + HALF });
              } else {
                auto cbuff2_acc     = cbuff2_sc.access();
                cbuff2_acc(static_cast<int>(i1(p)), static_cast<int>(i2(p))) += weight(p) * inv_n0_ /
                    metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                                        static_cast<real_t>(i2(p)) + HALF });
              }
          }

        });

            auto elec_ind_h = Kokkos::create_mirror(elec_ind);
            Kokkos::deep_copy(elec_ind_h, elec_ind);
            if ((s == 0) || (s == 1)) {
              species2_e.set_npart(offset_e + elec_ind_h());
            } else {
              species3_e.set_npart(offset_e + elec_ind_h());
            }

            // species_e.set_npart(offset_e + elec_ind_h());

            auto pos_ind_h = Kokkos::create_mirror(pos_ind);
            Kokkos::deep_copy(pos_ind_h, pos_ind);
            if ((s == 0) || (s == 1)) {
              species2_p.set_npart(offset_p + pos_ind_h());
            } else {
              species3_p.set_npart(offset_p + pos_ind_h());
            }

            // species_p.set_npart(offset_p + pos_ind_h());

          }

        Kokkos::Experimental::contribute(cbuff, cbuff_sc);
        Kokkos::Experimental::contribute(cbuff2, cbuff2_sc);
        } // Ad-hoc PP kernel

     // Resonant scattering kernel
    if (rics_pp) {
      {
      auto random_pool    = domain.random_pool;
      auto& photons_par   = domain.species[2];
      auto& photons_perp  = domain.species[3];
      auto EB             = domain.fields.em;
      auto metric         = domain.mesh.metric;
      auto Rstar_          = this->Rstar;
      auto bq_            = this->bq;
      auto dt_            = this->dt;
      auto fid_freq_      = this->fid_freq;
      auto cbuff2_sc = Kokkos::Experimental::create_scatter_view(cbuff2);
      auto inv_n0_      = this->inv_n0;

         for (std::size_t s { 0 }; s < 6; ++s) {
            // if (s == 2 || s == 3) {
            if (s == 1 || s == 2 || s == 3) {
            // if (s == 1) {
              continue;
            }

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

            array_t<std::size_t> ph_ind_par("ph_ind_par");
            const auto  ph_offset_par  = photons_par.npart();

            auto ux1_par    = photons_par.ux1;
            auto ux2_par    = photons_par.ux2;
            auto ux3_par    = photons_par.ux3;
            auto i1_par     = photons_par.i1;
            auto i2_par     = photons_par.i2;
            auto dx1_par    = photons_par.dx1;
            auto dx2_par    = photons_par.dx2;
            auto phi_par    = photons_par.phi;
            auto weight_par = photons_par.weight;
            auto tag_par    = photons_par.tag;

            array_t<std::size_t> ph_ind_perp("ph_ind_perp");
            const auto  ph_offset_perp = photons_perp.npart();

            auto ux1_perp    = photons_perp.ux1;
            auto ux2_perp    = photons_perp.ux2;
            auto ux3_perp    = photons_perp.ux3;
            auto i1_perp     = photons_perp.i1;
            auto i2_perp     = photons_perp.i2;
            auto dx1_perp    = photons_perp.dx1;
            auto dx2_perp    = photons_perp.dx2;
            auto phi_perp    = photons_perp.phi;
            auto weight_perp = photons_perp.weight;
            auto tag_perp    = photons_perp.tag;

    Kokkos::parallel_for(
        "ScatterPhotons", species.rangeActiveParticles(), Lambda(index_t p) {
          if (tag(p) == ParticleTag::dead) {
            return;
          }

            // Get particle coordinates for later processing
            const coord_t<Dim::_3D> xc3d {static_cast<real_t>(i1(p)) + dx1(p),
                  static_cast<real_t>(i2(p)) + dx2(p), phi(p)};    
            const coord_t<Dim::_2D> xc2d{static_cast<real_t>(i1(p)) + dx1(p),
                  static_cast<real_t>(i2(p)) + dx2(p)};               
            coord_t<Dim::_2D> xPh { ZERO };
            metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);

            // If particle is too close to atmosphere, skip (saving time)
            if (xPh[0] < Rstar_ + 0.1) return;
            if(math::sin(xPh[1]) < 0.1) return;

            // Define lepton properties for evaluation
            auto px      = ux1(p);
            auto py      = ux2(p);
            auto pz      = ux3(p);
            auto gamma   = math::sqrt(ONE + SQR(px) + SQR(py) + SQR(pz));
            auto betax   = px / gamma;
            auto betay   = py / gamma;
            auto betaz   = pz / gamma;
            auto beta_sq = SQR(betax) + SQR(betay) + SQR(betaz);

            // If particle is not relativistic, skip (testing for stopping)
            // if (gamma < 1.51186) return;
            // if (gamma < 1.0328) return;

            // Interpolation and conversion of electric and magnetic fields
            vec_t<Dim::_3D> b_int_Cart { ZERO };
            vec_t<Dim::_3D> e_int_Cart { ZERO };
            vec_t<Dim::_3D> b_int { ZERO };
            vec_t<Dim::_3D> e_int { ZERO };
            
            real_t      c000, c100, c010, c110, c00, c10;
            const auto   i { i1(p) + N_GHOSTS };
            const real_t dx1_ { dx1(p) };
            const auto   j { i2(p) + N_GHOSTS };
            const real_t dx2_ { dx2(p) };

            // Bx1
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

            metric.template transform_xyz<Idx::U, Idx::XYZ>(xc3d, b_int, b_int_Cart);
            metric.template transform_xyz<Idx::U, Idx::XYZ>(xc3d, e_int, e_int_Cart);

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


            coord_t<Dim::_3D>     x_cart { ZERO };
            metric.template convert_xyz<Crd::Cd, Crd::XYZ>(xc3d, x_cart);
            auto xnorm { 1.0 / NORM(x_cart[0], x_cart[1], x_cart[2]) };
            auto x1norm = x_cart[0] * xnorm;
            auto x2norm = x_cart[1] * xnorm;
            auto x3norm = x_cart[2] * xnorm;
            auto rand_costheta_RF { DOT(px, py, pz, x1norm, x2norm, x3norm) /
                                    NORM(px, py, pz) };

            auto bbq = math::sqrt(bx0_rest * bx0_rest + by0_rest * by0_rest
            + bz0_rest * bz0_rest) / bq_; auto eph_LF = bbq / (gamma * (1.0
            - sqrt(beta_sq) * rand_costheta_RF));

            auto u_ph   = eph_LF * x1norm;
            auto v_ph   = eph_LF * x2norm;
            auto w_ph   = eph_LF * x3norm;
            eph_LF = math::sqrt(SQR(u_ph) + SQR(v_ph) + SQR(w_ph));

            // Boost photon into the lepton rest frame
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

            real_t eph_LF_L, eph_RF_L, u_ph_RF_L, v_ph_RF_L, w_ph_RF_L, u_ph_L, v_ph_L, w_ph_L;
                eph_LF_L    = eph_LF;
                eph_RF_L    = eph_RF;
                u_ph_RF_L   = u_ph_RF;
                v_ph_RF_L   = v_ph_RF;
                w_ph_RF_L   = w_ph_RF;
                u_ph_L      = u_ph;
                v_ph_L      = v_ph;
                w_ph_L      = w_ph;

            auto tpeak = fid_freq_ / 2.821;

            // if (gamma < 2.0) {
            // // if (true) {
              
            //   auto pnorm { NORM(px, py, pz) };
            //   if (pnorm < 0.001) return;

            //   auto f_drag = 1.41631 * math::pow(10.0, 18) * SQR(Rstar_/xPh[0]) * (rand_costheta_RF - math::sqrt(beta_sq)) * CUBE(eph_LF) / (math::exp(eph_LF/tpeak) - 1.0);
            //   auto dp = dt_ * f_drag;

            //   auto pplus = pnorm * ( ONE + dp );
            //   auto beta_plus = pplus / math::sqrt(1.0 + SQR(pplus));

            //   if ((rand_costheta_RF - math::sqrt(beta_sq)) * (rand_costheta_RF - beta_plus) < 0.0) {
            //     auto beta_star = rand_costheta_RF;
            //     auto gamma_star = 1.0 / math::sqrt(1.0 - SQR(beta_star));
            //     auto p_star = gamma_star * beta_star;
            //     ux1(p) = px * p_star / pnorm;
            //     ux2(p) = py * p_star / pnorm;
            //     ux3(p) = pz * p_star / pnorm;
            //   } else {
            //     ux1(p) += dp * ux1(p);
            //     ux2(p) += dp * ux2(p);
            //     ux3(p) += dp * ux3(p);
            //   }
              
            //   return;
            // }

            auto ndot = 1.41631 * math::pow(10.0, 18) * SQR(Rstar_/xPh[0]) * 1.0 / gamma
                          * SQR(eph_LF) / (math::exp(eph_LF/tpeak) - 1.0);
            auto p_scatter = dt_ * ndot;

            auto  rand_gen = random_pool.get_state();

            // if (s == 1) {
            //   if (Random<real_t>(rand_gen) < 0.75) {
            //     random_pool.free_state(rand_gen);
            //     return;
            //   }
            // }

            if (Random<real_t>(rand_gen) < p_scatter) {

              // Make sure the photon has exact resonance energy (momentum according to the 'most' resonant one above)
              u_ph_RF_L = u_ph_RF_L / eph_RF_L *
                          math::sqrt(bx0_rest * bx0_rest + by0_rest * by0_rest +
                                    bz0_rest * bz0_rest) /
                          bq_;
              v_ph_RF_L = v_ph_RF_L / eph_RF_L *
                          math::sqrt(bx0_rest * bx0_rest + by0_rest * by0_rest +
                                    bz0_rest * bz0_rest) /
                          bq_;
              w_ph_RF_L = w_ph_RF_L / eph_RF_L *
                          math::sqrt(bx0_rest * bx0_rest + by0_rest * by0_rest +
                                    bz0_rest * bz0_rest) /
                          bq_;

              // Calculate lepton properties after collision in excitation rest frame
              auto eb = math::sqrt(
                1.0 + 2.0 *
                        math::sqrt(bx0_rest * bx0_rest + by0_rest * by0_rest +
                                   bz0_rest * bz0_rest) /
                        bq_);
              auto gammaeb     = 1.0 + math::sqrt(u_ph_RF_L * u_ph_RF_L +
                                              v_ph_RF_L * v_ph_RF_L +
                                              w_ph_RF_L * w_ph_RF_L);
              auto gamma_ex    = gammaeb / eb;
              auto betax_ex    = u_ph_RF_L / gammaeb;
              auto betay_ex    = v_ph_RF_L / gammaeb;
              auto betaz_ex    = w_ph_RF_L / gammaeb;
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

              ux1(p) = u_el_new;
              ux2(p) = v_el_new;
              ux3(p) = w_el_new;

              real_t eph_EX, kph_EX_x, kph_EX_y, kph_EX_z;
              boost_photon(eph_RFS,
                           kph_RFS_x,
                           kph_RFS_y,
                           kph_RFS_z,
                           gamma_ex,
                           -pel_ex_x,
                           -pel_ex_y,
                           -pel_ex_z,
                           eph_EX,
                           kph_EX_x,
                           kph_EX_y,
                           kph_EX_z);

              real_t eph, kph_x, kph_y, kph_z;
              boost_photon(eph_EX,
                           kph_EX_x,
                           kph_EX_y,
                           kph_EX_z,
                           gamma,
                           -px,
                           -py,
                           -pz,
                           eph,
                           kph_x,
                           kph_y,
                           kph_z);

              // Inject the scattered photon
              if ((eph > 2.0)) {
                if (pol_par) {
                  auto ph_p = Kokkos::atomic_fetch_add(&ph_ind_par(), 1);
                  i1_par(ph_p + ph_offset_par) = i1(p);
                  dx1_par(ph_p + ph_offset_par) = dx1(p);
                  i2_par(ph_p + ph_offset_par) = i2(p);
                  dx2_par(ph_p + ph_offset_par) = dx2(p);
                  phi_par(ph_p + ph_offset_par) = phi(p);
                  ux1_par(ph_p + ph_offset_par) = kph_x;
                  ux2_par(ph_p + ph_offset_par) = kph_y;
                  ux3_par(ph_p + ph_offset_par) = kph_z;
                  weight_par(ph_p + ph_offset_par) = weight(p);
                  tag_par(ph_p + ph_offset_par) = ParticleTag::alive;
                } 
                else {
                  auto ph_pp = Kokkos::atomic_fetch_add(&ph_ind_perp(), 1);
                  i1_perp(ph_pp + ph_offset_perp) = i1(p);
                  dx1_perp(ph_pp + ph_offset_perp) = dx1(p);
                  i2_perp(ph_pp + ph_offset_perp) = i2(p);
                  dx2_perp(ph_pp + ph_offset_perp) = dx2(p);
                  phi_perp(ph_pp + ph_offset_perp) = phi(p);
                  ux1_perp(ph_pp + ph_offset_perp) = kph_x;
                  ux2_perp(ph_pp + ph_offset_perp) = kph_y;
                  ux3_perp(ph_pp + ph_offset_perp) = kph_z;
                  weight_perp(ph_pp + ph_offset_perp) = weight(p);
                  tag_perp(ph_pp + ph_offset_perp) = ParticleTag::alive;
                }

                // auto cbuff2_acc     = cbuff2_sc.access();
                // cbuff2_acc(static_cast<int>(i1(p)), static_cast<int>(i2(p))) += weight(p) * inv_n0_ /
                //     metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                //                         static_cast<real_t>(i2(p)) + HALF });

              }

            }

          random_pool.free_state(rand_gen);

         });

          auto ph_ind_par_h = Kokkos::create_mirror(ph_ind_par);
          Kokkos::deep_copy(ph_ind_par_h, ph_ind_par);
          photons_par.set_npart(ph_offset_par + ph_ind_par_h());

          auto ph_ind_perp_h = Kokkos::create_mirror(ph_ind_perp);
          Kokkos::deep_copy(ph_ind_perp_h, ph_ind_perp);
          photons_perp.set_npart(ph_offset_perp + ph_ind_perp_h());

         }

        // Kokkos::Experimental::contribute(cbuff2, cbuff2_sc);
      } // Resonant scattering kernel

      // Pair production kernel (threshold)
      {
        auto& species_e   = domain.species[4];
        auto& species_p   = domain.species[5];
        auto metric       = domain.mesh.metric;
        auto EB           = domain.fields.em;
        auto cbuff_sc = Kokkos::Experimental::create_scatter_view(cbuff);
        auto bq_          = this->bq;
        auto inv_n0_      = this->inv_n0;
        auto Rstar_       = this->Rstar;
        auto is_first_step_ = this->is_first_step;

         for (std::size_t s { 0 }; s < 6; ++s) {
            if ((s == 0) || (s == 1) || (s == 4) || (s == 5)) {
              continue;
            }

            array_t<std::size_t> elec_ind("elec_ind");
            array_t<std::size_t> pos_ind("pos_ind");
            auto offset_e = species_e.npart();
            auto offset_p = species_p.npart();

            auto ux1_e    = species_e.ux1;
            auto ux2_e    = species_e.ux2;
            auto ux3_e    = species_e.ux3;
            auto i1_e     = species_e.i1;
            auto i2_e     = species_e.i2;
            auto dx1_e    = species_e.dx1;
            auto dx2_e    = species_e.dx2;
            auto phi_e    = species_e.phi;
            auto weight_e = species_e.weight;
            auto tag_e    = species_e.tag;

            auto ux1_p    = species_p.ux1;
            auto ux2_p    = species_p.ux2;
            auto ux3_p    = species_p.ux3;
            auto i1_p     = species_p.i1;
            auto i2_p     = species_p.i2;
            auto dx1_p    = species_p.dx1;
            auto dx2_p    = species_p.dx2;
            auto phi_p    = species_p.phi;
            auto weight_p = species_p.weight;
            auto tag_p    = species_p.tag;

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

    Kokkos::parallel_for(
        "InjectPairs", species.rangeActiveParticles(), Lambda(index_t p) {
          if (tag(p) == ParticleTag::dead) {
            return;
          }

            // Get particle coordinates for later processing
            const coord_t<Dim::_3D> xc3d {static_cast<real_t>(i1(p)) + dx1(p),
                  static_cast<real_t>(i2(p)) + dx2(p), phi(p)};   

            const coord_t<D> xCd{
                static_cast<real_t>(i1(p)) + dx1(p),
                static_cast<real_t>(i2(p)) + dx2(p)};

            coord_t<D> xPh { ZERO };
            metric.template convert<Crd::Cd, Crd::Ph>(xCd, xPh);

            if (xPh[0] > 25.0*Rstar_) {
              tag(p) = ParticleTag::dead;
              return;
            }

            if(math::sin(xPh[1]) < 0.1) return;

            // Interpolation and conversion of electric and magnetic fields
            vec_t<Dim::_3D> b_int_Cart { ZERO };
            vec_t<Dim::_3D> b_int { ZERO };
            
            real_t      c000, c100, c010, c110, c00, c10;
            const auto   i { i1(p) + N_GHOSTS };
            const real_t dx1_ { dx1(p) };
            const auto   j { i2(p) + N_GHOSTS };
            const real_t dx2_ { dx2(p) };

            // Bx1
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

            //  Check for the angle of photon propagation with magnetic field
            auto babs { NORM(b_int_Cart[0], b_int_Cart[1], b_int_Cart[2]) };
            b_int_Cart[0] /= (babs + 1e-12);
            b_int_Cart[1] /= (babs + 1e-12);
            b_int_Cart[2] /= (babs + 1e-12);
            auto ePh { NORM(ux1(p), ux2(p), ux3(p)) };
            auto cosAngle { DOT(b_int_Cart[0],
                                b_int_Cart[1],
                                b_int_Cart[2],
                                ux1(p),
                                ux2(p),
                                ux3(p)) /
                            ePh };
            auto sinAngle { math::sqrt(ONE - SQR(cosAngle)) };

            // auto ethres { 2.0 / math::abs(sinAngle + 1e-12) };
            // if (s == 3) {
            //   ethres *= math::sqrt(1.0 + 2.0 * babs / bq_);
            // }
            auto ethres { 2.0 };


              // Check for pair production trigger
              if (ePh >= ethres) {

                tag(p) = ParticleTag::dead;

                auto upar { math::abs(cosAngle) * math::sqrt(SQR(ePh) - FOUR) /
                            math::sqrt(SQR(ePh * sinAngle) + FOUR * SQR(cosAngle)) };

              auto elec_p = Kokkos::atomic_fetch_add(&elec_ind(), 1);
              i1_e(elec_p + offset_e) = i1(p);
              dx1_e(elec_p + offset_e) = dx1(p);
              i2_e(elec_p + offset_e) = i2(p);
              dx2_e(elec_p + offset_e) = dx2(p);
              phi_e(elec_p + offset_e) = phi(p);
              ux1_e(elec_p + offset_e) = SIGN(cosAngle) * upar * b_int_Cart[0];
              ux2_e(elec_p + offset_e) = SIGN(cosAngle) * upar * b_int_Cart[1];
              ux3_e(elec_p + offset_e) = SIGN(cosAngle) * upar * b_int_Cart[2];
              weight_e(elec_p + offset_e) = weight(p);
              tag_e(elec_p + offset_e) = ParticleTag::alive;

              auto pos_p  = Kokkos::atomic_fetch_add(&pos_ind(), 1);
              i1_p(pos_p + offset_p) = i1(p);
              dx1_p(pos_p + offset_p) = dx1(p);
              i2_p(pos_p + offset_p) = i2(p);
              dx2_p(pos_p + offset_p) = dx2(p);
              phi_p(pos_p + offset_p) = phi(p);
              ux1_p(pos_p + offset_p) = SIGN(cosAngle) * upar * b_int_Cart[0];
              ux2_p(pos_p + offset_p) = SIGN(cosAngle) * upar * b_int_Cart[1];
              ux3_p(pos_p + offset_p) = SIGN(cosAngle) * upar * b_int_Cart[2];
              weight_p(pos_p + offset_p) = weight(p);
              tag_p(pos_p + offset_p) = ParticleTag::alive;

              // HACK for more multiplicity
              // elec_p = Kokkos::atomic_fetch_add(&elec_ind(), 1);
              // i1_e(elec_p + offset_e) = i1(p);
              // dx1_e(elec_p + offset_e) = dx1(p);
              // i2_e(elec_p + offset_e) = i2(p);
              // dx2_e(elec_p + offset_e) = dx2(p);
              // phi_e(elec_p + offset_e) = phi(p);
              // ux1_e(elec_p + offset_e) = 0.9 * SIGN(cosAngle) * upar * b_int_Cart[0];
              // ux2_e(elec_p + offset_e) = 0.9 * SIGN(cosAngle) * upar * b_int_Cart[1];
              // ux3_e(elec_p + offset_e) = 0.9 * SIGN(cosAngle) * upar * b_int_Cart[2];
              // weight_e(elec_p + offset_e) = weight(p);
              // tag_e(elec_p + offset_e) = ParticleTag::alive;

              // pos_p  = Kokkos::atomic_fetch_add(&pos_ind(), 1);
              // i1_p(pos_p + offset_p) = i1(p);
              // dx1_p(pos_p + offset_p) = dx1(p);
              // i2_p(pos_p + offset_p) = i2(p);
              // dx2_p(pos_p + offset_p) = dx2(p);
              // phi_p(pos_p + offset_p) = phi(p);
              // ux1_p(pos_p + offset_p) = 0.9 * SIGN(cosAngle) * upar * b_int_Cart[0];
              // ux2_p(pos_p + offset_p) = 0.9 * SIGN(cosAngle) * upar * b_int_Cart[1];
              // ux3_p(pos_p + offset_p) = 0.9 * SIGN(cosAngle) * upar * b_int_Cart[2];
              // weight_p(pos_p + offset_p) = weight(p);
              // tag_p(pos_p + offset_p) = ParticleTag::alive;


              auto cbuff_acc     = cbuff_sc.access();
              cbuff_acc(static_cast<int>(i1(p)), static_cast<int>(i2(p))) += weight(p) * inv_n0_ /
                   metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                                       static_cast<real_t>(i2(p)) + HALF });
          }

        });

            auto elec_ind_h = Kokkos::create_mirror(elec_ind);
            Kokkos::deep_copy(elec_ind_h, elec_ind);
            species_e.set_npart(offset_e + elec_ind_h());

            auto pos_ind_h = Kokkos::create_mirror(pos_ind);
            Kokkos::deep_copy(pos_ind_h, pos_ind);
            species_p.set_npart(offset_p + pos_ind_h());

          }

      Kokkos::Experimental::contribute(cbuff, cbuff_sc);
      } // Pair production kernel (threshold)
    }

    if (is_first_step) {
        is_first_step = false;
    }


    }
  
void CustomFieldOutput(const std::string&   name,
                           ndfield_t<M::Dim, 6> buffer,
                           std::size_t          index,
                           const Domain<S, M>&  domain) {
  if (name == "pploc") {
        Kokkos::deep_copy(Kokkos::subview(buffer, Kokkos::ALL, Kokkos::ALL, index),
                          cbuff);
  } 
  if (name == "pplocsec") {
        Kokkos::deep_copy(Kokkos::subview(buffer, Kokkos::ALL, Kokkos::ALL, index),
                          cbuff2);
  }
}
  
  };

} // namespace user

#endif
