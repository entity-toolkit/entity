#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/field_setter.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

#include <algorithm>
#include <utility>

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    /*
      Sets up magnetic and electric field components for the simulation.
      Must satisfy E = -v x B for Lorentz Force to be zero.

      @param bmag: magnetic field scaling
      @param btheta: magnetic field polar angle
      @param bphi: magnetic field azimuthal angle
      @param drift_ux: drift velocity in the x direction
    */
    InitFields(real_t bmag, real_t btheta, real_t bphi, real_t drift_ux)
      : Bmag { bmag }
      , Btheta { btheta * static_cast<real_t>(convert::deg2rad) }
      , Bphi { bphi * static_cast<real_t>(convert::deg2rad) }
      , Vx { drift_ux } {}

    // magnetic field components
    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return Bmag * math::cos(Btheta);
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return Bmag * math::sin(Btheta) * math::sin(Bphi);
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return Bmag * math::sin(Btheta) * math::cos(Bphi);
    }

    // electric field components
    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return -Vx * Bmag * math::sin(Btheta) * math::cos(Bphi);
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return Vx * Bmag * math::sin(Btheta) * math::sin(Bphi);
    }

  private:
    const real_t Btheta, Bphi, Vx, Bmag;
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    // compatibility traits for the problem generator
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D> {}
    };
    const SimulationParams& params;
    Metadomain<S, M>&       metadomain;

    // domain properties
    const real_t  global_xmin, global_xmax;
    // gas properties
    const real_t  drift_ux, temperature, temperature_ratio, filling_fraction;
    // injector properties
    const real_t  injector_velocity, injection_start, dt;
    const int     injection_frequency;
    // magnetic field properties
    real_t        Btheta, Bphi, Bmag;
    InitFields<D> init_flds;

    PGen(const SimulationParams& p, Metadomain<S, M>& m)
      : params { p }
      , metadomain { m }
      , global_xmin { metadomain.mesh().extent(in::x1).first }
      , global_xmax { metadomain.mesh().extent(in::x1).second }
      , drift_ux { params.template get<real_t>("setup.drift_ux") }
      , temperature { params.template get<real_t>("setup.temperature") }
      , temperature_ratio { params.template get<real_t>(
          "setup.temperature_ratio") }
      , Bmag { params.template get<real_t>("setup.Bmag", ZERO) }
      , Btheta { params.template get<real_t>("setup.Btheta", ZERO) }
      , Bphi { params.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Bmag, Btheta, Bphi, drift_ux }
      , filling_fraction { params.template get<real_t>("setup.filling_fraction",
                                                       1.0) }
      , injector_velocity { params.template get<real_t>(
          "setup.injector_velocity",
          1.0) }
      , injection_start { params.template get<real_t>("setup.injection_start", 0.0) }
      , injection_frequency { params.template get<int>(
          "setup.injection_frequency",
          100) }
      , dt { params.template get<real_t>("algorithms.timestep.dt") } {}

    auto MatchFields(simtime_t) const -> InitFields<D> {
      return init_flds;
    }

    auto FixFieldsConst(const bc_in&, const em& comp) const
      -> std::pair<real_t, bool> {
      if (comp == em::ex1) {
        return { init_flds.ex1({ ZERO }), true };
      } else if ((comp == em::ex2) or (comp == em::ex3)) {
        return { ZERO, true };
      } else if (comp == em::bx1) {
        return { init_flds.bx1({ ZERO }), true };
      } else if (comp == em::bx2) {
        return { init_flds.bx2({ ZERO }), true };
      } else if (comp == em::bx3) {
        return { init_flds.bx3({ ZERO }), true };
      } else {
        raise::Error("Invalid component", HERE);
        return { ZERO, false };
      }
    }

    void InitPrtls(Domain<S, M>& domain) {

      /*
       *  Plasma setup as partially filled box
       *
       *  Plasma setup:
       *
       * global_xmin                            global_xmax
       * |                                      |
       * V                                      V
       * |:::::::::::|..........................|
       *             ^
       *             |
       *        filling_fraction
       */

      // minimum and maximum position of particles
      real_t xg_min = global_xmin;
      real_t xg_max = global_xmin + filling_fraction * (global_xmax - global_xmin);

      // define box to inject into
      boundaries_t<real_t> box;
      // loop over all dimensions
      for (auto d { 0u }; d < (unsigned int)M::Dim; ++d) {
        // compute the range for the x-direction
        if (d == static_cast<decltype(d)>(in::x1)) {
          box.emplace_back(xg_min, xg_max);
        } else {
          // inject into full range in other directions
          box.push_back(Range::All);
        }
      }

      // define temperatures of species
      const auto temperatures = std::make_pair(temperature,
                                               temperature_ratio * temperature);
      // define drift speed of species
      const auto drifts       = std::make_pair(
        std::vector<real_t> { -drift_ux, ZERO, ZERO },
        std::vector<real_t> { -drift_ux, ZERO, ZERO });

      // inject particles
      arch::InjectUniformMaxwellians<S, M>(params,
                                           domain,
                                           ONE,
                                           temperatures,
                                           { 1, 2 },
                                           drifts,
                                           false,
                                           box);
    }

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {

      /*
       *  Replenish plasma in a moving injector
       *
       *  Injector setup:
       *
       * global_xmin           purge/replenish  global_xmax
       * |         x_init            |          |
       * V           v               V          V
       * |:::::::::::;::::::::::|\\\\\\\\|......|
       *                       xmin    xmax
       *                                 ^
       *                                 |
       *                           moving injector
       */

      // check if the injector should be active
      if (step % injection_frequency != 0) {
        return;
      }

      // initial position of injector
      const auto x_init = global_xmin +
                          filling_fraction * (global_xmax - global_xmin);

      // compute the position of the injector after the current timestep
      auto xmax = x_init + injector_velocity *
                             (std::max<real_t>(time - injection_start, ZERO) + dt);
      if (xmax >= global_xmax) {
        xmax = global_xmax;
      }

      // compute the beginning of the injected region
      auto xmin = xmax - injection_frequency * dt;
      if (xmin <= global_xmin) {
        xmin = global_xmin;
      }

      // define indice range to reset fields
      boundaries_t<bool> incl_ghosts;
      for (auto d = 0; d < M::Dim; ++d) {
        incl_ghosts.emplace_back(false, false);
      }

      // define box to reset fields
      boundaries_t<real_t> purge_box;
      // loop over all dimension
      for (auto d = 0u; d < M::Dim; ++d) {
        if (d == 0) {
          purge_box.emplace_back(xmin, global_xmax);
        } else {
          purge_box.push_back(Range::All);
        }
      }

      const auto extent = domain.mesh.ExtentToRange(purge_box, incl_ghosts);
      tuple_t<std::size_t, M::Dim> x_min { 0 }, x_max { 0 };
      for (auto d = 0; d < M::Dim; ++d) {
        x_min[d] = extent[d].first;
        x_max[d] = extent[d].second;
      }

      Kokkos::parallel_for("ResetFields",
                           CreateRangePolicy<M::Dim>(x_min, x_max),
                           arch::SetEMFields_kernel<S, M, decltype(init_flds)> {
                             domain.fields.em,
                             init_flds,
                             domain.mesh.metric });
      metadomain.CommunicateFields(domain, Comm::E | Comm::B);

      /*
        tag particles inside the injection zone as dead
      */
      const auto& mesh = domain.mesh;

      // loop over particle species
      for (auto s { 0u }; s < 2; ++s) {
        // get particle properties
        auto& species = domain.species[s];
        auto  i1      = species.i1;
        auto  dx1     = species.dx1;
        auto  tag     = species.tag;

        Kokkos::parallel_for(
          "RemoveParticles",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            // check if the particle is already dead
            if (tag(p) == ParticleTag::dead) {
              return;
            }
            const auto x_Cd = static_cast<real_t>(i1(p)) +
                              static_cast<real_t>(dx1(p));
            const auto x_Ph = mesh.metric.template convert<1, Crd::Cd, Crd::XYZ>(
              x_Cd);

            if (x_Ph > xmin) {
              tag(p) = ParticleTag::dead;
            }
          });
      }

      /*
          Inject slab of fresh plasma
      */

      // define box to inject into
      boundaries_t<real_t> inj_box;
      // loop over all dimension
      for (auto d = 0u; d < M::Dim; ++d) {
        if (d == 0) {
          inj_box.emplace_back(xmin, xmax);
        } else {
          inj_box.push_back(Range::All);
        }
      }

      // same maxwell distribution as above
      const auto temperatures = std::make_pair(temperature,
                                               temperature_ratio * temperature);
      const auto drifts       = std::make_pair(
        std::vector<real_t> { -drift_ux, ZERO, ZERO },
        std::vector<real_t> { -drift_ux, ZERO, ZERO });
      arch::InjectUniformMaxwellians<S, M>(params,
                                           domain,
                                           ONE,
                                           temperatures,
                                           { 1, 2 },
                                           drifts,
                                           false,
                                           inj_box);
    }
  };

  // update domain if needed
  // todo: implement in main code like CustomPostStep
  void CustomUpdateDomain(const SimulationParams& params, Domain<S, M>& local_domain, Domain<S, M>& new_domain, Domain<S, M>& global_domain) {

    // check if the injector should be active
    // ToDo: read parameter into global variable
    if (step % params.template get<int>("setup.domain_decomposition_frequency") != 0) {
      return;
    }

    // compute size of local and global domains 
    const auto local_size   = local_domain->mesh.n_active()[in::x1];
    const auto local_offset = local_domain->offset_ncells()[in::x1];
    const auto global_size  = global_domain->mesh.n_active()[in::x1];

    // global number density field along x1
    index_t Nx_global[global_size] = { 0 };

    /*
      Option 1: Use built-in particle counting kernel to compute number density field and perform MPI allreduce to get global number density field. 
      Then compute new domain boundaries based on the global number density field and perform reshuffling of particles according to new domain boundaries.
    */
    //     tuple_t<std::size_t, M::Dim> local_cells{ 0 }, global_x_min { 0 }, global_x_max { 0 };
    // for (auto d = 0; d < M::Dim; ++d) {
    //   local_cells[d] = local_domain->mesh.n_active(d);
    //   global_x_min[d] = local_domain->offset_ncells(d);
    //   global_x_max[d] = local_domain->mesh.n_active(d) + local_domain->offset_ncells(d);
    // }

    // // compute number density field
    // array_t<int**> NumberOfParticles("num_particles", local_cells);
    // auto scatter_buff = Kokkos::Experimental::create_scatter_view(NumberOfParticles);
    // for (const auto& sp : specs) {
    //   auto& prtl_spec = prtl_species[sp - 1];
    //   // clang-format off
    //   Kokkos::parallel_for(
    //     "ComputeMoments",
    //     prtl_spec.rangeActiveParticles(),
    //     kernel::ParticleMoments_kernel<S, M, F, 6>({}, scatter_buff, buff_idx,
    //                                                prtl_spec.i1, prtl_spec.i2, prtl_spec.i3,
    //                                                prtl_spec.dx1, prtl_spec.dx2, prtl_spec.dx3,
    //                                                prtl_spec.ux1, prtl_spec.ux2, prtl_spec.ux3,
    //                                                prtl_spec.phi, prtl_spec.weight, prtl_spec.tag,
    //                                                prtl_spec.mass(), prtl_spec.charge(),
    //                                                false,
    //                                                mesh.metric, mesh.flds_bc(),
    //                                                ni2, ONE, 0));
    //   // clang-format on
    // }
    // Kokkos::Experimental::contribute(NumberOfParticles, scatter_buff);

    //     // compute particle profile along x1
    // index_t Nx[global_size] = { 0 };
    
    // for (auto i = 0; i < local_size; ++i) {
    //     for (auto d = 0u; d < M::Dim; ++d) {          
    //         // todo: sum over other dimensions  
    //         Nx[local_offset + i] += buffer(i, j, buff_idx);
    //     }
    // }
    // // todo: perform MPI allreduce to get global Nx
    // index_t Nx_global[global_size] = { 0 };
    // MPI_ALLREDUCE(MPI_SUM, Nx, Nx_global, global_size, MPI_TYPE_INT_T, MPI_COMM_WORLD);
    

    /*
      Option 2: Loop over particles and compute number density field manually. Then perform MPI allreduce to get global number density field. 
      Then compute new domain boundaries based on the global number density field and perform reshuffling of particles according to new domain boundaries.
    */
    // store total number of particles in each cell in x1 direction
    array_t<int**> NumberOfParticles("num_particles", local_size);
    // loop over particle species
    for (auto s { 0u }; s < 2; ++s) {
        // get particle properties
        auto& species = local_domain.species[s];
        auto  i1      = species.i1;
        auto  tag     = species.tag;

          auto NumParts_scatter = Kokkos::Experimental::create_scatter_view(
            NumberOfParticles);
          Kokkos::parallel_for(
            "ComputePPC",
            species.rangeActiveParticles(),
            Lambda(index_t p) {
              if (tag(p) != ParticleTag::alive) {
                return;
              }
              auto NumPart_acc    = NumParts_scatter.access();
              NumPart_acc(i1(p)) += 1;
            });
          Kokkos::Experimental::contribute(NumberOfParticles, NumParts_scatter);
    }

    // construct contribution to global number density field along x1 direction
    index_t Nx_local[global_size] = { 0 };
    for (auto i = 0; i < local_size; ++i) {
        Nx_local[i+local_offset] = NumberOfParticles(i);
    }
    // sum up all ranks
    MPI_ALLREDUCE(MPI_SUM, Nx_local, Nx_global, global_size, MPI_TYPE_INT_T, MPI_COMM_WORLD);

    // compute mean particle load
    npart_t total_N = 0;
    for (auto i = 0; i < global_size; ++i) {
        total_N += Nx_global[i];
    }

    // get threshold number of particles
    auto N_1_ranks = global_domain.ndomains_per_dim()[in::x1];
    auto N_23_ranks = 0;
    for (auto d = 1u; d < M::Dim; ++d) {         
        N_23_ranks += global_domain.ndomains_per_dim()[d];
    }

    // maximum allowed load imbalance 
    real_t tolerance = params.load_balancing_tolerance;
    index_t target_N = total_N / (N_1_ranks + N_23_ranks) * tolerance;
    // compute new domain boundaries in x1 direction
    index_t bound_start[N_1_ranks];
    index_t bound_end[N_1_ranks];

    // overwrite N_23_ranks to be 1 if it's initally 0 to avoid division by zero
    if (N_23_ranks == 0) {
        N_23_ranks = 1;
    }

    bound_start[0] = 0;
    for (auto r = 0; r < N_1_ranks-1; ++r) {
        real_t cum_N = 0;
        for (auto i = bound_start[r]; i < global_size; ++i) {
            cum_N += static_cast<real_t>(Nx_global[i]) / N_23_ranks;
            if (cum_N >= target_N) {
                bound_end[r] = i;
                // check if we have more than 5 cells
                index_t Ncells = bound_end[r] - bound_start[r] + 1;
                if (Ncells < 5) {
                    bound_end[r] = bound_start[r] + 5;
                }
                bound_start[r+1] = bound_end[r]+1;
                break;
            }
        }
    }
    // rest of the domain goes to the last rank
    bound_end[N_1_ranks-1] = global_size - 1;

    // compute maximum load imbalance after reshuffling
    index_t max_N = 0;
    for (auto r = 0; r < N_1_ranks; ++r) {
        index_t N_r = 0;
        for (auto i = bound_start[r]; i < bound_end[r]; ++i) {
            N_r += Nx_global[i];
        }
        if (N_r > max_N) {
            max_N = N_r;
        } 
    }
    real_t imbalance = static_cast<real_t>(max_N) / (total_N / N_1_ranks);
    

    // todo: reshuffling of particles according to new domain boundaries

  }
} // namespace user
#endif
