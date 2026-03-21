#include "framework/domain/metadomain.h"
#include "framework/domain/domain.h"
#include "framework/specialization_registry.h"
#include "arch/mpi_tags.h"
#include "utils/numeric.h"
#include "utils/reporter.h"
#include "framework/parameters/parameters.h"

#include <vector>
#include <cmath>
#include <iostream>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

namespace ntt {

  // Load balancing helper based on the 1D julia model
  bool negotiate_boundary_single(const std::vector<real_t>& N, std::vector<int>& bounds, int i, int n_ghost, double tol) {
    int left_start = bounds[i];
    int mid = bounds[i+1];
    int right_end = bounds[i+2];

    double w1 = 0;
    for (int k = left_start; k < mid; ++k) w1 += N[k];
    double w2 = 0;
    for (int k = mid; k < right_end; ++k) w2 += N[k];

    if (std::abs(w1 - w2) <= tol) return false;

    int L_min = 2 * n_ghost + 1;
    double best_diff = std::abs(w1 - w2);
    int best_shift = 0;

    if (w1 > w2) {
      int max_shift = (mid - left_start) - L_min;
      double current_transfer = 0.0;
      for (int s = 1; s <= max_shift; ++s) {
        current_transfer += N[mid - s];
        double new_diff = std::abs((w1 - current_transfer) - (w2 + current_transfer));
        if (new_diff < best_diff) {
          best_diff = new_diff;
          best_shift = s;
        } else {
          break;
        }
      }
      if (best_shift > 0) {
        bounds[i+1] -= best_shift;
        return true;
      }
    } else if (w2 > w1) {
      int max_shift = (right_end - mid) - L_min;
      double current_transfer = 0.0;
      for (int s = 1; s <= max_shift; ++s) {
        current_transfer += N[mid + s - 1]; // mid is inclusive for right domain.
        double new_diff = std::abs((w1 + current_transfer) - (w2 - current_transfer));
        if (new_diff < best_diff) {
          best_diff = new_diff;
          best_shift = s;
        } else {
          break;
        }
      }
      if (best_shift > 0) {
        bounds[i+1] += best_shift;
        return true;
      }
    }
    return false;
  }

  template <SimEngine::type S, class M>
    requires IsCompatibleWithMetadomain<M>
  void Metadomain<S, M>::BalanceLoad(const SimulationParams& params) {
    const auto lb_dims = params.template get<std::vector<int>>("simulation.domain.load_balancing.dimensions");
    const auto lb_max_iters = 1; //params.template get<int>("simulation.domain.load_balancing.max_iterations", 10);
    const auto lb_tol = 0.1; //params.template get<real_t>("simulation.domain.load_balancing.tolerance", 0.1);

    if (lb_dims.empty()) return;

    for (int dim : lb_dims) {
      if (dim < 1 || dim > D) continue; 
      int d = dim - 1; 

      int nx_domains = g_ndomains_per_dim[d];
      if (nx_domains < 2) continue;

      int global_ncells = g_mesh.n_active(static_cast<in>(d));
      Kokkos::View<real_t*> d_N("N", global_ncells);

      // 1. Gather particles histogram natively on the GPU device
      runOnLocalDomains([&](auto& dom) {
        for (const auto& sp : dom.species) {
          if (sp.npart() == 0) continue;
          
          auto global_offset = dom.offset_ncells()[d];
          auto i_view = (d == 0) ? sp.i1 : ((d == 1) ? sp.i2 : sp.i3);

          Kokkos::parallel_for("GatherHistogram", sp.rangeActiveParticles(), KOKKOS_LAMBDA(int p) {
            int local_cell = i_view(p) - N_GHOSTS;
            int global_cell = global_offset + local_cell;
            if (global_cell >= 0 && global_cell < global_ncells) {
                Kokkos::atomic_add(&d_N(global_cell), ONE); 
            }
          });
        }
      });

      auto h_N = Kokkos::create_mirror_view(d_N);
      Kokkos::deep_copy(h_N, d_N);
      std::vector<real_t> N(global_ncells, ZERO);
      for (int i = 0; i < global_ncells; ++i) {
          N[i] = h_N(i);
      }

#if defined(MPI_ENABLED)
      std::vector<real_t> N_global(global_ncells, ZERO);
      MPI_Allreduce(N.data(), N_global.data(), global_ncells, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      N = N_global;
#endif

      // 2. Setup bounds vector 
      std::vector<int> bounds(nx_domains + 1, 0);
      bounds[0] = 0;
      bounds[nx_domains] = global_ncells;
      
      for (int i = 0; i < nx_domains - 1; ++i) {
        std::vector<unsigned int> target_idx(D, 0);
        target_idx[d] = i; 
        unsigned int flatten_idx = g_domain_offset2index[target_idx];
        const auto& dom = g_subdomains[flatten_idx];
        bounds[i+1] = dom.offset_ncells()[d] + dom.mesh.n_active(static_cast<in>(d));
      }

      std::vector<int> old_bounds = bounds;

      // 3. Negotiate boundaries using iterative method
      for (int iter = 1; iter <= lb_max_iters; ++iter) {
        bool moved_global = false;
        // RED phase
        for (int i = 1; i < nx_domains; i += 2) {
            moved_global |= negotiate_boundary_single(N, bounds, i - 1, N_GHOSTS, lb_tol);
        }
        // BLACK phase
        for (int i = 2; i < nx_domains; i += 2) {
            moved_global |= negotiate_boundary_single(N, bounds, i - 1, N_GHOSTS, lb_tol);
        }
        if (!moved_global) break;
      }

      bool any_change = false;
      for (int i = 0; i <= nx_domains; ++i) {
        if (bounds[i] != old_bounds[i]) any_change = true;
      }

      if (!any_change) continue;

      info::Print(fmt::format("Load Balancing shifted boundaries in dimension {}", dim), true);

      // 4. Update domains if boundary changed
      std::vector<Domain<S, M>> new_subdomains;
      for (unsigned int idx = 0; idx < g_ndomains; ++idx) {
        auto& old_dom = g_subdomains[idx];
        
        std::vector<ncells_t> new_ncells = old_dom.mesh.n_active();
        std::vector<ncells_t> old_offset_ncells = old_dom.offset_ncells();
        std::vector<ncells_t> new_offset_ncells = old_offset_ncells;
        
        int grid_i = old_dom.offset_ndomains()[d]; 
        
        new_offset_ncells[d] = bounds[grid_i];
        new_ncells[d] = bounds[grid_i+1] - bounds[grid_i];
        
        boundaries_t<real_t> new_extent = old_dom.mesh.extent();
        real_t x_min = 0, x_max = 0;
        if (d == 0) {
           x_min = g_mesh.metric.template convert<1, Crd::Cd, Crd::Ph>(new_offset_ncells[d]);
           x_max = g_mesh.metric.template convert<1, Crd::Cd, Crd::Ph>(new_offset_ncells[d] + new_ncells[d]);
        } else if (d == 1) {
           if constexpr (D == Dim::_2D || D == Dim::_3D) {
              x_min = g_mesh.metric.template convert<2, Crd::Cd, Crd::Ph>(new_offset_ncells[d]);
              x_max = g_mesh.metric.template convert<2, Crd::Cd, Crd::Ph>(new_offset_ncells[d] + new_ncells[d]);
           }
        } else if (d == 2) {
           if constexpr (D == Dim::_3D) {
              x_min = g_mesh.metric.template convert<3, Crd::Cd, Crd::Ph>(new_offset_ncells[d]);
              x_max = g_mesh.metric.template convert<3, Crd::Cd, Crd::Ph>(new_offset_ncells[d] + new_ncells[d]);
           }
        }
        new_extent[d].first = x_min;
        new_extent[d].second = x_max;

        if (old_dom.is_placeholder()) {
           new_subdomains.emplace_back(true, idx, old_dom.offset_ndomains(), new_offset_ncells, new_ncells, new_extent, g_metric_params, g_species_params);
        } else {
           new_subdomains.emplace_back(idx, old_dom.offset_ndomains(), new_offset_ncells, new_ncells, new_extent, g_metric_params, g_species_params);
           auto& new_dom = new_subdomains.back();
           
           auto new_em = new_dom.fields.em;
           auto old_em = old_dom.fields.em;
           auto new_bckp = new_dom.fields.bckp;
           auto old_bckp = old_dom.fields.bckp;
           
           int new_off_0 = new_offset_ncells[0];
           int new_off_1 = (D > 1) ? new_offset_ncells[1] : 0;
           int new_off_2 = (D > 2) ? new_offset_ncells[2] : 0;
           
           int old_off_0 = old_offset_ncells[0];
           int old_off_1 = (D > 1) ? old_offset_ncells[1] : 0;
           int old_off_2 = (D > 2) ? old_offset_ncells[2] : 0;
           
           int old_ext_0 = old_em.extent(0);
           int old_ext_1 = (D > 1) ? old_em.extent(1) : 0;
           int old_ext_2 = (D > 2) ? old_em.extent(2) : 0;

           if constexpr (D == Dim::_1D) {
             Kokkos::parallel_for("CopyFields_LB_1D", new_dom.mesh.rangeActiveCells(), KOKKOS_LAMBDA(int i1) {
               int gx1 = new_off_0 + i1 - N_GHOSTS;
               int old_i1 = gx1 - old_off_0 + N_GHOSTS;
               if (old_i1 >= N_GHOSTS && old_i1 < old_ext_0 - N_GHOSTS) {
                  for (int comp = 0; comp < 6; ++comp) {
                     new_em(i1, comp) = old_em(old_i1, comp);
                     new_bckp(i1, comp) = old_bckp(old_i1, comp);
                  }
               }
             });
           } else if constexpr (D == Dim::_2D) {
             Kokkos::parallel_for("CopyFields_LB_2D", new_dom.mesh.rangeActiveCells(), KOKKOS_LAMBDA(int i1, int i2) {
               int gx1 = new_off_0 + i1 - N_GHOSTS;
               int gx2 = new_off_1 + i2 - N_GHOSTS;
               int old_i1 = gx1 - old_off_0 + N_GHOSTS;
               int old_i2 = gx2 - old_off_1 + N_GHOSTS;
               if (old_i1 >= N_GHOSTS && old_i1 < old_ext_0 - N_GHOSTS &&
                   old_i2 >= N_GHOSTS && old_i2 < old_ext_1 - N_GHOSTS) {
                  for (int comp = 0; comp < 6; ++comp) {
                     new_em(i1, i2, comp) = old_em(old_i1, old_i2, comp);
                     new_bckp(i1, i2, comp) = old_bckp(old_i1, old_i2, comp);
                  }
               }
             });
           } else if constexpr (D == Dim::_3D) {
             Kokkos::parallel_for("CopyFields_LB_3D", new_dom.mesh.rangeActiveCells(), KOKKOS_LAMBDA(int i1, int i2, int i3) {
               int gx1 = new_off_0 + i1 - N_GHOSTS;
               int gx2 = new_off_1 + i2 - N_GHOSTS;
               int gx3 = new_off_2 + i3 - N_GHOSTS;
               int old_i1 = gx1 - old_off_0 + N_GHOSTS;
               int old_i2 = gx2 - old_off_1 + N_GHOSTS;
               int old_i3 = gx3 - old_off_2 + N_GHOSTS;
               if (old_i1 >= N_GHOSTS && old_i1 < old_ext_0 - N_GHOSTS &&
                   old_i2 >= N_GHOSTS && old_i2 < old_ext_1 - N_GHOSTS &&
                   old_i3 >= N_GHOSTS && old_i3 < old_ext_2 - N_GHOSTS) {
                  for (int comp = 0; comp < 6; ++comp) {
                     new_em(i1, i2, i3, comp) = old_em(old_i1, old_i2, old_i3, comp);
                     new_bckp(i1, i2, i3, comp) = old_bckp(old_i1, old_i2, old_i3, comp);
                  }
               }
             });
           }

           for(size_t s_idx = 0; s_idx < g_species_params.size(); ++s_idx) {
               auto& old_sp = old_dom.species[s_idx];
               auto& new_sp = new_dom.species[s_idx];
               new_sp.set_npart(old_sp.npart());
               
               Kokkos::deep_copy(new_sp.i1, old_sp.i1);
               if(D>1) Kokkos::deep_copy(new_sp.i2, old_sp.i2);
               if(D>2) Kokkos::deep_copy(new_sp.i3, old_sp.i3);
               Kokkos::deep_copy(new_sp.dx1, old_sp.dx1);
               if(D>1)Kokkos::deep_copy(new_sp.dx2, old_sp.dx2);
               if(D>2)Kokkos::deep_copy(new_sp.dx3, old_sp.dx3);
               Kokkos::deep_copy(new_sp.ux1, old_sp.ux1);
               Kokkos::deep_copy(new_sp.ux2, old_sp.ux2);
               Kokkos::deep_copy(new_sp.ux3, old_sp.ux3);
               Kokkos::deep_copy(new_sp.weight, old_sp.weight);
               
               int offset_diff1 = old_offset_ncells[0] - new_offset_ncells[0];
               if constexpr (D == Dim::_1D) {
                 if (offset_diff1 != 0) {
                   auto i1_view = new_sp.i1;
                   auto tag_view = new_sp.tag;
                   int ni1 = new_ncells[0];
                   Kokkos::parallel_for("ShiftParticles_1D", new_sp.rangeActiveParticles(), KOKKOS_LAMBDA(int p) {
                      i1_view(p) += offset_diff1;
#if defined(MPI_ENABLED)
                      tag_view(p) = mpi::SendTag(tag_view(p), i1_view(p) < 0, i1_view(p) >= ni1);
#endif
                   });
                 }
               } else if constexpr (D == Dim::_2D) {
                 int offset_diff2 = old_offset_ncells[1] - new_offset_ncells[1];
                 if (offset_diff1 != 0 || offset_diff2 != 0) {
                   auto i1_view = new_sp.i1;
                   auto i2_view = new_sp.i2;
                   auto tag_view = new_sp.tag;
                   int ni1 = new_ncells[0];
                   int ni2 = new_ncells[1];
                   Kokkos::parallel_for("ShiftParticles_2D", new_sp.rangeActiveParticles(), KOKKOS_LAMBDA(int p) {
                      i1_view(p) += offset_diff1;
                      i2_view(p) += offset_diff2;
#if defined(MPI_ENABLED)
                      tag_view(p) = mpi::SendTag(tag_view(p), i1_view(p) < 0, i1_view(p) >= ni1, i2_view(p) < 0, i2_view(p) >= ni2);
#endif
                   });
                 }
               } else if constexpr (D == Dim::_3D) {
                 int offset_diff2 = old_offset_ncells[1] - new_offset_ncells[1];
                 int offset_diff3 = old_offset_ncells[2] - new_offset_ncells[2];
                 if (offset_diff1 != 0 || offset_diff2 != 0 || offset_diff3 != 0) {
                   auto i1_view = new_sp.i1;
                   auto i2_view = new_sp.i2;
                   auto i3_view = new_sp.i3;
                   auto tag_view = new_sp.tag;
                   int ni1 = new_ncells[0];
                   int ni2 = new_ncells[1];
                   int ni3 = new_ncells[2];
                   Kokkos::parallel_for("ShiftParticles_3D", new_sp.rangeActiveParticles(), KOKKOS_LAMBDA(int p) {
                      i1_view(p) += offset_diff1;
                      i2_view(p) += offset_diff2;
                      i3_view(p) += offset_diff3;
#if defined(MPI_ENABLED)
                      tag_view(p) = mpi::SendTag(tag_view(p),
                                                 i1_view(p) < 0, i1_view(p) >= ni1,
                                                 i2_view(p) < 0, i2_view(p) >= ni2,
                                                 i3_view(p) < 0, i3_view(p) >= ni3);
#endif
                   });
                 }
               }
           }
        }
      }
      g_subdomains = std::move(new_subdomains);

      redefineNeighbors();
      redefineBoundaries();
      
      runOnLocalDomains([&](auto& dom){
        CommunicateParticles(dom);
        CommunicateFields(dom, Comm::E | Comm::B | Comm::J); 
      });
    }
  }

#define METADOMAIN_LB(S, M, D) \
  template void Metadomain<S, M<D>>::BalanceLoad(const SimulationParams&);

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_LB)
#undef METADOMAIN_LB

} // namespace ntt
