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
    const auto lb_dims       = params.template get<std::vector<int>>("simulation.domain.load_balancing.dimensions");
    const auto lb_max_iters  = static_cast<int>(params.template get<unsigned int>("simulation.domain.load_balancing.max_iterations"));
    const auto lb_tol        = static_cast<double>(params.template get<real_t>("simulation.domain.load_balancing.tolerance"));

    // if no dimensions specified, skip load balancing
    if (lb_dims.empty()) return;

    auto global_boundaries = std::vector<std::vector<ncells_t>> {};
    auto offset_ncells     = std::vector<std::vector<ncells_t>> {};
    auto global_ncells     = std::vector<std::vector<ncells_t>> {};
    
    // track if any boundary changed across all dimensions to avoid unnecessary domain updates
    bool any_change = false;

    // loop over all dimenstions to be load balanced
    for (int dim : lb_dims) {
      // ToDo: fallback options for dimentions that should not be load-balanced.
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
      std::vector<ncells_t> bounds(nx_domains + 1, 0);
      bounds[0] = 0;
      bounds[nx_domains] = global_ncells;
      
      for (int i = 0; i < nx_domains - 1; ++i) {
        std::vector<unsigned int> target_idx(D, 0);
        target_idx[d] = i; 
        unsigned int flatten_idx = g_domain_offset2index[target_idx];
        const auto& dom = g_subdomains[flatten_idx];
        // compute global cell index of the right boundary of this domain
        bounds[i+1] = dom.offset_ncells()[d] + dom.mesh.n_active(static_cast<int>(d));
      }

      std::vector<ncells_t> old_bounds = bounds;

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

      // check if any boundary changed
      for (int i = 0; i <= nx_domains; ++i) {
        if (bounds[i] != old_bounds[i]) any_change = true;
      }

      // store updated boundaries for this dimension
      global_boundaries.push_back(bounds);

      // store offsets for this dimension (for later use in shifting particles)
      std::vector<ncells_t> offsets(nx_domains, 0);
      for (int i = 1; i <= nx_domains; ++i) { // first boundary is fixed at 0
        offsets[i] = bounds[i] - old_bounds[i];
      }
      offset_ncells.push_back(offsets);

      // store number of cells for each domain in this dimension (for later use in domain updates)
      std::vector<ncells_t> new_ncells(nx_domains, 0);
      for (int i = 1; i <= nx_domains; ++i) {
        new_ncells[i] = bounds[i] - bounds[i-1];
      }
      global_ncells.push_back(new_ncells);
      info::Print(fmt::format("Load Balancing shifted boundaries in dimension {}", dim), true);

    } // loop over dimensions

    // no changes, skip domain updates
    if (!any_change) return;

    // ToDo: Mesh update


    // ToDo: Field update
    for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
#if defined(MPI_ENABLED)
      // !TODO: need to change to support multiple domains per rank
      // assuming ONE local subdomain
      const auto local = ((int)idx == g_mpi_rank);
      if (local) {
        auto nxnew = std::vector<ncells_t>(D, 0);
        auto nxold = std::vector<ncells_t>(D, 0);
        for (auto d { 0 }; d < (short)D; ++d) {
          nxnew.push_back(new_ncells[d][idx]);
          nxold.push_back(nxnew - offset_ncells[d][idx]);
        }
        if (offset_ncells[d][idx] < 0) {
          // domain is shrinking -> right boundary moves left, need to send data to the right neighbor
          // ToDo: define fields
          ndfield_t<D, 6> new_fields {... };
          ndfield_t<D, 6> send_fields {... };

          if constexpr (D == Dim::_1D) {
            Kokkos::deep_copy(new_fields, Kokkos::slice(old_fields, { 0, nxnew[0] }));
            Kokkos::deep_copy(send_fields, Kokkos::slice(old_fields, { nxnew[0], nxold[0] }));
          } else if constexpr (D == Dim::_2D) {
            Kokkos::deep_copy(new_fields, Kokkos::slice(old_fields, { 0, nxnew[0] }, { 0, nxnew[1] }));
            Kokkos::deep_copy(send_fields, Kokkos::slice(old_fields, { nxnew[0], nxold[0] }, { nxnew[1], nxold[1] }));
          } else if constexpr (D == Dim::_3D) {
            Kokkos::deep_copy(new_fields, Kokkos::slice(old_fields, { 0, nxnew[0] }, { 0, nxnew[1] }, { 0, nxnew[2] }));
            Kokkos::deep_copy(send_fields, Kokkos::slice(old_fields, { nxnew[0], nxold[0] }, { nxnew[1], nxold[1] }, { nxnew[2], nxold[2] }));
          }
      
          MPI_SEND(send_fields, ...);
        } else if (offset_ncells[d][idx] > 0) {
          // domain is growing -> right boundary moves right, need to receive data from the left neighbor
          ndfield_t<D, 6> new_fields {... };
          ndfield_t<D, 6> recv_fields {... };
          
          if constexpr (D == Dim::_1D) {
            Kokkos::deep_copy(Kokkos::slice(new_fields, { nxnew[0] - nxold[0], nxnew[0] }), old_fields);
          } else if constexpr (D == Dim::_2D) {
            Kokkos::deep_copy(Kokkos::slice(new_fields, { nxnew[0] - nxold[0], nxnew[0] }, { nxnew[1] - nxold[1], nxnew[1] }), old_fields);
          } else if constexpr (D == Dim::_3D) {
            Kokkos::deep_copy(Kokkos::slice(new_fields, { nxnew[0] - nxold[0], nxnew[0] }, { nxnew[1] - nxold[1], nxnew[1] }, 
              { nxnew[2] - nxold[2], nxnew[2] }), old_fields);
          }

          MPI_RECV(recv_fields, ...);
          Kokkos::deep_copy(Kokkos::slice(new_fields, { 0, nxnew[0] - nxold[0] }), recv_fields);
        }
      }
      
      g_subdomains.back().set_mpi_rank(idx);
      if (g_subdomains.back().mpi_rank() == g_mpi_rank) {
        g_local_subdomain_indices.push_back(idx);
      }
#endif // MPI_ENABLED  

    }
      
    // ToDo: Particle update
    for(size_t s_idx = 0; s_idx < g_species_params.size(); ++s_idx) {
      auto& sp = dom.species[s_idx];

      // Reset all copied particle tags to 'alive': particles with
      // send-direction tags from the previous pusher step must not be
      // re-sent by CommunicateParticles; ShiftParticles below will
      // re-tag any particle that is now out of the new domain bounds.
      {
        auto tag_view = sp.tag;
        Kokkos::parallel_for("ResetTags_LB", sp.rangeActiveParticles(), KOKKOS_LAMBDA(int p) {
            tag_view(p) = ParticleTag::alive;
        });
      }
      
      int offset_diff1 = offset_ncells[0][idx];
      if constexpr (D == Dim::_1D) {
        if (offset_diff1 != 0) {
          auto i1_view = sp.i1;
          auto i1_prev_view = sp.i1_prev;
          auto tag_view = sp.tag;
          int ni1 = new_ncells[0];
          Kokkos::parallel_for("ShiftParticles_1D", sp.rangeActiveParticles(), KOKKOS_LAMBDA(int p) {
             i1_view(p) += offset_diff1;
             i1_prev_view(p) += offset_diff1;
#if defined(MPI_ENABLED)
              tag_view(p) = mpi::SendTag(tag_view(p), i1_view(p) < 0, i1_view(p) >= ni1);
#endif
          });
        }
      } else if constexpr (D == Dim::_2D) {
        int offset_diff2 = offset_ncells[domain_idx][1];
        if (offset_diff1 != 0 || offset_diff2 != 0) {
          auto i1_view = sp.i1;
          auto i1_prev_view = sp.i1_prev;
          auto i2_view = sp.i2;
          auto i2_prev_view = sp.i2_prev;
          auto tag_view = sp.tag;
          int ni1 = new_ncells[0];
          int ni2 = new_ncells[1];
          Kokkos::parallel_for("ShiftParticles_2D", sp.rangeActiveParticles(), KOKKOS_LAMBDA(int p) {
             i1_view(p) += offset_diff1;
             i2_view(p) += offset_diff2;
             i1_prev_view(p) += offset_diff1;
             i2_prev_view(p) += offset_diff2;
#if defined(MPI_ENABLED)
                      tag_view(p) = mpi::SendTag(tag_view(p), i1_view(p) < 0, i1_view(p) >= ni1, i2_view(p) < 0, i2_view(p) >= ni2);
#endif
                   });
                 }
               } else if constexpr (D == Dim::_3D) {
                 int offset_diff2 = offset_ncells[domain_idx][1];
                 int offset_diff3 = offset_ncells[domain_idx][2];
                 if (offset_diff1 != 0 || offset_diff2 != 0 || offset_diff3 != 0) {
                   auto i1_view = sp.i1;
                   auto i2_view = sp.i2;
                   auto i3_view = sp.i3;
                   auto i1_prev_view = sp.i1_prev;
                   auto i2_prev_view = sp.i2_prev;
                   auto i3_prev_view = sp.i3_prev;
                   auto tag_view = sp.tag;
                   int ni1 = new_ncells[0];
                   int ni2 = new_ncells[1];
                   int ni3 = new_ncells[2];
                   Kokkos::parallel_for("ShiftParticles_3D", sp.rangeActiveParticles(), KOKKOS_LAMBDA(int p) {
                      i1_view(p) += offset_diff1;
                      i2_view(p) += offset_diff2;
                      i3_view(p) += offset_diff3;
                      i1_prev_view(p) += offset_diff1;
                      i2_prev_view(p) += offset_diff2;
                      i3_prev_view(p) += offset_diff3;
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

      CommunicateParticles(dom);
      CommunicateFields(dom, Comm::E | Comm::B | Comm::J); 
    }
  }

#define METADOMAIN_LB(S, M, D) \
  template void Metadomain<S, M<D>>::BalanceLoad(const SimulationParams&);

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_LB)
#undef METADOMAIN_LB

} // namespace ntt
