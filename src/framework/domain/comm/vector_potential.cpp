#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"

#include "framework/domain/domain.h"
#include "framework/domain/mesh.h"
#include "framework/domain/metadomain.h"
#include "framework/specialization_registry.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"

  #include <mpi.h>
#endif // MPI_ENABLED

#include <string>
#include <vector>

namespace ntt {

#if defined(MPI_ENABLED) && defined(OUTPUT_ENABLED)
  template <SimEngine::type S, MetricClass M>
  void ExtractVectorPotential(ndfield_t<M::Dim, 6>& buffer,
                              array_t<real_t*>&     aphi_r,
                              unsigned short        buff_idx,
                              const Mesh<M>&        mesh) {
    Kokkos::parallel_for(
      "AddVectorPotential",
      mesh.rangeActiveCells(),
      Lambda(cellidx_t i1, cellidx_t i2) {
        buffer(i1, i2, buff_idx) += aphi_r(i1 - N_GHOSTS);
      });
  }

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::CommunicateVectorPotential(unsigned short buff_idx) {
    if constexpr (M::Dim == Dim::_2D) {
      auto       local_domain = subdomain_ptr(l_subdomain_indices()[0]);
      const auto nx1          = local_domain->mesh.n_active(in::x1);
      const auto nx2          = local_domain->mesh.n_active(in::x2);

      auto& buffer = local_domain->fields.bckp;

      const auto nranks_x1 = ndomains_per_dim()[0];
      const auto nranks_x2 = ndomains_per_dim()[1];

      for (auto nr2 { 1u }; nr2 < nranks_x2; ++nr2) {
        const auto rank_send_pre = (nr2 - 1u) * nranks_x1;
        const auto rank_recv_pre = nr2 * nranks_x1;
        for (auto nr1 { 0u }; nr1 < nranks_x1; ++nr1) {
          const auto rank_send = rank_send_pre + nr1;
          const auto rank_recv = rank_recv_pre + nr1;
          if (static_cast<unsigned int>(local_domain->mpi_rank()) == rank_send) {
            array_t<real_t*> aphi_r { "Aphi_r", nx1 };
            Kokkos::deep_copy(
              aphi_r,
              Kokkos::subview(buffer,
                              std::make_pair(N_GHOSTS, N_GHOSTS + nx1),
                              N_GHOSTS + nx2 - 1,
                              buff_idx));
  #if !defined(DEVICE_ENABLED) || defined(GPU_AWARE_MPI)
            MPI_Send(aphi_r.data(),
                     nx1,
                     mpi::get_type<real_t>(),
                     rank_recv,
                     0,
                     MPI_COMM_WORLD);
  #else
            auto aphi_r_h = Kokkos::create_mirror_view(aphi_r);
            Kokkos::deep_copy(aphi_r_h, aphi_r);
            MPI_Send(aphi_r_h.data(),
                     nx1,
                     mpi::get_type<real_t>(),
                     rank_recv,
                     0,
                     MPI_COMM_WORLD);
  #endif
          } else if (local_domain->mpi_rank() == rank_recv) {
            array_t<real_t*> aphi_r { "Aphi_r", nx1 };
  #if !defined(DEVICE_ENABLED) || defined(GPU_AWARE_MPI)
            MPI_Recv(aphi_r.data(),
                     nx1,
                     mpi::get_type<real_t>(),
                     rank_send,
                     0,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
  #else
            auto aphi_r_h = Kokkos::create_mirror_view(aphi_r);
            MPI_Recv(aphi_r_h.data(),
                     nx1,
                     mpi::get_type<real_t>(),
                     rank_send,
                     0,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            Kokkos::deep_copy(aphi_r, aphi_r_h);
  #endif
            ExtractVectorPotential<S, M>(buffer, aphi_r, buff_idx, local_domain->mesh);
          }
        }
      }
    } else {
      raise::Error("CommunicateVectorPotential: comm vector potential only "
                   "possible for 2D",
                   HERE);
    }
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
  #define COMMVECTORPOTENTIAL(S, M, D)                                         \
    template void Metadomain<S, M<D>>::CommunicateVectorPotential(unsigned short);

  NTT_FOREACH_SPECIALIZATION(COMMVECTORPOTENTIAL)

  #undef COMMVECTORPOTENTIAL
#endif
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
