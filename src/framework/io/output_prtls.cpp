#ifdef OUTPUT_ENABLED

#  include "output_prtls.hpp"

#  include "wrapper.h"

#  include "particle_macros.h"
#  include "sim_params.h"

#  include "io/output.h"
#  include "meshblock/meshblock.h"
#  include "meshblock/particles.h"

#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>

#  include <string>

#  ifdef MPI_ENABLED
#    include <mpi.h>
#  endif

namespace ntt {
  template <Dimension D, SimulationEngine S>
  void OutputParticles::put(adios2::IO&             io,
                            adios2::Engine&         writer,
                            const SimulationParams& params,
                            const Metadomain<D>&    metadomain,
                            Meshblock<D, S>&        mblock) const {
    for (auto& s : speciesID()) {
      auto prtls       = mblock.particles[s - 1];
      auto prtl_stride = params.outputPrtlStride();
      auto size        = (std::size_t)(prtls.npart() / prtl_stride);
      if (size == 0 and prtls.npart() > 0) {
        size        = prtls.npart();
        prtl_stride = 1;
      }
#  ifdef MPI_ENABLED
      std::vector<std::size_t> sizes_g(metadomain.globalNdomains());
      MPI_Allgather(
        &size, 1, MPI_UNSIGNED_LONG, sizes_g.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
      const auto total_size = std::accumulate(sizes_g.begin(), sizes_g.end(), (std::size_t)0);
      auto       offset     = (std::size_t)0;
      for (auto i { 0 }; i < metadomain.localDomain()->mpiRank(); ++i) {
        offset += sizes_g[i];
      }
#  else    // not MPI_ENABLED
      const auto offset     = (std::size_t)0;
      const auto total_size = size;
      (void)metadomain;
#  endif

      if (m_id == PrtlID::X) {
        // phi is treated separately for 2D non-Minkowski metric
#  ifndef MINKOWSKI_METRIC
        const auto dmax = (D == Dim2) ? 3 : (short)D;
#  else
        const auto dmax = (short)D;
#  endif
        for (auto d { 0 }; d < dmax; ++d) {
          array_t<real_t*> xi("xi", size);

          Kokkos::parallel_for(
            "ParticlesOutput_Xi",
            Kokkos::RangePolicy<AccelExeSpace, OutputPositions_t>(0, size),
            PreparePrtlQuantities_kernel<D, S>(mblock, prtls, xi, prtl_stride, d));

          auto xi_host = Kokkos::create_mirror_view(xi);
          Kokkos::deep_copy(xi_host, xi);
          auto varname = "X" + std::to_string(d + 1) + "_" + std::to_string(s);
          auto var     = io.InquireVariable<real_t>(varname);
          var.SetShape({ total_size });
          var.SetSelection({ { offset }, { size } });
          writer.Put<real_t>(var, xi_host);
        }
      } else if (m_id == PrtlID::U) {
        for (auto d { 0 }; d < 3; ++d) {
          array_t<real_t*> ui("ui", size);

          if (params.outputAsIs()) {
            auto prtl_ui = (d == 0) ? prtls.ux1 : ((d == 1) ? prtls.ux2 : prtls.ux3);
            Kokkos::parallel_for(
              "ParticlesOutput_Ui",
              Kokkos::RangePolicy<AccelExeSpace>(0, size),
              Lambda(index_t p) { ui(p) = prtl_ui(p * prtl_stride); });
          } else {
            Kokkos::parallel_for(
              "ParticlesOutput_Ui",
              Kokkos::RangePolicy<AccelExeSpace, OutputVelocities_t>(0, size),
              PreparePrtlQuantities_kernel<D, S>(mblock, prtls, ui, prtl_stride, d));
          }

          auto ui_host = Kokkos::create_mirror_view(ui);
          Kokkos::deep_copy(ui_host, ui);
          auto varname = "U" + std::to_string(d + 1) + "_" + std::to_string(s);
          auto var     = io.InquireVariable<real_t>(varname);
          var.SetShape({ total_size });
          var.SetSelection({ { offset }, { size } });
          writer.Put<real_t>(var, ui_host);
        }
      } else if (m_id == PrtlID::W) {
        array_t<real_t*> w("w", size);
        Kokkos::parallel_for(
          "ParticlesOutput_Ui",
          Kokkos::RangePolicy<AccelExeSpace>(0, size),
          Lambda(index_t p) { w(p) = prtls.weight(p * prtl_stride); });
        auto w_host = Kokkos::create_mirror_view(w);
        Kokkos::deep_copy(w_host, w);
        auto varname = "W_" + std::to_string(s);
        auto var     = io.InquireVariable<real_t>(varname);
        var.SetShape({ total_size });
        var.SetSelection({ { offset }, { size } });
        writer.Put<real_t>(var, w_host);
      }
    }
  }

}    // namespace ntt

#endif