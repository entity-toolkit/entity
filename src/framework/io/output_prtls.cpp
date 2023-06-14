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

namespace ntt {
  template <Dimension D, SimulationEngine S>
  void OutputParticles::put(adios2::IO&             io,
                            adios2::Engine&         writer,
                            const SimulationParams& params,
                            Meshblock<D, S>&        mblock) const {
    for (auto& s : speciesID()) {
      auto prtls     = mblock.particles[s - 1];

      // remove all the dead particles before output
      auto npart_tag = prtls.CountTaggedParticles();
      auto dead_fraction
        = (double)(npart_tag[(short)(ParticleTag::dead)]) / (double)(prtls.npart());
      if (prtls.npart() > 0) {
        prtls.ReshuffleByTags();
        prtls.setNpart(npart_tag[(short)(ParticleTag::alive)]);
      }
      auto prtl_stride = params.outputPrtlStride();
      auto size        = static_cast<std::size_t>(prtls.npart() / prtl_stride);
      if (size == 0 and prtls.npart() > 0) {
        size        = prtls.npart();
        prtl_stride = 1;
      }

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
          var.SetSelection(adios2::Box<adios2::Dims>({}, { size }));
          writer.Put<real_t>(var, xi_host);
        }
      } else if (m_id == PrtlID::U) {
        for (auto d { 0 }; d < 3; ++d) {
          array_t<real_t*> ui("ui", size);

          Kokkos::parallel_for(
            "ParticlesOutput_Ui",
            Kokkos::RangePolicy<AccelExeSpace, OutputVelocities_t>(0, size),
            PreparePrtlQuantities_kernel<D, S>(mblock, prtls, ui, prtl_stride, d));

          auto ui_host = Kokkos::create_mirror_view(ui);
          Kokkos::deep_copy(ui_host, ui);
          auto varname = "U" + std::to_string(d + 1) + "_" + std::to_string(s);
          auto var     = io.InquireVariable<real_t>(varname);
          var.SetSelection(adios2::Box<adios2::Dims>({}, { size }));
          writer.Put<real_t>(var, ui_host);
        }
      } else if (m_id == PrtlID::W) {
        NTTHostError("OutputParticles::put() not implemented for PrtlID::W");
      }
    }
  }

}    // namespace ntt

#endif