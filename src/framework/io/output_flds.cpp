#ifdef OUTPUT_ENABLED

#  include "wrapper.h"

#  include "fields.h"
#  include "meshblock.h"
#  include "output.h"
#  include "sim_params.h"
#  include "utils.h"

#  ifdef OUTPUT_ENABLED
#    include <adios2.h>
#    include <adios2/cxx11/KokkosView.h>
#  endif

#  include <string>
#  include <vector>

namespace ntt {

  namespace {
    template <Dimension D, int N>
    void PutField(adios2::IO&            io,
                  adios2::Engine&        writer,
                  const std::string&     varname,
                  const ndfield_t<D, N>& field,
                  const int&             comp) {
      auto slice_i1 = Kokkos::ALL;
      auto slice_i2 = Kokkos::ALL;
      auto slice_i3 = Kokkos::ALL;

      auto var      = io.InquireVariable<real_t>(varname);

      if constexpr (D == Dim1) {
        auto slice        = Kokkos::subview(field, slice_i1, comp);
        auto output_field = array_t<real_t*>("output_field", slice.extent(0));
        Kokkos::deep_copy(output_field, slice);
        auto output_field_host = Kokkos::create_mirror_view(output_field);
        Kokkos::deep_copy(output_field_host, output_field);
        writer.Put<real_t>(var, output_field_host);
      } else if constexpr (D == Dim2) {
        auto slice = Kokkos::subview(field, slice_i1, slice_i2, comp);
        auto output_field
          = array_t<real_t**>("output_field", slice.extent(0), slice.extent(1));
        Kokkos::deep_copy(output_field, slice);
        auto output_field_host = Kokkos::create_mirror_view(output_field);
        Kokkos::deep_copy(output_field_host, output_field);
        writer.Put<real_t>(var, output_field_host);
      } else if constexpr (D == Dim3) {
        auto slice        = Kokkos::subview(field, slice_i1, slice_i2, slice_i3, comp);
        auto output_field = array_t<real_t***>(
          "output_field", slice.extent(0), slice.extent(1), slice.extent(2));
        Kokkos::deep_copy(output_field, slice);
        auto output_field_host = Kokkos::create_mirror_view(output_field);
        Kokkos::deep_copy(output_field_host, output_field);
        writer.Put<real_t>(var, output_field_host);
      }
    }
  }    // namespace

  template <Dimension D, SimulationEngine S>
  void OutputField::put(adios2::IO&             io,
                        adios2::Engine&         writer,
                        const SimulationParams& params,
                        Meshblock<D, S>&        mblock) const {
    int prepare_flag = PrepareOutput_Default;
    if constexpr (S == GRPICEngine) {
      // do not convert to hat fields for GRPIC
      // just convert to spherical coordinates
      prepare_flag = PrepareOutput_InterpToCellCenter | PrepareOutput_ConvertToSphCntrv;
      if (m_id == FieldID::E) {
        NTTHostError("Output of E (aux) fields is not supported");
      }
    }
    if ((m_id == FieldID::E) || (m_id == FieldID::B) || (m_id == FieldID::D)) {
      mblock.PrepareFieldsForOutput(prepare_flag);
      ImposeEmptyContent(mblock.bckp_content);
      std::vector<em>      comp_options;
      std::vector<Content> content_options
        = { Content::ex1_hat_int, Content::ex2_hat_int, Content::ex3_hat_int,
            Content::bx1_hat_int, Content::bx2_hat_int, Content::bx3_hat_int };
      if (m_id == FieldID::E || m_id == FieldID::D) {
        comp_options = { em::ex1, em::ex2, em::ex3 };
      } else if (m_id == FieldID::B) {
        comp_options = { em::bx1, em::bx2, em::bx3 };
      }
      for (std::size_t i { 0 }; i < comp.size(); ++i) {
        auto comp_id = comp_options[comp[i][0] - 1];
        PutField<D, 6>(io, writer, name(i), mblock.bckp, (int)(comp_id));
      }
    } else if (m_id == FieldID::H) {
      // for GRPIC write H_phi-field as is
      std::vector<em> comp_options = { em::hx1, em::hx2, em::hx3 };
      Kokkos::deep_copy(mblock.bckp, mblock.aux);
      for (std::size_t i { 0 }; i < comp.size(); ++i) {
        auto comp_id = comp_options[comp[i][0] - 1];
        PutField<D, 6>(io, writer, name(i), mblock.bckp, (int)(comp_id));
      }
    } else if (m_id == FieldID::J) {
      mblock.PrepareCurrentsForOutput();
      ImposeEmptyContent(mblock.cur_content);
      std::vector<cur>     comp_options = { cur::jx1, cur::jx2, cur::jx3 };
      std::vector<Content> content_options
        = { Content::jx1_hat_int, Content::jx2_hat_int, Content::jx3_hat_int };
      for (std::size_t i { 0 }; i < comp.size(); ++i) {
        auto comp_id = comp_options[comp[i][0] - 1];
        PutField<D, 3>(io, writer, name(i), mblock.cur, (int)(comp_id));
      }
    } else {
      for (std::size_t i { 0 }; i < comp.size(); ++i) {
        // no smoothing for FieldID::Nppc
        mblock.ComputeMoments(params,
                              m_id,
                              comp[i],
                              species,
                              i % 3,
                              m_id == FieldID::Nppc ? 0 : params.outputMomSmooth());
        PutField<D, 3>(io, writer, name(i), mblock.buff, i % 3);
      }
    }
  }
}    // namespace ntt

#endif