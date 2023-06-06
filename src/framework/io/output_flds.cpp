#ifdef OUTPUT_ENABLED

#  include "wrapper.h"

#  include "sim_params.h"

#  include "io/output.h"
#  include "meshblock/fields.h"
#  include "meshblock/meshblock.h"
#  include "utils/utils.h"

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
    const auto is_moment = (id() == FieldID::T || id() == FieldID::Rho || id() == FieldID::Nppc
                            || id() == FieldID::N);
    const auto is_field
      = (id() == FieldID::E || id() == FieldID::B || id() == FieldID::D || id() == FieldID::H);
    const auto is_current      = (id() == FieldID::J);
    const auto is_efield       = (id() == FieldID::E || id() == FieldID::D);
    const auto is_gr_aux_field = (id() == FieldID::E || id() == FieldID::H);
    const auto is_vpotential   = (id() == FieldID::A);

    auto       prepare_flag    = PrepareOutput_None;
    if constexpr (S == GRPICEngine) {
      prepare_flag
        = is_gr_aux_field ? PrepareOutput_ConvertToPhysCov : PrepareOutput_ConvertToPhysCntrv;
    } else if constexpr (S == PICEngine) {
      prepare_flag = PrepareOutput_ConvertToHat;
    }

    if (is_field) {
      NTTHostErrorIf(comp.size() != 3, "Fields are always 3 components for output");
      std::vector<int> components;
      auto             interp_flag = PrepareOutput_None;
      if (is_efield) {
        components.push_back(em::ex1);
        components.push_back(em::ex2);
        components.push_back(em::ex3);
        interp_flag = PrepareOutput_InterpToCellCenterFromEdges;
      } else {
        components.push_back(em::bx1);
        components.push_back(em::bx2);
        components.push_back(em::bx3);
        interp_flag = PrepareOutput_InterpToCellCenterFromFaces;
      }
      Kokkos::deep_copy(mblock.bckp, mblock.em);
      mblock.template PrepareFieldsForOutput<6, 6>(mblock.em,
                                                   mblock.bckp,
                                                   components[0],
                                                   components[1],
                                                   components[2],
                                                   interp_flag | prepare_flag);
      for (auto i { 0 }; i < 3; ++i) {
        PutField<D, 6>(io, writer, name(i), mblock.bckp, components[i]);
      }
    } else if (is_gr_aux_field) {
      if constexpr (S == GRPICEngine) {
        NTTHostErrorIf(comp.size() != 3, "Aux fields are always 3 components for output");
        std::vector<int> components;
        auto             interp_flag = PrepareOutput_None;
        if (is_efield) {
          components.push_back(em::ex1);
          components.push_back(em::ex2);
          components.push_back(em::ex3);
          interp_flag = PrepareOutput_InterpToCellCenterFromEdges;
        } else {
          components.push_back(em::hx1);
          components.push_back(em::hx2);
          components.push_back(em::hx3);
          interp_flag = PrepareOutput_InterpToCellCenterFromFaces;
        }
        Kokkos::deep_copy(mblock.bckp, mblock.em);
        mblock.template PrepareFieldsForOutput<6, 6>(mblock.aux,
                                                     mblock.bckp,
                                                     components[0],
                                                     components[1],
                                                     components[2],
                                                     interp_flag | prepare_flag);
        for (auto i { 0 }; i < 3; ++i) {
          PutField<D, 6>(io, writer, name(i), mblock.bckp, components[i]);
        }
      } else {
        NTTHostError("GRPICEngine is required for GR aux fields");
      }
    } else if (is_current) {
      NTTHostErrorIf(comp.size() != 3, "Currents are always 3 components for output");
      Kokkos::deep_copy(mblock.buff, mblock.cur);
      mblock.template PrepareFieldsForOutput<3, 3>(mblock.cur,
                                                   mblock.buff,
                                                   cur::jx1,
                                                   cur::jx2,
                                                   cur::jx3,
                                                   PrepareOutput_InterpToCellCenterFromEdges
                                                     | prepare_flag);
      std::vector<int> components = { cur::jx1, cur::jx2, cur::jx3 };
      for (auto i { 0 }; i < 3; ++i) {
        PutField<D, 3>(io, writer, name(i), mblock.buff, components[i]);
      }
    } else if (is_moment) {
      for (std::size_t i { 0 }; i < comp.size(); ++i) {
        // !TODO: can perhaps do this better
        // no smoothing for FieldID::Nppc
        mblock.ComputeMoments(params,
                              m_id,
                              comp[i],
                              species,
                              i % 3,
                              m_id == FieldID::Nppc ? 0 : params.outputMomSmooth());
        PutField<D, 3>(io, writer, name(i), mblock.buff, i % 3);
      }
    } else if (is_vpotential) {
      NTTHostErrorIf(comp.size() != 1,
                     "Vector potential is always 1 components for output, but given "
                       + std::to_string(comp.size()));
      mblock.ComputeVectorPotential(mblock.bckp, 0);
      PutField<D, 6>(io, writer, name(0), mblock.bckp, 0);
    } else {
      NTTHostError("Unrecognized field type for output");
    }
  }
}    // namespace ntt

#endif