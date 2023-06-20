#include "wrapper.h"

#include "sim_params.h"

#include "io/output.h"
#include "meshblock/fields.h"
#include "meshblock/meshblock.h"
#include "utils/utils.h"

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#endif

#include <string>
#include <vector>

namespace ntt {

  void OutputField::initialize(const SimulationEngine& S) {
    if (S == GRPICEngine) {
      prepare_flag = is_gr_aux_field() ? PrepareOutput_ConvertToPhysCov
                                       : PrepareOutput_ConvertToPhysCntrv;
    } else if (S == PICEngine) {
      prepare_flag = PrepareOutput_ConvertToHat;
    }

    if (is_field()) {
      NTTHostErrorIf(comp.size() != 3, "Fields are always 3 components for output");
      if (is_efield()) {
        address.push_back(em::ex1);
        address.push_back(em::ex2);
        address.push_back(em::ex3);
        interp_flag = PrepareOutput_InterpToCellCenterFromEdges;
      } else {
        address.push_back(em::bx1);
        address.push_back(em::bx2);
        address.push_back(em::bx3);
        interp_flag = PrepareOutput_InterpToCellCenterFromFaces;
      }
    } else if (is_gr_aux_field()) {
      if (S == GRPICEngine) {
        NTTHostErrorIf(comp.size() != 3, "Aux fields are always 3 components for output");
        if (is_efield()) {
          address.push_back(em::ex1);
          address.push_back(em::ex2);
          address.push_back(em::ex3);
          interp_flag = PrepareOutput_InterpToCellCenterFromEdges;
        } else {
          address.push_back(em::hx1);
          address.push_back(em::hx2);
          address.push_back(em::hx3);
          interp_flag = PrepareOutput_InterpToCellCenterFromFaces;
        }
      } else {
        NTTHostError("GRPICEngine is required for GR aux fields");
      }
    } else if (is_current()) {
      NTTHostErrorIf(comp.size() != 3, "Currents are always 3 components for output");
      interp_flag = PrepareOutput_InterpToCellCenterFromEdges;
      address.push_back(cur::jx1);
      address.push_back(cur::jx2);
      address.push_back(cur::jx3);
    } else if (is_moment()) {
      NTTHostErrorIf(comp.size() > 3, "Cannot output more than 3 components for moments");
      for (std::size_t i { 0 }; i < comp.size(); ++i) {
        address.push_back(i);
      }
    } else if (is_vpotential()) {
      address.push_back(0);
      NTTHostErrorIf(comp.size() != 1,
                     "Vector potential is always 1 components for output, but given "
                       + std::to_string(comp.size()));
    } else {
      NTTHostError("Unrecognized field type for output");
    }
  }

  [[nodiscard]] auto OutputField::name(const int& i) const -> std::string {
    std::string myname { m_name };
    for (auto& cc : comp[i]) {
#ifdef MINKOWSKI_METRIC
      myname += (cc == 0 ? "t" : (cc == 1 ? "x" : (cc == 2 ? "y" : "z")));
#else
      myname += std::to_string(cc);
#endif
    }
    if (species.size() > 0) {
      myname += "_";
      for (auto& s : species) {
        myname += std::to_string(s);
        myname += "_";
      }
      myname.pop_back();
    }
    return myname;
  }

  template <Dimension D, SimulationEngine S>
  void OutputField::compute(const SimulationParams& params, Meshblock<D, S>& mblock) {
    if (is_field()) {
      Kokkos::deep_copy(mblock.bckp, mblock.em);
      mblock.template PrepareFieldsForOutput<6, 6>(
        mblock.em, mblock.bckp, address[0], address[1], address[2], interp_flag | prepare_flag);
    } else if (is_gr_aux_field()) {
      Kokkos::deep_copy(mblock.bckp, mblock.em);
      mblock.template PrepareFieldsForOutput<6, 6>(
        mblock.aux, mblock.bckp, address[0], address[1], address[2], interp_flag | prepare_flag);
    } else if (is_current()) {
      Kokkos::deep_copy(mblock.buff, mblock.cur);
      mblock.template PrepareFieldsForOutput<3, 3>(
        mblock.cur, mblock.buff, address[0], address[1], address[2], interp_flag | prepare_flag);
    } else if (is_moment()) {
      for (std::size_t i { 0 }; i < comp.size(); ++i) {
        mblock.ComputeMoments(params,
                              m_id,
                              comp[i],
                              species,
                              address[i],
                              id() == FieldID::Nppc ? 0 : params.outputMomSmooth());
      }
    } else if (is_vpotential()) {
      mblock.ComputeVectorPotential(mblock.bckp, address[0]);
    }
  }

#ifdef OUTPUT_ENABLED

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
                        Meshblock<D, S>&        mblock) const {
    if (is_field() || is_gr_aux_field() || is_vpotential()) {
      for (std::size_t i { 0 }; i < address.size(); ++i) {
        PutField<D, 6>(io, writer, name(i), mblock.bckp, address[i]);
      }
    } else if (is_current() || is_moment()) {
      for (std::size_t i { 0 }; i < address.size(); ++i) {
        PutField<D, 3>(io, writer, name(i), mblock.buff, address[i]);
      }
    }
  }
#endif

}    // namespace ntt
