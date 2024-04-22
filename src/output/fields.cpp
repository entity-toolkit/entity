#include "output/fields.h"

#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"

#include "output/utils/interpret_prompt.h"

#include <Kokkos_Core.hpp>

#include <cctype>
#include <string>
#include <vector>

using namespace ntt;

namespace out {

  OutputField::OutputField(const SimEngine& S, const std::string& name)
    : m_name { name } {
    // determine the field ID
    const auto pos = name.find("_");
    auto name_raw  = (pos == std::string::npos) ? name : name.substr(0, pos);
    name_raw       = name_raw.substr(0, name_raw.find_first_of("0123ijxyzt"));
    m_id           = FldsID::pick(fmt::toLower(name_raw).c_str());
    // determine the species and components to output
    if (is_moment()) {
      species = InterpretSpecies(name);
    } else {
      species = {};
    }
    if (is_field() || is_current()) {
      // always write all the field/current components
      comp = { { 1 }, { 2 }, { 3 } };
    } else if (id() == FldsID::A) {
      // only write A3
      comp = { { 3 } };
    } else if (id() == FldsID::T) {
      // energy-momentum tensor
      comp = InterpretComponents({ name.substr(1, 1), name.substr(2, 1) });
    } else {
      // scalar (Rho, divE, etc.)
      comp = {};
    }
    // data preparation flags
    if (not is_moment()) {
      if (S == SimEngine::SRPIC) {
        prepare_flag = PrepareOutput::ConvertToHat;
      } else {
        prepare_flag = is_gr_aux_field() ? PrepareOutput::ConvertToPhysCov
                                         : PrepareOutput::ConvertToPhysCntrv;
      }
    }
    // interpolation flags
    if (is_current() || is_efield()) {
      interp_flag = PrepareOutput::InterpToCellCenterFromEdges;
    } else if (is_field() || is_gr_aux_field()) {
      interp_flag = PrepareOutput::InterpToCellCenterFromFaces;
    } else if (not(is_moment() || is_vpotential() || is_divergence())) {
      raise::Error("Unrecognized field type for output", HERE);
    }
  }

  // template <Dimension D, SimulationEngine S>
  // void OutputField::compute(const SimulationParams& params,
  //                           Meshblock<D, S>&        mblock) const {
  //   auto slice_i1 = Kokkos::ALL;
  //   auto slice_i2 = Kokkos::ALL;
  //   auto slice_i3 = Kokkos::ALL;

  //   if (is_field()) {
  //     auto slice_comp = std::make_pair(address[0], address[2] + 1);
  //     if constexpr (D == Dim1) {
  //       Kokkos::deep_copy(Kokkos::subview(mblock.bckp, slice_i1, slice_comp),
  //                         Kokkos::subview(mblock.em, slice_i1, slice_comp));
  //     } else if constexpr (D == Dim2) {
  //       Kokkos::deep_copy(
  //         Kokkos::subview(mblock.bckp, slice_i1, slice_i2, slice_comp),
  //         Kokkos::subview(mblock.em, slice_i1, slice_i2, slice_comp));
  //     } else if constexpr (D == Dim3) {
  //       Kokkos::deep_copy(
  //         Kokkos::subview(mblock.bckp, slice_i1, slice_i2, slice_i3, slice_comp),
  //         Kokkos::subview(mblock.em, slice_i1, slice_i2, slice_i3, slice_comp));
  //     }
  //     if (!params.outputAsIs()) {
  //       mblock.template PrepareFieldsForOutput<6, 6>(mblock.em,
  //                                                    mblock.bckp,
  //                                                    address[0],
  //                                                    address[1],
  //                                                    address[2],
  //                                                    interp_flag | prepare_flag);
  //     }
  //   } else if (is_gr_aux_field()) {
  //     auto slice_comp = std::make_pair(address[0], address[2] + 1);
  //     if constexpr (D == Dim1) {
  //       Kokkos::deep_copy(Kokkos::subview(mblock.bckp, slice_i1, slice_comp),
  //                         Kokkos::subview(mblock.aux, slice_i1, slice_comp));
  //     } else if constexpr (D == Dim2) {
  //       Kokkos::deep_copy(
  //         Kokkos::subview(mblock.bckp, slice_i1, slice_i2, slice_comp),
  //         Kokkos::subview(mblock.aux, slice_i1, slice_i2, slice_comp));
  //     } else if constexpr (D == Dim3) {
  //       Kokkos::deep_copy(
  //         Kokkos::subview(mblock.bckp, slice_i1, slice_i2, slice_i3, slice_comp),
  //         Kokkos::subview(mblock.aux, slice_i1, slice_i2, slice_i3, slice_comp));
  //     }
  //     if (!params.outputAsIs()) {
  //       mblock.template PrepareFieldsForOutput<6, 6>(mblock.aux,
  //                                                    mblock.bckp,
  //                                                    address[0],
  //                                                    address[1],
  //                                                    address[2],
  //                                                    interp_flag | prepare_flag);
  //     }
  //   } else if (is_current()) {
  //     Kokkos::deep_copy(mblock.buff, mblock.cur);
  //     if (!params.outputAsIs()) {
  //       mblock.template PrepareFieldsForOutput<3, 3>(mblock.cur,
  //                                                    mblock.buff,
  //                                                    address[0],
  //                                                    address[1],
  //                                                    address[2],
  //                                                    interp_flag | prepare_flag);
  //     }
  //   } else if (is_moment()) {
  //     for (std::size_t i { 0 }; i < comp.size(); ++i) {
  //       mblock.ComputeMoments(params,
  //                             m_id,
  //                             comp[i],
  //                             species,
  //                             address[i],
  //                             id() == FieldID::Nppc ? 0 : params.outputMomSmooth());
  //     }
  //   } else if (is_vpotential()) {
  //     mblock.ComputeVectorPotential(mblock.bckp, address[0]);
  //   } else if (is_divergence()) {
  //     mblock.ComputeDivergenceED(mblock.buff, address[0]);
  //   } else {
  //     NTTHostError("Unrecognized field type for output");
  //   }
  // }

} // namespace out