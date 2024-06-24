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

} // namespace out
