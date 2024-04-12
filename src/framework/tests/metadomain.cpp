#include "framework/logistics/metadomain.h"

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "utils/error.h"
// #include "utils/plog.h"

#include "metrics/minkowski.h"
#include "metrics/qspherical.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

using namespace ntt;

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    {
      const std::vector<std::size_t> res { 64, 32 };
      const boundaries_t<real_t>     extent {
            {1.0,         10.0},
            {0.0, constant::PI}
      };
      const boundaries_t<FldsBC> fldsbc {
        {FldsBC::ATMOSPHERE, FldsBC::ABSORB},
        {      FldsBC::AXIS,   FldsBC::AXIS}
      };
      const boundaries_t<PrtlBC> prtlbc {
        {PrtlBC::ATMOSPHERE, PrtlBC::ABSORB},
        {      PrtlBC::AXIS,   PrtlBC::AXIS}
      };
      const std::map<std::string, real_t> params {
        {"r0",         -ONE},
        { "h", (real_t)0.25}
      };
      Metadomain<QSpherical<Dim::_2D>> metadomain {
        4, {-1, -1},
         res, extent, fldsbc, prtlbc, params
      };
      std::size_t nx1 { 0 }, nx2 { 0 };
      for (unsigned int idx = 0; idx < 4; ++idx) {
        auto self  = metadomain.idx2subdomain(idx);
        nx1       += self.mesh.n_active(0);
        nx2       += self.mesh.n_active(1);
        raise::ErrorIf(self.index() != idx, "Domain::index() failed", HERE);
        if (idx % 2 == 0) {
          raise::ErrorIf(self.offset_ndomains()[0] != 0,
                         "Domain::offset_ndomains() failed",
                         HERE);
          raise::ErrorIf(self.mesh.flds_bc_in({ -1, 0 }) != FldsBC::ATMOSPHERE,
                         "Mesh::flds_bc_in() failed",
                         HERE);
          raise::ErrorIf(self.mesh.prtl_bc_in({ -1, 0 }) != PrtlBC::ATMOSPHERE,
                         "Mesh::prtl_bc_in() failed",
                         HERE);
          raise::ErrorIf(self.mesh.flds_bc_in({ +1, 0 }) != FldsBC::SYNC,
                         "Mesh::flds_bc_in() failed",
                         HERE);
          raise::ErrorIf(self.mesh.prtl_bc_in({ +1, 0 }) != PrtlBC::SYNC,
                         "Mesh::prtl_bc_in() failed",
                         HERE);
        } else {
          raise::ErrorIf(self.offset_ndomains()[0] != 1,
                         "Domain::offset_ndomains() failed",
                         HERE);
          raise::ErrorIf(self.mesh.flds_bc_in({ +1, 0 }) != FldsBC::ABSORB,
                         "Mesh::flds_bc_in() failed",
                         HERE);
          raise::ErrorIf(self.mesh.prtl_bc_in({ +1, 0 }) != PrtlBC::ABSORB,
                         "Mesh::prtl_bc_in() failed",
                         HERE);
          raise::ErrorIf(self.mesh.flds_bc_in({ -1, 0 }) != FldsBC::SYNC,
                         "Mesh::flds_bc_in() failed",
                         HERE);
          raise::ErrorIf(self.mesh.prtl_bc_in({ -1, 0 }) != PrtlBC::SYNC,
                         "Mesh::prtl_bc_in() failed",
                         HERE);
        }
        if (idx < 2) {
          raise::ErrorIf(self.offset_ndomains()[1] != 0,
                         "Domain::offset_ndomains() failed",
                         HERE);
          raise::ErrorIf(self.mesh.flds_bc_in({ 0, -1 }) != FldsBC::AXIS,
                         "Mesh::flds_bc_in() failed",
                         HERE);
          raise::ErrorIf(self.mesh.prtl_bc_in({ 0, -1 }) != PrtlBC::AXIS,
                         "Mesh::prtl_bc_in() failed",
                         HERE);
          raise::ErrorIf(self.mesh.flds_bc_in({ 0, +1 }) != FldsBC::SYNC,
                         "Mesh::flds_bc_in() failed",
                         HERE);
          raise::ErrorIf(self.mesh.prtl_bc_in({ 0, +1 }) != PrtlBC::SYNC,
                         "Mesh::prtl_bc_in() failed",
                         HERE);
        } else {
          raise::ErrorIf(self.offset_ndomains()[1] != 1,
                         "Domain::offset_ndomains() failed",
                         HERE);
          raise::ErrorIf(self.mesh.flds_bc_in({ 0, +1 }) != FldsBC::AXIS,
                         "Mesh::flds_bc_in() failed",
                         HERE);
          raise::ErrorIf(self.mesh.prtl_bc_in({ 0, +1 }) != PrtlBC::AXIS,
                         "Mesh::prtl_bc_in() failed",
                         HERE);
          raise::ErrorIf(self.mesh.flds_bc_in({ 0, -1 }) != FldsBC::SYNC,
                         "Mesh::flds_bc_in() failed",
                         HERE);
          raise::ErrorIf(self.mesh.prtl_bc_in({ 0, -1 }) != PrtlBC::SYNC,
                         "Mesh::prtl_bc_in() failed",
                         HERE);
        }
        for (auto& direction : dir::Directions<Dim::_2D>::all) {
          raise::ErrorIf(self.neighbor_in(direction)->neighbor_in(direction) == &self,
                         "Domain::neighbor_in() failed",
                         HERE);
          raise::ErrorIf(self.neighbor_in(direction)->neighbor_in(-direction) ==
                           &self,
                         "Domain::neighbor_in() failed",
                         HERE);
        }
      }
      raise::ErrorIf(nx1 != 2 * res[0], "Mesh::n_active() failed", HERE);
      raise::ErrorIf(nx2 != 2 * res[1], "Mesh::n_active() failed", HERE);
    }
  } catch (const std::exception&) {
    Kokkos::finalize();
    return -1;
  }
  Kokkos::finalize();
  return 0;
}