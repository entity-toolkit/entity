#include "framework/domain/metadomain.h"

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "utils/error.h"

#include "metrics/minkowski.h"
#include "metrics/qspherical.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    using namespace ntt;
    using namespace metric;
    {
      const std::vector<ncells_t> res { 64, 32 };
      const boundaries_t<real_t>  extent {
         { 1.0,         10.0 },
         { 0.0, constant::PI }
      };
      const boundaries_t<FldsBC> fldsbc {
        { FldsBC::ATMOSPHERE, FldsBC::MATCH },
        {       FldsBC::AXIS,  FldsBC::AXIS }
      };
      const boundaries_t<PrtlBC> prtlbc {
        { PrtlBC::ATMOSPHERE, PrtlBC::ABSORB },
        {       PrtlBC::AXIS,   PrtlBC::AXIS }
      };
      const std::map<std::string, real_t> params {
        { "r0",         -ONE },
        {  "h", (real_t)0.25 }
      };
#if defined(OUTPUT_ENABLED)
      Metadomain<SimEngine::SRPIC, QSpherical<Dim::_2D>> metadomain {
        4u, { -1, -1 },
         res, extent, fldsbc, prtlbc, params, {}
      };
#else
      Metadomain<SimEngine::SRPIC, QSpherical<Dim::_2D>> metadomain {
        4u, { -1, -1 },
         res, extent, fldsbc, prtlbc, params, {}
      };
#endif
      std::size_t nx1 { 0 }, nx2 { 0 };
      raise::ErrorIf(metadomain.mesh().n_active() != res,
                     "Mesh::n_active() failed",
                     HERE);
      raise::ErrorIf(metadomain.mesh().extent() != extent,
                     "Mesh::extent() failed",
                     HERE);
      raise::ErrorIf(metadomain.mesh().flds_bc() != fldsbc,
                     "Mesh::flds_bc() failed",
                     HERE);
      raise::ErrorIf(metadomain.mesh().prtl_bc() != prtlbc,
                     "Mesh::prtl_bc() failed",
                     HERE);
      for (unsigned int idx = 0; idx < 4; ++idx) {
        auto& self  = metadomain.subdomain(idx);
        nx1        += self.mesh.n_active(in::x1);
        nx2        += self.mesh.n_active(in::x2);
        raise::ErrorIf(self.index() != idx, "Domain::index() failed", HERE);

        // check field allocations
        raise::ErrorIf(self.fields.em.extent(0) != self.mesh.n_all(in::x1),
                       "Domain::fields.em(0) failed",
                       HERE);
        raise::ErrorIf(self.fields.em.extent(1) != self.mesh.n_all(in::x2),
                       "Domain::fields.em(1) failed",
                       HERE);
        raise::ErrorIf(self.fields.em.extent(2) != 6,
                       "Domain::fields.em(2) failed",
                       HERE);

        // check current allocations
        raise::ErrorIf(self.fields.cur.extent(0) != self.mesh.n_all(in::x1),
                       "Domain::fields.cur(0) failed",
                       HERE);
        raise::ErrorIf(self.fields.cur.extent(1) != self.mesh.n_all(in::x2),
                       "Domain::fields.cur(1) failed",
                       HERE);
        raise::ErrorIf(self.fields.cur.extent(2) != 3,
                       "Domain::fields.cur(2) failed",
                       HERE);

        // check em0 allocations (should be unallocated)
        raise::ErrorIf(self.fields.em0.extent(0) != 0,
                       "Domain::fields.em0(0) failed",
                       HERE);
        raise::ErrorIf(self.fields.em0.extent(1) != 0,
                       "Domain::fields.em0(1) failed",
                       HERE);
        raise::ErrorIf(self.fields.aux.extent(0) != 0,
                       "Domain::fields.aux(0) failed",
                       HERE);
        raise::ErrorIf(self.fields.aux.extent(1) != 0,
                       "Domain::fields.aux(1) failed",
                       HERE);
        raise::ErrorIf(self.fields.cur0.extent(0) != 0,
                       "Domain::fields.cur0(0) failed",
                       HERE);
        raise::ErrorIf(self.fields.cur0.extent(1) != 0,
                       "Domain::fields.cur0(1) failed",
                       HERE);

        // check boundaries and offsets
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
          raise::ErrorIf(self.mesh.flds_bc_in({ +1, 0 }) != FldsBC::MATCH,
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
          const auto& neighbor1 = metadomain.subdomain(
            self.neighbor_idx_in(direction));
          const auto& neighbor2 = metadomain.subdomain(
            neighbor1.neighbor_idx_in(direction));
          const auto& neighbor3 = metadomain.subdomain(
            neighbor1.neighbor_idx_in(-direction));
          raise::ErrorIf(neighbor2.index() != self.index(),
                         "Domain::neighbor_in() failed",
                         HERE);
          raise::ErrorIf(neighbor3.index() != self.index(),
                         "Domain::neighbor_in() failed",
                         HERE);
        }
      }
      raise::ErrorIf(nx1 != 2 * res[0], "Mesh::n_active() failed", HERE);
      raise::ErrorIf(nx2 != 2 * res[1], "Mesh::n_active() failed", HERE);
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return -1;
  }
  Kokkos::finalize();
  return 0;
}
