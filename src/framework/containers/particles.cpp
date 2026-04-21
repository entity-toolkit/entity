#include "framework/containers/particles.h"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "framework/containers/species.h"
#include "kernels/pushers/context.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <string>

namespace ntt {

  template <Dimension D, Coord::type C>
  Particles<D, C>::Particles(spidx_t             index,
                             const std::string&  label,
                             float               m,
                             float               ch,
                             npart_t             maxnpart,
                             timestep_t          clearing_interval,
                             timestep_t          spatial_sorting_interval,
                             ParticlePusherFlags particle_pusher_flags,
                             bool                use_tracking,
                             RadiativeDragFlags  radiative_drag_flags,
                             EmissionTypeFlag    emission_policy_flag,
                             unsigned short      npld_r,
                             unsigned short      npld_i)
    : ParticleSpecies { index,
                        label,
                        m,
                        ch,
                        maxnpart,
                        clearing_interval,
                        spatial_sorting_interval,
                        particle_pusher_flags,
                        use_tracking,
                        radiative_drag_flags,
                        emission_policy_flag,
                        npld_r,
                        npld_i } {

    if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
      i1       = array_t<int*> { label + "_i1", maxnpart };
      dx1      = array_t<prtldx_t*> { label + "_dx1", maxnpart };
      i1_prev  = array_t<int*> { label + "_i1_prev", maxnpart };
      dx1_prev = array_t<prtldx_t*> { label + "_dx1_prev", maxnpart };
    }

    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      i2       = array_t<int*> { label + "_i2", maxnpart };
      dx2      = array_t<prtldx_t*> { label + "_dx2", maxnpart };
      i2_prev  = array_t<int*> { label + "_i2_prev", maxnpart };
      dx2_prev = array_t<prtldx_t*> { label + "_dx2_prev", maxnpart };
    }

    if constexpr (D == Dim::_3D) {
      i3       = array_t<int*> { label + "_i3", maxnpart };
      dx3      = array_t<prtldx_t*> { label + "_dx3", maxnpart };
      i3_prev  = array_t<int*> { label + "_i3_prev", maxnpart };
      dx3_prev = array_t<prtldx_t*> { label + "_dx3_prev", maxnpart };
    }

    ux1 = array_t<real_t*> { label + "_ux1", maxnpart };
    ux2 = array_t<real_t*> { label + "_ux2", maxnpart };
    ux3 = array_t<real_t*> { label + "_ux3", maxnpart };

    weight = array_t<real_t*> { label + "_w", maxnpart };

    tag = array_t<short*> { label + "_tag", maxnpart };

    if (npld_r > 0) {
      pld_r = array_t<real_t**> { label + "_pld_r", maxnpart, npld_r };
    }
    if (npld_i > 0) {
      pld_i = array_t<npart_t**> { label + "_pld_i", maxnpart, npld_i };
    }

    if ((D == Dim::_2D) && (C != Coord::Cartesian)) {
      phi = array_t<real_t*> { label + "_phi", maxnpart };
    }
  }

  template <Dimension D, Coord::type C>
  auto Particles<D, C>::PusherKernelArrays() -> kernel::PusherArrays {
    kernel::PusherArrays pusher_arrays {};
    pusher_arrays.sp       = index();
    pusher_arrays.i1       = i1;
    pusher_arrays.i2       = i2;
    pusher_arrays.i3       = i3;
    pusher_arrays.i1_prev  = i1_prev;
    pusher_arrays.i2_prev  = i2_prev;
    pusher_arrays.i3_prev  = i3_prev;
    pusher_arrays.dx1      = dx1;
    pusher_arrays.dx2      = dx2;
    pusher_arrays.dx3      = dx3;
    pusher_arrays.dx1_prev = dx1_prev;
    pusher_arrays.dx2_prev = dx2_prev;
    pusher_arrays.dx3_prev = dx3_prev;
    pusher_arrays.ux1      = ux1;
    pusher_arrays.ux2      = ux2;
    pusher_arrays.ux3      = ux3;
    pusher_arrays.phi      = phi;
    pusher_arrays.weight   = weight;
    pusher_arrays.tag      = tag;
    return pusher_arrays;
  }

  template struct Particles<Dim::_1D, Coord::Cartesian>;
  template struct Particles<Dim::_2D, Coord::Cartesian>;
  template struct Particles<Dim::_3D, Coord::Cartesian>;
  template struct Particles<Dim::_2D, Coord::Spherical>;
  template struct Particles<Dim::_3D, Coord::Spherical>;
  template struct Particles<Dim::_2D, Coord::Qspherical>;
  template struct Particles<Dim::_3D, Coord::Qspherical>;

} // namespace ntt
