#include "enums.h"
#include "global.h"

#include "utils/error.h"

#include "framework/containers/particles.h"
#include "framework/domain/grid.h"

#include <iostream>

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    {
      // 2D
      auto grid = ntt::Grid<Dim::_2D> {
        {           10u,             30u },
        { { -5.0, 5.0 }, { -15.0, 15.0 } }
      };
      auto prtls = ntt::Particles<Dim::_2D, ntt::Coord::Cart>(
        1,
        "test",
        1.0f,
        1.0f,
        100u,
        0u,
        ntt::ParticlePusher::BORIS,
        false,
        ntt::RadiativeDrag::NONE,
        ntt::EmissionType::NONE,
        2u,
        1u);
      auto& i1_p     = prtls.i1;
      auto& i2_p     = prtls.i2;
      auto& tag_p    = prtls.tag;
      auto& weight_p = prtls.weight;
      auto& pld_r    = prtls.pld_r;
      auto& pld_i    = prtls.pld_i;
      Kokkos::parallel_for(
        "InitParticles",
        prtls.maxnpart(),
        Lambda(index_t p) {
          if (p < 66u) {
            tag_p(p) = (p % 10u == 0u) ? ntt::ParticleTag::dead
                                       : ntt::ParticleTag::alive;
            if (p % 4u == 0u) {
              i1_p(p)     = 8u;
              i2_p(p)     = 2u;
              weight_p(p) = 0.0;
            } else if (p % 4u == 1u) {
              i1_p(p)     = 2u;
              i2_p(p)     = 8u;
              weight_p(p) = 1.0;
            } else if (p % 4u == 2u) {
              i1_p(p)     = 5u;
              i2_p(p)     = 15u;
              weight_p(p) = 2.0;
            } else {
              i1_p(p)     = 0u;
              i2_p(p)     = 23u;
              weight_p(p) = 3.0;
            }
            pld_r(p, 0) = weight_p(p) + 0.5;
            pld_r(p, 1) = weight_p(p) + 10.5;
            pld_i(p, 0) = static_cast<npart_t>(weight_p(p) + 10.0);
          } else {
            tag_p(p) = ntt::ParticleTag::dead;
          }
          if (tag_p(p) == ntt::ParticleTag::dead) {
            weight_p(p) = -1.0;
          }
        });
      prtls.set_npart(66u);

      prtls.SortSpatially(grid);

      auto weight_h = Kokkos::create_mirror_view(prtls.weight);
      auto pld_r_h  = Kokkos::create_mirror_view(prtls.pld_r);
      auto pld_i_h  = Kokkos::create_mirror_view(prtls.pld_i);
      Kokkos::deep_copy(weight_h, prtls.weight);
      Kokkos::deep_copy(pld_r_h, prtls.pld_r);
      Kokkos::deep_copy(pld_i_h, prtls.pld_i);

      for (auto p { 0u }; p < 75u; ++p) {
        if (p < 16u) {
          raise::ErrorIf(weight_h(p) != 3.0, "error in sorting particles", HERE);
        } else if (p < 33u) {
          raise::ErrorIf(weight_h(p) != 1.0, "error in sorting particles", HERE);
        } else if (p < 46u) {
          raise::ErrorIf(weight_h(p) != 2.0, "error in sorting particles", HERE);
        } else if (p < 59u) {
          raise::ErrorIf(weight_h(p) != 0.0, "error in sorting particles", HERE);
        } else {
          raise::ErrorIf(weight_h(p) != -1.0, "error in sorting particles", HERE);
        }
        if (p < 59u) {
          raise::ErrorIf(pld_r_h(p, 0) != weight_h(p) + 0.5,
                         "error in sorting particle real payload 0",
                         HERE);
          raise::ErrorIf(pld_r_h(p, 1) != weight_h(p) + 10.5,
                         "error in sorting particle real payload 1",
                         HERE);
          raise::ErrorIf(pld_i_h(p, 0) != static_cast<npart_t>(weight_h(p) + 10.0),
                         "error in sorting particle integer payload 0",
                         HERE);
        }
      }
    }
    {
      // 3D
      auto grid = ntt::Grid<Dim::_3D> {
        {            6u,            7u,            8u },
        { { -3.0, 3.0 }, { -3.5, 3.5 }, { -4.0, 4.0 } }
      };
      auto prtls = ntt::Particles<Dim::_3D, ntt::Coord::Cart>(
        1,
        "test",
        1.0f,
        1.0f,
        100u,
        0u,
        ntt::ParticlePusher::BORIS,
        false,
        ntt::RadiativeDrag::NONE,
        ntt::EmissionType::NONE,
        0u,
        0u);
      auto& i1_p     = prtls.i1;
      auto& i2_p     = prtls.i2;
      auto& i3_p     = prtls.i3;
      auto& tag_p    = prtls.tag;
      auto& weight_p = prtls.weight;
      Kokkos::parallel_for(
        "InitParticles",
        prtls.maxnpart(),
        Lambda(index_t p) {
          if (p < 66u) {
            tag_p(p) = (p % 10u == 0u) ? ntt::ParticleTag::dead
                                       : ntt::ParticleTag::alive;
            if (p % 5u == 0u) {
              i1_p(p)     = 3u;
              i2_p(p)     = 2u;
              i3_p(p)     = 7u;
              weight_p(p) = 0.0;
            } else if (p % 5u == 1u) {
              i1_p(p)     = 2u;
              i2_p(p)     = 4u;
              i3_p(p)     = 3u;
              weight_p(p) = 1.0;
            } else if (p % 5u == 2u) {
              i1_p(p)     = 2u;
              i2_p(p)     = 6u;
              i3_p(p)     = 6u;
              weight_p(p) = 2.0;
            } else if (p % 5u == 3u) {
              i1_p(p)     = 3u;
              i2_p(p)     = 6u;
              i3_p(p)     = 6u;
              weight_p(p) = 3.0;
            } else {
              i1_p(p)     = 0u;
              i2_p(p)     = 6u;
              i3_p(p)     = 7u;
              weight_p(p) = 4.0;
            }
          } else {
            tag_p(p) = ntt::ParticleTag::dead;
          }
          if (tag_p(p) == ntt::ParticleTag::dead) {
            weight_p(p) = -1.0;
          }
        });
      prtls.set_npart(66u);

      prtls.SortSpatially(grid);

      auto weight_h = Kokkos::create_mirror_view(prtls.weight);
      Kokkos::deep_copy(weight_h, prtls.weight);
      for (auto p { 0u }; p < 75u; ++p) {
        if (p < 13u) {
          raise::ErrorIf(weight_h(p) != 4.0, "error in sorting particles", HERE);
        } else if (p < 26u) {
          raise::ErrorIf(weight_h(p) != 1.0, "error in sorting particles", HERE);
        } else if (p < 39u) {
          raise::ErrorIf(weight_h(p) != 2.0, "error in sorting particles", HERE);
        } else if (p < 46u) {
          raise::ErrorIf(weight_h(p) != 0.0, "error in sorting particles", HERE);
        } else if (p < 59u) {
          raise::ErrorIf(weight_h(p) != 3.0, "error in sorting particles", HERE);
        } else {
          raise::ErrorIf(weight_h(p) != -1.0, "error in sorting particles", HERE);
        }
      }
      std::cout << std::endl;
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    ntt::GlobalFinalize();
    return 1;
  }
  ntt::GlobalFinalize();
  return 0;
}
