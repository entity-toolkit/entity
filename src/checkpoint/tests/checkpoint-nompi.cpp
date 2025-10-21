#include "enums.h"
#include "global.h"

#include "utils/comparators.h"

#include "checkpoint/reader.h"
#include "checkpoint/writer.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#include <filesystem>
#include <iostream>

using namespace ntt;
using namespace checkpoint;

void cleanup() {
  namespace fs = std::filesystem;
  fs::path temp_path { "chck" };
  fs::remove_all(temp_path);
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    constexpr auto nx1    = 10;
    constexpr auto nx1_gh = nx1 + 2 * N_GHOSTS;
    constexpr auto nx2    = 13;
    constexpr auto nx2_gh = nx2 + 2 * N_GHOSTS;
    constexpr auto nx3    = 9;
    constexpr auto nx3_gh = nx3 + 2 * N_GHOSTS;
    constexpr auto i1min  = N_GHOSTS;
    constexpr auto i2min  = N_GHOSTS;
    constexpr auto i3min  = N_GHOSTS;
    constexpr auto i1max  = nx1 + N_GHOSTS;
    constexpr auto i2max  = nx2 + N_GHOSTS;
    constexpr auto i3max  = nx3 + N_GHOSTS;
    constexpr auto npart1 = 100;
    constexpr auto npart2 = 100;

    // init data
    ndfield_t<Dim::_3D, 6> field1 { "fld1", nx1_gh, nx2_gh, nx3_gh };
    ndfield_t<Dim::_3D, 6> field2 { "fld2", nx1_gh, nx2_gh, nx3_gh };

    array_t<int*>    i1 { "i_1", npart1 };
    array_t<real_t*> u1 { "u_1", npart1 };
    array_t<int*>    i2 { "i_2", npart2 };
    array_t<real_t*> u2 { "u_2", npart2 };

    array_t<real_t**> pldr_2 { "pldr_2", npart2, 2 };

    array_t<npart_t**> pldi_1 { "pldi_1", npart1, 1 };
    array_t<npart_t**> pldi_2 { "pldi_2", npart2, 2 };

    {
      // fill data
      Kokkos::parallel_for(
        "fillFlds",
        CreateRangePolicy<Dim::_3D>({ i1min, i2min, i3min },
                                    { i1max, i2max, i3max }),
        Lambda(index_t i1, index_t i2, index_t i3) {
          const auto i1_        = static_cast<real_t>(i1);
          const auto i2_        = static_cast<real_t>(i2);
          const auto i3_        = static_cast<real_t>(i3);
          field1(i1, i2, i3, 0) = i1_ + i2_ + i3_;
          field1(i1, i2, i3, 1) = i1_ * i2_ / i3_;
          field1(i1, i2, i3, 2) = i1_ / i2_ * i3_;
          field1(i1, i2, i3, 3) = i1_ + i2_ - i3_;
          field1(i1, i2, i3, 4) = i1_ * i2_ + i3_;
          field1(i1, i2, i3, 5) = i1_ / i2_ - i3_;
          field2(i1, i2, i3, 0) = -(i1_ + i2_ + i3_);
          field2(i1, i2, i3, 1) = -(i1_ * i2_ / i3_);
          field2(i1, i2, i3, 2) = -(i1_ / i2_ * i3_);
          field2(i1, i2, i3, 3) = -(i1_ + i2_ - i3_);
          field2(i1, i2, i3, 4) = -(i1_ * i2_ + i3_);
          field2(i1, i2, i3, 5) = -(i1_ / i2_ - i3_);
        });
      Kokkos::parallel_for(
        "fillPrtl1",
        npart1,
        Lambda(index_t p) {
          u1(p)        = static_cast<real_t>(p);
          i1(p)        = static_cast<int>(p);
          pldi_1(p, 0) = static_cast<npart_t>(p * 10);
        });
      Kokkos::parallel_for(
        "fillPrtl2",
        npart2,
        Lambda(index_t p) {
          u2(p)        = -static_cast<real_t>(p);
          i2(p)        = -static_cast<int>(p);
          pldr_2(p, 0) = static_cast<real_t>(p);
          pldr_2(p, 1) = static_cast<real_t>(p * 2);
          pldi_2(p, 0) = static_cast<npart_t>(p * 3);
          pldi_2(p, 1) = static_cast<npart_t>(p * 4);
        });
    }

    adios2::ADIOS adios;
    const path_t  checkpoint_path { "chck" };

    {
      // write checkpoint
      Writer writer {};
      writer.init(&adios, checkpoint_path, 0, 0.0, 1);

      writer.defineFieldVariables(SimEngine::GRPIC,
                                  { nx1_gh, nx2_gh, nx3_gh },
                                  { 0, 0, 0 },
                                  { nx1_gh, nx2_gh, nx3_gh });
      writer.defineParticleVariables(Coord::Sph, Dim::_3D, 2, { 0, 2 }, { 1, 2 });

      writer.beginSaving(0, 0.0);

      writer.saveField<Dim::_3D, 6>("em", field1);
      writer.saveField<Dim::_3D, 6>("em0", field2);

      writer.savePerDomainVariable<npart_t>("s1_npart", 1, 0, npart1);
      writer.savePerDomainVariable<npart_t>("s2_npart", 1, 0, npart2);

      writer.saveParticleQuantity<int>("s1_i1", npart1, 0, npart1, i1);
      writer.saveParticleQuantity<real_t>("s1_ux1", npart1, 0, npart1, u1);
      writer.saveParticleQuantity<int>("s2_i1", npart2, 0, npart2, i2);
      writer.saveParticleQuantity<real_t>("s2_ux1", npart2, 0, npart2, u2);

      writer.saveParticlePayloads<real_t>("s2_pld_r", 2, npart2, 0, npart2, pldr_2);

      writer.saveParticlePayloads<npart_t>("s1_pld_i", 1, npart1, 0, npart1, pldi_1);
      writer.saveParticlePayloads<npart_t>("s2_pld_i", 2, npart2, 0, npart2, pldi_2);

      writer.endSaving();
    }

    {
      // read checkpoint
      ndfield_t<Dim::_3D, 6> field1_read { "fld1_read", nx1_gh, nx2_gh, nx3_gh };
      ndfield_t<Dim::_3D, 6> field2_read { "fld2_read", nx1_gh, nx2_gh, nx3_gh };

      array_t<int*>    i1_read { "i_1", npart1 };
      array_t<real_t*> u1_read { "u_1", npart1 };
      array_t<int*>    i2_read { "i_2", npart2 };
      array_t<real_t*> u2_read { "u_2", npart2 };

      array_t<real_t**> pldr_2_read { "pldr_2", npart2, 2 };

      array_t<npart_t**> pldi_1_read { "pldi_1", npart1, 1 };
      array_t<npart_t**> pldi_2_read { "pldi_2", npart2, 2 };

      adios2::IO     io     = adios.DeclareIO("checkpointRead");
      adios2::Engine reader = io.Open(checkpoint_path / "step-00000000.bp",
                                      adios2::Mode::Read);
      reader.BeginStep();

      auto fieldRange = adios2::Box<adios2::Dims>({ 0, 0, 0, 0 },
                                                  { nx1_gh, nx2_gh, nx3_gh, 6 });
      ReadFields<Dim::_3D, 6>(io, reader, "em", fieldRange, field1_read);
      ReadFields<Dim::_3D, 6>(io, reader, "em0", fieldRange, field2_read);

      auto [nprtl1, noff1] = ReadParticleCount(io, reader, 0, 0, 1);
      auto [nprtl2, noff2] = ReadParticleCount(io, reader, 1, 0, 1);

      ReadParticleData<real_t>(io, reader, "ux1", 0, u1_read, nprtl1, noff1);
      ReadParticleData<real_t>(io, reader, "ux1", 1, u2_read, nprtl2, noff2);
      ReadParticleData<int>(io, reader, "i1", 0, i1_read, nprtl1, noff1);
      ReadParticleData<int>(io, reader, "i1", 1, i2_read, nprtl2, noff2);

      ReadParticlePayloads<real_t>(io, reader, "r", 1, pldr_2_read, 2, nprtl2, noff2);

      ReadParticlePayloads<npart_t>(io, reader, "i", 0, pldi_1_read, 1, nprtl1, noff1);
      ReadParticlePayloads<npart_t>(io, reader, "i", 1, pldi_2_read, 2, nprtl2, noff2);

      reader.EndStep();
      reader.Close();

      // check the validity
      Kokkos::parallel_for(
        "checkFields",
        CreateRangePolicy<Dim::_3D>({ 0, 0, 0 }, { nx1_gh, nx2_gh, nx3_gh }),
        Lambda(index_t i1, index_t i2, index_t i3) {
          for (int i = 0; i < 6; ++i) {
            if (not cmp::AlmostEqual(field1(i1, i2, i3, i),
                                     field1_read(i1, i2, i3, i))) {
              raise::KernelError(HERE, "Field1 read failed");
            }
            if (not cmp::AlmostEqual(field2(i1, i2, i3, i),
                                     field2_read(i1, i2, i3, i))) {
              raise::KernelError(HERE, "Field2 read failed");
            }
          }
        });

      raise::ErrorIf(npart1 != nprtl1, "Particle count 1 mismatch", HERE);
      raise::ErrorIf(npart2 != nprtl2, "Particle count 2 mismatch", HERE);
      raise::ErrorIf(noff1 != 0, "Particle offset 1 mismatch", HERE);
      raise::ErrorIf(noff2 != 0, "Particle offset 2 mismatch", HERE);

      Kokkos::parallel_for(
        "checkPrtl1",
        npart1,
        Lambda(index_t p) {
          if (not cmp::AlmostEqual(u1(p), u1_read(p))) {
            raise::KernelError(HERE, "u1 read failed");
          }
          if (i1(p) != i1_read(p)) {
            raise::KernelError(HERE, "i1 read failed");
          }
          if (pldi_1(p, 0) != pldi_1_read(p, 0)) {
            raise::KernelError(HERE, "pldi_1 read failed");
          }
        });
      Kokkos::parallel_for(
        "checkPrtl2",
        npart2,
        Lambda(index_t p) {
          if (not cmp::AlmostEqual(u2(p), u2_read(p))) {
            raise::KernelError(HERE, "u2 read failed");
          }
          if (i2(p) != i2_read(p)) {
            raise::KernelError(HERE, "i2 read failed");
          }
          if (not cmp::AlmostEqual(pldr_2(p, 0), pldr_2_read(p, 0))) {
            raise::KernelError(HERE, "pldr_2(0) read failed");
          }
          if (not cmp::AlmostEqual(pldr_2(p, 1), pldr_2_read(p, 1))) {
            raise::KernelError(HERE, "pldr_2(1) read failed");
          }
          if (pldi_2(p, 0) != pldi_2_read(p, 0)) {
            raise::KernelError(HERE, "pldi_2(0) read failed");
          }
          if (pldi_2(p, 1) != pldi_2_read(p, 1)) {
            raise::KernelError(HERE, "pldi_2(1) read failed");
          }
        });
    }

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    cleanup();
    Kokkos::finalize();
    return 1;
  }
  cleanup();
  Kokkos::finalize();
  return 0;
}
