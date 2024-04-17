#include "enums.h"

#include "utils/error.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "engines/grpic/grpic.h"
#include "engines/srpic/srpic.h"
#include "framework/simulation.h"

#include <iostream>

auto main(int argc, char* argv[]) -> int {
  ntt::Simulation sim { argc, argv };
  if (sim.requested_engine() == ntt::SimEngine::SRPIC) {
    /* SRPIC -------------------------------------------------------------- */
    if (sim.requested_metric() == ntt::Metric::Minkowski) {
      /* minkowski ------------------------------------------------------ */
      if (sim.requested_dimension() == Dim::_1D) {
        sim.run<ntt::SRPICEngine<metric::Minkowski<Dim::_1D>>>();
      } else if (sim.requested_dimension() == Dim::_2D) {
        sim.run<ntt::SRPICEngine<metric::Minkowski<Dim::_2D>>>();
      } else if (sim.requested_dimension() == Dim::_3D) {
        sim.run<ntt::SRPICEngine<metric::Minkowski<Dim::_3D>>>();
      } else {
        raise::Fatal("Invalid dimension", HERE);
      }
    } else if (sim.requested_metric() == ntt::Metric::Spherical) {
      /* spherical ------------------------------------------------------ */
      if (sim.requested_dimension() == Dim::_2D) {
        sim.run<ntt::SRPICEngine<metric::Spherical<Dim::_2D>>>();
      } else {
        raise::Fatal("Invalid dimension", HERE);
      }
    } else if (sim.requested_metric() == ntt::Metric::QSpherical) {
      /* qspherical ----------------------------------------------------- */
      if (sim.requested_dimension() == Dim::_2D) {
        sim.run<ntt::SRPICEngine<metric::QSpherical<Dim::_2D>>>();
      } else {
        raise::Fatal("Invalid dimension", HERE);
      }
    } else {
      raise::Fatal("Invalid metric", HERE);
    }
  } else if (sim.requested_engine() == ntt::SimEngine::GRPIC) {
    /* GRPIC -------------------------------------------------------------- */
    if (sim.requested_metric() == ntt::Metric::Kerr_Schild) {
      /* kerr_schild ---------------------------------------------------- */
      if (sim.requested_dimension() == Dim::_2D) {
        sim.run<ntt::GRPICEngine<metric::KerrSchild<Dim::_2D>>>();
      } else {
        raise::Fatal("Invalid dimension", HERE);
      }
    } else if (sim.requested_metric() == ntt::Metric::QKerr_Schild) {
      /* qkerr_schild --------------------------------------------------- */
      if (sim.requested_dimension() == Dim::_2D) {
        sim.run<ntt::GRPICEngine<metric::QKerrSchild<Dim::_2D>>>();
      } else {
        raise::Fatal("Invalid dimension", HERE);
      }
    } else if (sim.requested_metric() == ntt::Metric::Kerr_Schild_0) {
      /* kerr_schild_0 -------------------------------------------------- */
      if (sim.requested_dimension() == Dim::_2D) {
        sim.run<ntt::GRPICEngine<metric::KerrSchild0<Dim::_2D>>>();
      } else {
        raise::Fatal("Invalid dimension", HERE);
      }
    }
  }
  return 0;
}