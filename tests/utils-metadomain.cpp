#include "wrapper.h"

#include "communications/decomposition.h"
#include "communications/metadomain.h"
#include "utilities/qmath.h"

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    const auto resolution = std::vector<unsigned int>({ 5000, 1800 });
#ifdef MINKOWSKI_METRIC
    const auto extent     = std::vector<real_t>({ 1.0, 100.0, -20.0, 15.64 });
    const auto boundaries = std::vector<std::vector<ntt::BoundaryCondition>> {
      { ntt::BoundaryCondition::PERIODIC },
      { ntt::BoundaryCondition::OPEN }
    };
#else
    const auto extent = std::vector<real_t>({ 1.0, 100.0, ZERO, ntt::constant::PI });
    const auto boundaries = std::vector<std::vector<ntt::BoundaryCondition>> {
      { ntt::BoundaryCondition::CUSTOM, ntt::BoundaryCondition::ABSORB },
      { ntt::BoundaryCondition::AXIS }
    };
#endif
    // optional for GR
    const auto spin    = (real_t)(0.9);
    const auto rh      = ONE + std::sqrt(ONE - SQR(spin));
    // optional for Qspherical
    const auto qsph_r0 = (real_t)(0.0);
    const auto qsph_h  = (real_t)(0.25);

    auto params = new real_t[6];
    params[0]   = qsph_r0;
    params[1]   = qsph_h;
    params[4]   = spin;
    params[5]   = rh;

    const auto decomposition = std::vector<unsigned int> { 7, 3 };

    auto metadomain = ntt::Metadomain<ntt::Dim2>(resolution,
                                                 extent,
                                                 decomposition,
                                                 params,
                                                 boundaries,
                                                 true);

    auto first_domain = *metadomain.domainByOffset({ 0, 0 });
    auto last_domain  = *metadomain.domainByOffset(
      { decomposition[0] - 1, decomposition[1] - 1 });
    for (auto d { 0 }; d < 2; ++d) {
      if (first_domain.offsetNdomains()[d] != 0) {
        throw std::logic_error("first_domain.offsetNdomains()[d] != 0");
      }
      if (first_domain.offsetNcells()[d] != 0) {
        throw std::logic_error("first_domain.offsetNcells()[d] != 0");
      }
      if (last_domain.offsetNdomains()[d] != decomposition[d] - 1) {
        throw std::logic_error(
          "last_domain.offsetNdomains()[d] != decomposition[d] - 1");
      }
      if (last_domain.offsetNcells()[d] + last_domain.ncells()[d] != resolution[d]) {
        throw std::logic_error("last_domain.offsetNcells()[d] + "
                               "last_domain.ncells()[d] != resolution[d]");
      }
    }

    if (!ntt::AlmostEqual(first_domain.extent()[0], extent[0])) {
      throw std::logic_error("first_domain.extent()[0] != extent[0]");
    }
    if (!ntt::AlmostEqual(first_domain.extent()[2], extent[2])) {
      throw std::logic_error("first_domain.extent()[2] != extent[2]");
    }
    if (!ntt::AlmostEqual(last_domain.extent()[1], extent[1])) {
      throw std::logic_error("last_domain.extent()[1] != extent[1]");
    }
    if (!ntt::AlmostEqual(last_domain.extent()[3], extent[3])) {
      throw std::logic_error("last_domain.extent()[3] != extent[3]");
    }
    if (!(first_domain.boundaries()[0][0] == boundaries[0][0])) {
      throw std::logic_error("wrong first_domain.boundaries()[0][0]");
    }
    if (!(first_domain.boundaries()[0][1] == ntt::BoundaryCondition::COMM)) {
      throw std::logic_error("wrong first_domain.boundaries()[0][1]");
    }
    if (!(first_domain.boundaries()[1][0] ==
          (boundaries[1].size() > 1 ? boundaries[1][1] : boundaries[1][0]))) {
      throw std::logic_error("wrong first_domain.boundaries()[1][0]");
    }
    if (!(first_domain.boundaries()[1][1] == ntt::BoundaryCondition::COMM)) {
      throw std::logic_error("wrong first_domain.boundaries()[1][1]");
    }
    if (!(last_domain.boundaries()[0][0] == ntt::BoundaryCondition::COMM)) {
      throw std::logic_error("wrong last_domain.boundaries()[0][0]");
    }
    if (!(last_domain.boundaries()[0][1] ==
          (boundaries[0].size() > 1 ? boundaries[0][1] : boundaries[0][0]))) {
      throw std::logic_error("wrong last_domain.boundaries()[0][1]");
    }
    if (!(last_domain.boundaries()[1][0] == ntt::BoundaryCondition::COMM)) {
      throw std::logic_error("wrong last_domain.boundaries()[1][0]");
    }
    if (!(last_domain.boundaries()[1][1] ==
          (boundaries[1].size() > 1 ? boundaries[1][1] : boundaries[1][0]))) {
      throw std::logic_error("wrong last_domain.boundaries()[1][1]");
    }

    auto first_domain1 = first_domain.neighbors({ 0, +1 })
                           ->neighbors({ 0, +1 })
                           ->neighbors({ 0, -1 })
                           ->neighbors({ 0, -1 });

    auto first_domain2 = last_domain.neighbors({ -1, -1 })
                           ->neighbors({ -1, -1 })
                           ->neighbors({ -1, 0 })
                           ->neighbors({ -1, 0 })
                           ->neighbors({ -1, 0 })
                           ->neighbors({ -1, 0 });

    if (first_domain1 != metadomain.domainByOffset({ 0, 0 })) {
      throw std::logic_error("Wrong neighbor assignment");
    }
    if (first_domain2 != first_domain1) {
      throw std::logic_error("Wrong neighbor assignment");
    }

    if (first_domain.neighbors({ 0, -1 }) != nullptr) {
      throw std::logic_error("Wrong neighbor assignment: boundaries");
    }
    if (last_domain.neighbors({ 0, 1 }) != nullptr) {
      throw std::logic_error("Wrong neighbor assignment: boundaries");
    }

    for (auto& domain : metadomain.domains) {
      for (auto& direction : ntt::Directions<ntt::Dim2>::all) {
        if ((domain.neighbors(direction) == nullptr) &&
            (domain.boundaryIn(direction) == ntt::BoundaryCondition::COMM)) {
          throw std::logic_error("Neighbor == null && BC == COMM.");
        }
        if ((domain.neighbors(direction) != nullptr) &&
            (domain.boundaryIn(direction) != ntt::BoundaryCondition::COMM)) {
          throw std::logic_error("Neighbor != null && BC != COMM.");
        }
      }
    }

  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();
    return -1;
  }
  ntt::GlobalFinalize();

  return 0;
}