#include "enums.h"
#include "global.h"

#include "framework/containers/particles.h"

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  // auto species = ntt::ParticleSpecies<Dim::_3D, ntt::Coord::Cart>(1u,
  //                                                           "test_e",
  //                                                           1.0f,
  //                                                           1.0f,
  //                                                           10000000,
  //                                                           ntt::PrtlPusher::BORIS,
  //                                                           false,
  //                                                           ntt::Cooling::NONE);
  ntt::GlobalFinalize();
  // * @param global_ndomains total number of domains
  // * @param global_decomposition decomposition of the global domain
  // * @param global_ncells number of cells in each dimension
  // * @param global_extent physical extent of the global domain
  // * @param global_flds_bc boundary conditions for fields
  // * @param global_prtl_bc boundary conditions for particles
  // * @param metric_params parameters for the metric
  // * @param species_params parameters for the particle species
  // Metadomain(unsigned int,
  //            const std::vector<int>&,
  //            const std::vector<std::size_t>&,
  //            const boundaries_t<real_t>&,
  //            const boundaries_t<FldsBC>&,
  //            const boundaries_t<PrtlBC>&,
  //            const std::map<std::string, real_t>&,
  //            const std::vector<ParticleSpecies>&);

  return 0;
}
