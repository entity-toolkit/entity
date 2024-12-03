#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/containers/particles.h"
#include "metrics/metric_base.h"
#include "metrics/minkowski.h"
#include "arch/mpi_tags.h"

#include <Kokkos_Random.hpp>
#define TIMER_START(label) \
    Kokkos::fence(); \
    auto start_##label = std::chrono::high_resolution_clock::now();

#define TIMER_STOP(label) \
    Kokkos::fence(); \
    auto stop_##label = std::chrono::high_resolution_clock::now(); \
    auto duration_##label = std::chrono::duration_cast<std::chrono::microseconds>(stop_##label - start_##label).count(); \
    std::cout << "Timer [" #label "]: " << duration_##label << " microseconds" << std::endl;

/*
  Test to check the performance of the new particle allocation scheme
    - Create a metadomain object
    - Create particle array
    - Initialize the position and velocities of the particles
    - Set a large timestep (see where that is set)
    - Make a loop of N iterations, where the positions of particles is sorted
      and pushed
    - Check if the particle tags are correct after each iteration
    - Compute the time taken for best of N iterations for the communication
 */


/*
    Structure of the 2D domain
    ---------------------------------- (3,3)
    |          |          |          |
    |          |          |          |
    |          |          |          |
    |          |          |          |
    ---------------------------------- (3,2)
    |          |          |          |
    |          |          |          |
    |          |          |          |
    |          |          |          |
    ---------------------------------- (3,1)
    |          |          |          |
    |          |          |          |
    |          |          |          |
    |          |          |          |
    ----------------------------------
  (0,0)       (1,0)      (2,0)         (3,0)
*/

/*
  Function to check the tags of a domain object to make sure that
  all the tags are alive. If the tags are not alive then the function
  prints the tag count for each of the particles along with the rank
  of the domain.
*/
template <SimEngine::type S, class M>
void CheckDomainTags(Domain<S, M>&  domain)
{
  bool all_alive                = true;
  bool no_dead_particles        = true;
  for (auto& species : domain.species) {
    std::cout << "Checking domain tags for species: " << species.label() << std::endl;
    const auto npart_per_tag_arr  = species.npart_per_tag();
    const auto npart              = species.npart();
    if (npart != npart_per_tag_arr[ParticleTag::alive]){
      all_alive = false;
    }
    for (std::size_t i = 0; i < npart_per_tag_arr.size(); ++i) {
      if (i == ParticleTag::alive) {
        continue;
      }
      if (npart_per_tag_arr[i] != 0) {
        no_dead_particles = false;
      }
    }

    raise::ErrorIf(all_alive == false,
                   "Array contains particles with tags other than alive",
                   HERE);
    raise::ErrorIf(no_dead_particles == false,
                    "Array contains dead particles",
                    HERE);
    //raise::ErrorIf(tag_check_h(0) == false,
    //                "Tag check failed",
    //                HERE);
  }
  return;
}

void InitializePositionsDomain(Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>& domain)
{
  for (auto& species : domain.species) {
    TIMER_START(Sorting_timer);
    species.SortByTags();
    TIMER_STOP(Sorting_timer);
    species.SyncHostDevice();
    std::cout << "Number of particles in domain: " << species.npart() << std::endl;
    //std::cout << "Extent of i1" << species.i1.extent(0) << std::endl;
  }
  CheckDomainTags(domain);
}



auto main(int argc, char* argv[]) -> int {
  std::cout << "Constructing the domain" << std::endl;
  ntt::GlobalInitialize(argc, argv);
  // Create a Metadomain object
  const unsigned int ndomains = 9;
  const std::vector<int> global_decomposition = {{}};
  const std::vector<std::size_t> global_ncells = {32, 32};
  const boundaries_t<real_t> global_extent = {{0.0, 0.0}, {3.0, 3.0}};
  const boundaries_t<FldsBC> global_flds_bc = {{FldsBC::PERIODIC, FldsBC::PERIODIC}, {FldsBC::PERIODIC, FldsBC::PERIODIC}};
  const boundaries_t<PrtlBC> global_prtl_bc = {{PrtlBC::PERIODIC, PrtlBC::PERIODIC}, {PrtlBC::PERIODIC, PrtlBC::PERIODIC}};
  const std::map<std::string, real_t> metric_params = {};
  const int maxnpart = 1000;
  auto species = ntt::Particles<Dim::_2D, ntt::Coord::Cart>(1u,
                                                            "test_e",
                                                            1.0f,
                                                            1.0f,
                                                            maxnpart,
                                                            ntt::PrtlPusher::BORIS,
                                                            false,
                                                            ntt::Cooling::NONE);

    species.set_npart(maxnpart);
    auto &this_i1  = species.i1;
    auto &this_i2  = species.i2;
    auto &this_i3  = species.i3;
    auto &this_dx1 = species.dx1;
    auto &this_dx2 = species.dx2;
    auto &this_dx3 = species.dx3;
    auto &this_ux1 = species.ux1;
    auto &this_ux2 = species.ux2;
    auto &this_ux3 = species.ux3;
    auto &this_tag = species.tag;

    std::cout << "Species particle count is " << species.npart() << std::endl;
    Kokkos::parallel_for("SetPositions", 
          species.npart(), Lambda(const std::size_t i) {
      this_i1(i)       = 1;
      this_i2(i)       = 1;
      this_i3(i)       = 1;
      this_dx1(i)      = 0.01;
      this_dx2(i)      = 0.01;
      this_ux1(i)      = 0.;
      this_ux2(i)      = 0.;
      this_ux3(i)      = 0.;
      this_tag(i)      = 1;
    });
    Kokkos::fence();
  std::cout << "Species set " << species.npart() << std::endl;
  auto metadomain = Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>
                    ( ndomains,
                      global_decomposition,
                      global_ncells,
                      global_extent,
                      global_flds_bc,
                      global_prtl_bc,
                      metric_params,
                      {species}
                    ); 

   //metadomain.runOnLocalDomains([&](auto& loc_dom) {
   // InitializePositionsDomain(loc_dom);
   //});

  // Get the pointer to the subdomain
  //const auto local_subdomain_idx = metadomain.l_subdomain_indices()[0];
  //auto local_domain = metadomain.subdomain_ptr(local_subdomain_idx);

  // Set the positions of the particles in each domain
  //for (auto& species : local_domain->species)
  //{
  //  auto tag       = ParticleTag::alive;
  //  auto &this_i1  = species.i1;
  //  auto &this_i2  = species.i2;
  //  auto &this_i3  = species.i3;
  //  auto &this_dx1 = species.dx1;
  //  auto &this_dx2 = species.dx2;
  //  auto &this_dx3 = species.dx3;
  //  auto &this_ux1 = species.ux1;
  //  auto &this_ux2 = species.ux2;
  //  auto &this_ux3 = species.ux3;
  //  auto &this_tag = species.tag;
  //  Kokkos::parallel_for("SetPositions", 
  //        species.npart(), Lambda(const std::size_t i) {
  //    this_i1(i)       = 1;
  //    this_i2(i)       = 1;
  //    this_i3(i)       = 0;
  //    this_dx1(i)      = 0.01;
  //    this_dx2(i)      = 0.01;
  //    this_ux1(i)      = 0.5;
  //    this_ux2(i)      = 0.5;
  //    this_tag(i)   = tag;
  //  });
//
    //species.SortByTags();
    //species.SyncHostDevice();
  //}

  // Get and print the extent of each domain
  //std::cout << fmt::format("x1 extent {%.2f; %.2f} \n", 
  //                        local_domain->mesh.extent(in::x1).first, 
  //                        local_domain->mesh.extent(in::x1).second);
  //std::cout << fmt::format("x2 extent {%.2f; %.2f} \n", 
  //                        local_domain->mesh.extent(in::x2).first, 
  //                        local_domain->mesh.extent(in::x2).second);
  // Print the number of particles per domain
  //std::cout << "Number of particles in domain " << local_subdomain_idx << ": " << local_domain->species[0].npart() << std::endl;
  // Print the position of the 5 particles in the domain

  ntt::GlobalFinalize();

  std::cout << "Terminating" << std::endl;

  return 0;
}
