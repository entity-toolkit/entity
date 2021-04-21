#include "global.h"
#include "constants.h"
#include "arrays.h"

// #include <Kokkos_Core.hpp>

#include <iostream>
#include <cassert>

int main() {
  ntt::real_t a { 2.0 };
  std::cout << sizeof(a) << std::endl;
  std::cout << ntt::sim.title << std::endl;
  ntt::sim.setTitle("daradur");
  std::cout << ntt::sim.title << std::endl;
  std::cout << ntt::sim.precision << std::endl;
  return 0;
}
// int main(int argc, char* argv[]) {
  // Kokkos::initialize(argc, argv);
  // {
    // int N = (argc > 1) ? std::stoi(argv[1]) : 10000;
    // int M = (argc > 2) ? std::stoi(argv[2]) : 10000;
    // int R = (argc > 3) ? std::stoi(argv[3]) : 10;

    // printf("Called with: %i %i %i\n", N, M, R);
  // }
  // Kokkos::finalize();
// }

