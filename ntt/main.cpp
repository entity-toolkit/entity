#include "global.h"
#include "constants.h"
#include "arrays.h"

#include <iostream>
#include <cassert>

int main() {
  ntt::real_t a { 2.0 };
  std::cout << sizeof(a) << std::endl;
  std::cout << ntt::sim.title << std::endl;
  ntt::sim.setTitle("NOTHING");
  std::cout << ntt::sim.title << std::endl;
  std::cout << ntt::sim.precision << std::endl;
  return 0;
}
