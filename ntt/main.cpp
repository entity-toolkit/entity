#include "global.h"
#include "sim.h"

#include <string>
#include <iostream>

int main(int argc, char *argv[]) {
  // 1. parse CL args
  // 2. init simulation object
  // 3. init inputparams object (read input)
  ntt::initializeAll(argc, argv);
  return 0;
}
