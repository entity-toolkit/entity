#include "cargs.h"

#include <iostream>

auto main(int argc, char *argv[]) -> int {
  {
    // Example #1: reading command line arguments
    // ... `cl_args` is a global constat that contains all the arguments
    // ... we initialize it with `.readCommandLineArguments(argc, argv)`
    // ... then access arguments via `.getArgument(<KEY>, <DEFAULT>)`
    // ... `<DEFAULT>` value is used when the `<KEY>` is unspecified
    // ... if no default value provided and key is not found -- throws assertion
    // error

    // using namespace ntt::io;
    // cl_args.readCommandLineArguments(argc, argv);
    // std::cout << cl_args.getArgument("-input") << "\n";
    // std::cout << cl_args.getArgument("-output", "my_output") << "\n";
  }
  return 0;
}
