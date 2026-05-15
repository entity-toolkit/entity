#include "framework/simulation.h"

#include "utils/error.h"
#include "utils/log.h"

auto main(int argc, char* argv[]) -> int {
  using namespace ntt;
  Simulation sim(argc, argv);

  PLOGV << "This is a VERBOSE message";
  PLOGD << "This is a DEBUG message";
  PLOGI << "This is an INFO message";
  PLOGW << "This is a WARNING message";
  PLOGE << "This is an ERROR message";
  PLOGF << "This is a FATAL message";

  logger::Checkpoint(HERE);
  raise::Warning("This is a warning message", HERE);
  raise::Error("This is an error message", HERE);

  return 0;
}