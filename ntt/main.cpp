#include "global.h"
#include "simulation.h"

#include <Kokkos_Core.hpp>

#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Formatters/FuncMessageFormatter.h>
#include <plog/Appenders/ColorConsoleAppender.h>

#include <iostream>

using plog_t = plog::ColorConsoleAppender<plog::FuncMessageFormatter>;

void initLogger(plog_t *console_appender);

// Logging is done via `plog` library...
// ... Use the following commands:
//  `PLOGI << ...` for general info
//  `PLOGF << ...` for fatal error messages (development)
//  `PLOGD << ...` for debug messages (development)
//  `PLOGE << ...` for simple error messages
//  `PLOGW << ...` for warnings

auto main(int argc, char *argv[]) -> int {
  plog_t console_appender;
  initLogger(&console_appender);

  Kokkos::initialize();
  try {
    ntt::Simulation<ntt::One_D> sim(argc, argv);
    sim.initialize();
    sim.verify();
    sim.printDetails();
    sim.mainloop();
    sim.finalize();
  } catch (std::exception &err) {
    std::cerr << err.what() << std::endl;
    Kokkos::finalize();

    return -1;
  }
  Kokkos::finalize();

  return 0;
}

void initLogger(plog_t *console_appender) {
  plog::Severity max_severity;
#ifdef VERBOSE
  max_severity = plog::verbose;
#elif DEBUG
  max_severity = plog::debug;
#else
  max_severity = plog::info;
#endif
  plog::init(max_severity, console_appender);
}
