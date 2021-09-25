#include "global.h"
#include "pgen.h"
#include "meshblock.h"

#include <Kokkos_Core.hpp>

#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ColorConsoleAppender.h>

#include <iostream>

void initLogger(plog::ColorConsoleAppender<plog::TxtFormatter> *console_appender);

// Logging is done via `plog` library...
// ... Use the following commands:
//  `PLOGI << ...` for non-fatal warning messages (development)
//  `PLOGF << ...` for fatal error messages (development)
//  `PLOGD << ...` for debug messages (development)
//  `PLOGE << ...` for simple error messages
//  `PLOGW << ...` for warnings

auto main(int argc, char *argv[]) -> int {
  plog::ColorConsoleAppender<plog::TxtFormatter> console_appender;
  initLogger(&console_appender);

  Kokkos::initialize();
  try {
    ntt::Meshblock<ntt::One_D> mblock(100);
    Kokkos::parallel_for("init", 100,
      KL (ntt::index_t i) {
        mblock.ex1(i) = 2.2;
        mblock.ex2(i) = 2.3;
      }
    );
    Kokkos::parallel_for("run", 100,
      KL (ntt::index_t i) {
        mblock.ex3(i) = mblock.ex2(i) - mblock.ex1(i);
      }
    );
    double sum {0.0};
    Kokkos::parallel_reduce("run", 100,
      KL (ntt::index_t i, double & sum_) {
        sum_ += mblock.ex3(i);
      }, sum
    );
    std::cout << sum << "\n";
    // ProblemGenerator ntt_pgen(argc, argv);
    // ntt_pgen.start(argc, argv);
    // ntt_pgen.start(argc, argv);
  } catch (std::exception &err) {
    std::cerr << err.what() << std::endl;
    return -1;
  }
  Kokkos::finalize();

  return 0;
}

void initLogger(plog::ColorConsoleAppender<plog::TxtFormatter> *console_appender) {
  plog::Severity max_severity;
#ifdef VERBOSE
  max_severity = plog::verbose;
#elif DEBUG
  max_severity = plog::debug;
#else
  max_severity = plog::warning;
#endif
  plog::init(max_severity, console_appender);
}

// ProblemGenerator::ProblemGenerator(int argc, char *argv[]); {
//   simulation = new UserSimulation(argc, argv);
// }
