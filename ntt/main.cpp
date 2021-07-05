#include "global.h"
#include "sim.h"

#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ColorConsoleAppender.h>

#include <string>
#include <iostream>

void initLogger(plog::ColorConsoleAppender<plog::TxtFormatter> *console_appender);

int main(int argc, char *argv[]) {
  plog::ColorConsoleAppender<plog::TxtFormatter> console_appender;
  initLogger(&console_appender);

  ntt::PICSimulation1D sim(ntt::CARTESIAN);
  sim.parseInput(argc, argv);
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
