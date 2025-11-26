/**
 * @file utils/plog.h
 * @brief Plog formatters for the Entity
 * @implements
 *   - plog::NttConsoleFormatter
 *   - plog::NttInfoFormatter
 *   - logger::initPlog -> void
 * @namespaces:
 *   - plog::
 *   - logger::
 */

#ifndef GLOBAL_UTILS_PLOG_H
#define GLOBAL_UTILS_PLOG_H

#include "utils/colors.h"
#include "utils/formatting.h"

#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Appenders/RollingFileAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>

#include <filesystem>
#include <iomanip>
#include <string>

namespace plog {

  class NttConsoleFormatter {
  public:
    static auto header() -> util::nstring {
      return util::nstring();
    }

    static auto format(const Record& record) -> util::nstring {
      util::nostringstream ss;
      if (record.getSeverity() != plog::none) {
        ss << std::setw(6) << std::left
           << severityToString(record.getSeverity()) << PLOG_NSTR(": ");
      }
      ss << record.getMessage() << PLOG_NSTR("\n");
      return ss.str();
    }
  };

  class NttStdoutFormatter {
  public:
    static auto header() -> util::nstring {
      return util::nstring();
    }

    static auto format(const Record& record) -> util::nstring {
      util::nostringstream ss;
      if (record.getSeverity() != plog::none) {
        ss << std::setw(6) << std::left
           << severityToString(record.getSeverity()) << PLOG_NSTR(": ");
      }
      ss << color::strip(record.getMessage()) << PLOG_NSTR("\n");
      return ss.str();
    }
  };

  class NttInfoFormatter {
  public:
    static auto header() -> util::nstring {
      return util::nstring();
    }

    static auto format(const Record& record) -> util::nstring {
      util::nostringstream ss;
      ss << record.getMessage() << PLOG_NSTR("\n");
      return ss.str();
    }
  };
} // namespace plog

namespace logger {

  template <int log_tag, int info_tag, int err_tag>
  inline void initPlog(const std::string& fpath,
                       const std::string& fname,
                       const std::string& log_level) {
    // setup logging
    const auto outfile_name  = fname + ".out";
    const auto logfile_name  = fname + ".log";
    const auto infofile_name = fname + ".info";
    const auto errfile_name  = fname + ".err";

    namespace fs = std::filesystem;
    if (not fpath.empty() and not fs::exists(fpath)) {
      fs::create_directory(fpath);
    }
    fs::path outfile_path { fs::path { fpath } / outfile_name };
    fs::path logfile_path { fs::path { fpath } / logfile_name };
    fs::path infofile_path { fs::path { fpath } / infofile_name };
    fs::path errfile_path { fs::path { fpath } / errfile_name };
    fs::remove(outfile_path);
    fs::remove(logfile_path);
    fs::remove(infofile_path);
    fs::remove(errfile_path);

    static plog::RollingFileAppender<plog::NttStdoutFormatter> outfileAppender(
      outfile_path.c_str());
    static plog::RollingFileAppender<plog::TxtFormatter> logfileAppender(
      logfile_path.c_str());
    static plog::RollingFileAppender<plog::NttInfoFormatter> infofileAppender(
      infofile_path.c_str());
    static plog::RollingFileAppender<plog::NttInfoFormatter> errfileAppender(
      errfile_path.c_str());
    auto log_severity = plog::verbose;
    if (fmt::toLower(log_level) == "warning") {
      log_severity = plog::warning;
    } else if (fmt::toLower(log_level) == "error") {
      log_severity = plog::error;
    }
    plog::init<log_tag>(log_severity, &logfileAppender);
    plog::init<info_tag>(plog::verbose, &infofileAppender);
    plog::init<err_tag>(plog::verbose, &errfileAppender);

#if defined(DEBUG)
    const auto severity = plog::verbose;
#else
    const auto severity = plog::info;
#endif

    static plog::ColorConsoleAppender<plog::NttConsoleFormatter> consoleAppender;
    plog::init(severity, &consoleAppender).addAppender(&outfileAppender);
  }

} // namespace logger

#endif // GLOBAL_UTILS_PLOG_H
