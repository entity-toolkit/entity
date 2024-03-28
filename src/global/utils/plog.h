/**
 * @file utils/plog.h
 * @brief Plog formatters for the Entity
 * @implements
 *   - plog::Nt2ConsoleFormatter
 *   - plog::Nt2InfoFormatter
 * @namespaces:
 *   - plog::
 */

#ifndef GLOBAL_UTILS_PLOG_H
#define GLOBAL_UTILS_PLOG_H

#include <plog/Log.h>

namespace plog {

  class Nt2ConsoleFormatter {
  public:
    static auto header() -> util::nstring {
      return util::nstring();
    }

    static auto format(const Record& record) -> util::nstring {
      util::nostringstream ss;
      if (record.getSeverity() == plog::debug &&
          plog::get()->getMaxSeverity() == plog::verbose) {
        ss << PLOG_NSTR("\n") << record.getFunc() << PLOG_NSTR(" @ ")
           << record.getLine() << PLOG_NSTR("\n");
      }
      ss << std::setw(9) << std::left << severityToString(record.getSeverity())
         << PLOG_NSTR(": ");
      ss << record.getMessage() << PLOG_NSTR("\n");
      return ss.str();
    }
  };

  class Nt2InfoFormatter {
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

#endif // GLOBAL_UTILS_PLOG_H