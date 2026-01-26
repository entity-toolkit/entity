#include "utils/reporter.h"

#include "utils/colors.h"
#include "utils/formatting.h"

#include <string>
#include <utility>
#include <vector>

namespace reporter {
  void AddHeader(std::string&                    report,
                 const std::vector<std::string>& lines,
                 const std::vector<const char*>& colors) {
    report += fmt::format("%s╔%s╗%s\n",
                          color::BRIGHT_BLACK,
                          fmt::repeat("═", 58).c_str(),
                          color::RESET);
    for (auto i { 0u }; i < lines.size(); ++i) {
      report += fmt::format("%s║%s %s%s%s%s%s║%s\n",
                            color::BRIGHT_BLACK,
                            color::RESET,
                            colors[i],
                            lines[i].c_str(),
                            color::RESET,
                            fmt::repeat(" ", 57 - lines[i].size()).c_str(),
                            color::BRIGHT_BLACK,
                            color::RESET);
    }
    report += fmt::format("%s╚%s╝%s\n",
                          color::BRIGHT_BLACK,
                          fmt::repeat("═", 58).c_str(),
                          color::RESET);
  }

  void AddCategory(std::string& report, unsigned short indent, const char* name) {
    report += fmt::format("%s%s%s%s\n",
                          std::string(indent, ' ').c_str(),
                          color::BLUE,
                          name,
                          color::RESET);
  }

  void AddSubcategory(std::string& report, unsigned short indent, const char* name) {
    report += fmt::format("%s%s-%s %s:\n",
                          std::string(indent, ' ').c_str(),
                          color::BRIGHT_BLACK,
                          color::RESET,
                          name);
  }

  void AddLabel(std::string& report, unsigned short indent, const char* label) {
    report += fmt::format("%s%s\n", std::string(indent, ' ').c_str(), label);
  }

  auto Bytes2HumanReadable(std::size_t bytes) -> std::pair<double, std::string> {
    const std::vector<std::string> units { "B", "KB", "MB", "GB", "TB" };
    idx_t                          unit_idx = 0;
    auto                           size     = static_cast<double>(bytes);
    while ((size >= 1024.0) and (unit_idx < units.size() - 1)) {
      size /= 1024.0;
      ++unit_idx;
    }
    return { size, units[unit_idx] };
  }
} // namespace reporter
