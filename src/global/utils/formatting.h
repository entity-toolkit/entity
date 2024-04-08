/**
 * @file utils/formatting.h
 * @brief String formatting utilities
 * @implements
 *   - fmt::format<> -> std::string
 *   - fmt::toLower -> std::string
 *   - fmt::splitString -> std::vector<std::string>
 * @namespaces:
 *   - fmt::
 */

#ifndef GLOBAL_UTILS_FORMATTING_H
#define GLOBAL_UTILS_FORMATTING_H

#include <algorithm>
#include <cctype>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace fmt {
  /**
   * @brief Format a string with arguments
   * @note Implements minimal C-style formatting
   */
  template <typename... Args>
  inline auto format(const char* format, Args... args) -> std::string {
    auto size_s = std::snprintf(nullptr, 0, format, args...) + 1;
    if (size_s <= 0) {
      throw std::runtime_error("Error during formatting.");
    }
    auto                    size { static_cast<std::size_t>(size_s) };
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format, args...);
    return std::string(buf.get(), buf.get() + size - 1);
  }

  /**
   * @brief Convert a string to lowercase
   */
  inline auto toLower(const std::string& str) -> std::string {
    std::string result { str };
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
  }

  /**
   * @brief Split a string into a vector of strings
   * @param str String to split
   * @param delim Delimiter
   * @return Vector of strings
   */
  inline auto splitString(const std::string& str,
                          const std::string& delim) -> std::vector<std::string> {
    std::regex regexz(delim);
    return { std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
             std::sregex_token_iterator() };
  }

} // namespace fmt

#endif // GLOBAL_UTILS_FORMATTING_H