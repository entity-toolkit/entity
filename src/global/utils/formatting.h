/**
 * @file utils/formatting.h
 * @brief String formatting utilities
 * @implements
 *   - fmt::format<> -> std::string
 *   - fmt::pad -> std::string
 *   - fmt::toLower -> std::string
 *   - fmt::splitString -> std::vector<std::string>
 *   - fmt::repeat -> std::string
 *   - fmt::formatVector -> std::string
 * @namespaces:
 *   - fmt::
 */

#ifndef GLOBAL_UTILS_FORMATTING_H
#define GLOBAL_UTILS_FORMATTING_H

#include "arch/traits.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <regex>
#include <sstream>
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
   * @brief pads a string with a given character
   * @param str String to pad
   * @param n Number of characters in total (incl. padding)
   * @param c Character to pad with
   * @param right Pad on the right
   */
  inline auto pad(const std::string& str, std::size_t n, char c, bool right = false)
    -> std::string {
    if (n <= str.size()) {
      return str;
    }
    if (right) {
      return str + std::string(n - str.size(), c);
    }
    return std::string(n - str.size(), c) + str;
  }

  /**
   * @brief formats a vector of arbitrary type: [a, b, c, ...]
   */
  template <typename T>
  auto formatVector(const std::vector<T>& vec) -> std::string {
    std::ostringstream oss;
    oss << "[";
    if (!vec.empty()) {
      if constexpr (traits::is_pair<T>::value) {
        if constexpr (
          traits::has_method<traits::to_string_t, typename T::first_type>::value) {
          oss << "{" << vec[0].first.to_string() << ", "
              << vec[0].second.to_string() << "}";
          for (size_t i = 1; i < vec.size(); ++i) {
            oss << ", {" << vec[i].first.to_string() << ", "
                << vec[i].second.to_string() << "}";
          }
        } else {
          oss << "{" << vec[0].first << ", " << vec[0].second << "}";
          for (size_t i = 1; i < vec.size(); ++i) {
            oss << ", {" << vec[i].first << ", " << vec[i].second << "}";
          }
        }
      } else {
        oss << vec[0];
        for (size_t i = 1; i < vec.size(); ++i) {
          oss << ", " << vec[i];
        }
      }
    }
    oss << "]";
    return oss.str();
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
  inline auto splitString(const std::string& str, const std::string& delim)
    -> std::vector<std::string> {
    std::regex regexz(delim);
    return { std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
             std::sregex_token_iterator() };
  }

  /**
   * @brief Repeat a string n number of times
   * @param s String to repeat
   * @param n Number of times to repeat
   */
  inline auto repeat(const std::string& s, std::size_t n) -> std::string {
    std::string result;
    result.reserve(n * s.size());
    for (std::size_t i = 0; i < n; ++i) {
      result += s;
    }
    return result;
  }

} // namespace fmt

#endif // GLOBAL_UTILS_FORMATTING_H
