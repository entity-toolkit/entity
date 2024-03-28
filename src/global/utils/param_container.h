/**
 * @file utils/param_container.h
 * @brief Smart container for safely storing and retrieving parameters with stringize support
 * @implements
 *   - prm::Parameters
 * @namespaces:
 *   - prm::
 * @depends:
 *   - utils/log.h
 *   - utils/error.h
 *   - utils/formatting.h
 */

#ifndef GLOBAL_UTILS_PARAM_CONTAINER_H
#define GLOBAL_UTILS_PARAM_CONTAINER_H

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include <any>
#include <ios>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <type_traits>

namespace prm {

  class Parameters {
    using key_t = const std::string&;

  protected:
    std::map<std::string, std::any> vars;
    std::vector<std::string>        promises;

  public:
    auto contains(key_t key) const -> bool {
      return vars.find(key) != vars.end();
    }

    auto promiseToDefine(key_t key) -> void {
      promises.push_back(key);
    }

    auto promisesFulfilled() const -> bool {
      return std::all_of(promises.begin(), promises.end(), [this](key_t key) {
        return contains(key);
      });
    }

    void set(key_t key, const std::any& value) {
      vars[key] = value;
    }

    template <typename T>
    [[nodiscard]]
    auto get(key_t key, const std::optional<T>& def = std::nullopt) const -> T {
      try {
        return std::any_cast<T>(vars.at(key));
      } catch (const std::out_of_range&) {
        if (def.has_value()) {
          raise::Warning(
            fmt::format("Key %s not found, falling back to default", key),
            HERE);
          return def.value();
        }
        raise::Error(fmt::format("Key %s not found", key), HERE);
      } catch (const std::bad_any_cast&) {
        raise::Error(fmt::format("Bad any_cast for %s", key), HERE);
      }
    }

    template <typename T>
    [[nodiscard]]
    auto stringize(key_t key) const -> std::string {
      std::ostringstream result;
      auto remove_last_n = [](std::ostringstream& ss, std::size_t n) {
        auto temp = ss.str();
        temp      = temp.substr(0, temp.size() - n);
        ss.str("");
        ss << temp;
      };
      auto stringize_key = [](std::ostringstream& ss, std::any v) {
        if constexpr (std::is_object_v<T>) {
          statc_assert(std::is_member_function_pointer_v<decltype(&T::stringize)>);
          ss << T::stringize(std::any_cast<std::string_view>(v));
        } else {
          ss << std::any_cast<T>(v);
        }
      };
      result << std::boolalpha;
      try {
        using vecT    = std::vector<T>;
        using vecvecT = std::vector<vecT>;
        if (vars.at(key).type() == typeid(vecvecT)) {
          result << "[";
          for (const auto& v : std::any_cast<vecvecT>(vars.at(key))) {
            result << "{";
            for (const auto& i : v) {
              stringize_key(result, i);
              result << ", ";
            }
            remove_last_n(result, 2);
            result << "}, ";
          }
          remove_last_n(result, 2);
          result << "]";
        } else if (vars.at(key).type() == typeid(vecT)) {
          result << "[";
          for (const auto& i : std::any_cast<vecT>(vars.at(key))) {
            stringize_key(result, i);
            result << ", ";
          }
          remove_last_n(result, 2);
          result << "]";
        } else if (vars.at(key).type() == typeid(T)) {
          stringize_key(result, vars.at(key));
        }
      } catch (const std::out_of_range&) {
        raise::Error(fmt::format("Key %s not found", key), HERE);
      } catch (const std::bad_any_cast&) {
        raise::Error(fmt::format("Bad any_cast for %s", key), HERE);
      } catch (const std::exception& e) {
        raise::Error(fmt::format("Unknown error for %s : %s", key, e.what()), HERE);
      }
      return result.str();
    }
  };

} // namespace prm

#endif // GLOBAL_UTILS_PARAM_CONTAINER_H