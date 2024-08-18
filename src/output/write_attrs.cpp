#include "enums.h"
#include "global.h"

#include "output/writer.h"

#include <any>
#include <functional>
#include <map>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace out {

  template <typename T, typename = void>
  struct has_to_string : std::false_type {};

  template <typename T>
  struct has_to_string<T, std::void_t<decltype(std::declval<T>().to_string())>>
    : std::true_type {};

  template <typename T>
  auto write(adios2::IO& io, const std::string& name, T var) ->
    typename std::enable_if<has_to_string<T>::value, void>::type {
    io.DefineAttribute(name, std::string(var.to_string()));
  }

  template <typename T>
  auto write(adios2::IO& io, const std::string& name, T var)
    -> decltype(void(T()), void()) {
    io.DefineAttribute(name, var);
  }

  template <>
  void write(adios2::IO& io, const std::string& name, bool var) {
    io.DefineAttribute(name, var ? 1 : 0);
  }

  template <>
  void write(adios2::IO& io, const std::string& name, Dimension var) {
    io.DefineAttribute(name, (unsigned short)var);
  }

  template <typename T>
  auto write_vec(adios2::IO& io, const std::string& name, std::vector<T> var) ->
    typename std::enable_if<has_to_string<T>::value, void>::type {
    std::vector<std::string> var_str;
    for (const auto& v : var) {
      var_str.push_back(v.to_string());
    }
    io.DefineAttribute(name, var_str.data(), var_str.size());
  }

  template <typename T>
  auto write_vec(adios2::IO& io, const std::string& name, std::vector<T> var)
    -> decltype(void(T()), void()) {
    io.DefineAttribute(name, var.data(), var.size());
  }

  std::map<std::type_index, std::function<void(adios2::IO&, const std::string&, std::any)>>
    write_functions;

  template <typename T>
  void register_write_function() {
    write_functions[std::type_index(typeid(T))] =
      [](adios2::IO& io, const std::string& name, std::any a) {
        write(io, name, std::any_cast<T>(a));
      };
  }

  template <typename T>
  void register_write_function_for_vector() {
    write_functions[std::type_index(typeid(std::vector<T>))] =
      [](adios2::IO& io, const std::string& name, std::any a) {
        write_vec(io, name, std::any_cast<std::vector<T>>(a));
      };
  }

  void write_any(adios2::IO& io, const std::string& name, std::any a) {
    auto it = write_functions.find(a.type());
    if (it != write_functions.end()) {
      it->second(io, name, a);
    } else {
      throw std::runtime_error("No write function registered for this type");
    }
  }

  void Writer::writeAttrs(const prm::Parameters& params) {
    register_write_function<double>();
    register_write_function<float>();
    register_write_function<int>();
    register_write_function<std::size_t>();
    register_write_function<unsigned int>();
    register_write_function<long int>();
    register_write_function<long double>();
    register_write_function<unsigned long int>();
    register_write_function<short>();
    register_write_function<bool>();
    register_write_function<unsigned short>();
    register_write_function<FldsBC>();
    register_write_function<PrtlBC>();
    register_write_function<Coord>();
    register_write_function<Metric>();
    register_write_function<SimEngine>();
    register_write_function<PrtlPusher>();
    register_write_function<Dimension>();
    register_write_function<std::string>();
    register_write_function_for_vector<double>();
    register_write_function_for_vector<float>();
    register_write_function_for_vector<int>();
    register_write_function_for_vector<std::size_t>();
    register_write_function_for_vector<unsigned int>();
    register_write_function_for_vector<long int>();
    register_write_function_for_vector<long double>();
    register_write_function_for_vector<unsigned long int>();
    register_write_function_for_vector<short>();
    register_write_function_for_vector<unsigned short>();
    register_write_function_for_vector<std::string>();
    register_write_function_for_vector<FldsBC>();
    register_write_function_for_vector<PrtlBC>();
    register_write_function_for_vector<Coord>();
    register_write_function_for_vector<Metric>();
    register_write_function_for_vector<SimEngine>();
    register_write_function_for_vector<PrtlPusher>();

    for (auto& [key, value] : params.allVars()) {
      try {
        write_any(m_io, key, value);
      } catch (const std::exception& e) {
        continue;
      }
    }
  }
} // namespace out