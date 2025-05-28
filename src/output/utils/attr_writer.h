/**
 * @file output/utils/attr_writer.h
 * @brief Functions to write custom type attributes to ADIOS2
 * @implements
 *   - out::writeAnyAttr -> void
 *   - out::defineAttribute<> -> void
 * @namespaces:
 *   - out::
 */
#ifndef OUTPUT_UTILS_ATTR_WRITER_H
#define OUTPUT_UTILS_ATTR_WRITER_H

#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#include <any>
#include <functional>
#include <string>
#include <typeindex>
#include <unordered_map>

namespace out {

  template <typename T>
  void defineAttribute(adios2::IO& io, const std::string& name, const std::any& value) {
    io.DefineAttribute(name, std::any_cast<T>(value));
  }

  template <typename T>
  void defineAttribute<std::vector<T>>(adios2::IO&        io,
                                       const std::string& name,
                                       const std::any&    value) {
    auto v = std::any_cast<std::vector<T>>(value);
    io.DefineAttribute(name, v.data(), v.size());
  }

  // clang-format off
  void writeAnyAttr(adios2::IO& io, const std::string& name, const std::any& value) {
      static std::unordered_map<std::type_index, std::function<void(adios2::IO&, const std::string&, const std::any&)>> handlers = {
          {typeid(int), defineAttribute<int>},
          {typeid(short), defineAttribute<short>},
          {typeid(unsigned int), defineAttribute<unsigned int>},
          {typeid(long int), defineAttribute<long int>},
          {typeid(unsigned long int), defineAttribute<unsigned long int>},
          {typeid(long long int), defineAttribute<long long int>},
          {typeid(unsigned long long int), defineAttribute<unsigned long long int>},
          {typeid(unsigned short), defineAttribute<unsigned short>},
          {typeid(float), defineAttribute<float>},
          {typeid(double), defineAttribute<double>},
          {typeid(std::string), defineAttribute<std::string>},
          {typeid(bool), defineAttribute<bool>},
          {typeid(std::vector<int>), defineAttribute<std::vector<int>>},
          {typeid(std::vector<short>), defineAttribute<std::vector<short>>},
          {typeid(std::vector<unsigned int>), defineAttribute<std::vector<unsigned int>>},
          {typeid(std::vector<long int>), defineAttribute<std::vector<long int>>},
          {typeid(std::vector<unsigned long int>), defineAttribute<std::vector<unsigned long int>>},
          {typeid(std::vector<long long int>), defineAttribute<std::vector<long long int>>},
          {typeid(std::vector<unsigned long long int>), defineAttribute<std::vector<unsigned long long int>>},
          {typeid(std::vector<unsigned short>), defineAttribute<std::vector<unsigned short>>},
          {typeid(std::vector<float>), defineAttribute<std::vector<float>>},
          {typeid(std::vector<double>), defineAttribute<std::vector<double>>},
          {typeid(std::vector<std::string>), defineAttribute<std::vector<std::string>>},
          {typeid(std::vector<bool>), defineAttribute<std::vector<bool>>},
      };

      auto it = handlers.find(value.type());
      if (it != handlers.end()) {
          it->second(io, name, value);
      } else {
          throw std::runtime_error("Unsupported type");
      }
  }

  // clang-format on

} // namespace out

#endif // OUTPUT_UTILS_ATTR_WRITER_H
