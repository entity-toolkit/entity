#ifndef IO_INPUT_H
#define IO_INPUT_H

#include "global.h"

#include <toml/toml.hpp>

#include <utility>
#include <map>
#include <string>
#include <string_view>

namespace ntt {
  namespace io {
    class Param {
      bool m_value_bool;
      int m_value_int;
      float m_value_float;
      double m_value_double;
      std::string_view m_value_string;
    public:
      const bool * const value_bool;
      const int * const value_int;
      const float * const value_float;
      const double * const value_double;
      const std::string_view * const value_string;
      Param(bool value) : m_value_bool(value),
                value_bool(&m_value_bool),
                value_int(nullptr),
                value_float(nullptr),
                value_double(nullptr),
                value_string(nullptr) {};
      Param(int value) : m_value_int(value),
                value_bool(nullptr),
                value_int(&m_value_int),
                value_float(nullptr),
                value_double(nullptr),
                value_string(nullptr) {};
      Param(float value) : m_value_float(value),
                value_bool(nullptr),
                value_int(nullptr),
                value_float(&m_value_float),
                value_double(nullptr),
                value_string(nullptr) {};
      Param(double value) : m_value_double(value),
                value_bool(nullptr),
                value_int(nullptr),
                value_float(nullptr),
                value_double(&m_value_double),
                value_string(nullptr) {};
      Param(std::string_view value) : m_value_string(value),
                value_bool(nullptr),
                value_int(nullptr),
                value_float(nullptr),
                value_double(nullptr),
                value_string(&m_value_string) {};
      ~Param() = default;
    };

    class InputParams {
    private:
      std::string m_input_filename;
      std::map<std::pair<std::string_view, std::string_view>, Param*> m_params;
    public:
      void set_parameter(std::string_view block, std::string_view variable, bool value);
      void set_parameter(std::string_view block, std::string_view variable, int value);
      void set_parameter(std::string_view block, std::string_view variable, float value);
      void set_parameter(std::string_view block, std::string_view variable, double value);
      void set_parameter(std::string_view block, std::string_view variable, std::string_view value);

      Param* get_parameter(std::string_view block, std::string_view variable);

      void set_input_filename(std::string_view input_filename) { m_input_filename = input_filename; }
      auto get_input_filename() const -> std::string_view { return m_input_filename; }
    };

    // template<typename T>
    // T readFromInput(std::string_view blockname, std::string_view variable) {
    // }
  }
}

#endif
