#include "global.h"
#include "input.h"

#include <map>
#include <string_view>
#include <string>
#include <utility>

namespace ntt::io {
void InputParams::set_parameter(std::string_view block,
                                std::string_view variable, bool value) {
  m_params.insert({{block, variable}, new Param(value)});
}
void InputParams::set_parameter(std::string_view block,
                                std::string_view variable, int value) {
  m_params.insert({{block, variable}, new Param(value)});
}
void InputParams::set_parameter(std::string_view block,
                                std::string_view variable, float value) {
  m_params.insert({{block, variable}, new Param(value)});
}
void InputParams::set_parameter(std::string_view block,
                                std::string_view variable, double value) {
  m_params.insert({{block, variable}, new Param(value)});
}
void InputParams::set_parameter(std::string_view block,
                                std::string_view variable,
                                std::string_view value) {
  m_params.insert({{block, variable}, new Param(value)});
}

auto InputParams::get_parameter(std::string_view block,
                                std::string_view variable) -> Param * {
  return m_params[{block, variable}];
}
} // namespace ntt::io
