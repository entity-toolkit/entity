/**
 * @file output/utils/interpret_prompt.h
 * @brief
 * Defines the function that interprets ...
 * ... the user-defined species, e.g. when computing moments
 * @implements
 *   - out::InterpretSpecies -> std::vector<spidx_t>
 *   - out::InterpretComponents -> std::vector<std::vector<unsigned short>>
 * @cpp:
 *   - interpret_prompt.cpp
 * @namespaces:
 *   - out::
 * @example out::InterpretSpecies takes "Foo_2_3_4" and returns {2, 3, 4}
 * @example out::InterpretComponents takes "Bar_ti" and returns {{0, 1}, {0, 2}}
 */

#ifndef OUTPUT_UTILS_INTERPRET_PROMPT_H
#define OUTPUT_UTILS_INTERPRET_PROMPT_H

#include "global.h"

#include <string>
#include <vector>

namespace out {

  auto InterpretSpecies(const std::string&) -> std::vector<spidx_t>;

  auto InterpretComponents(const std::vector<std::string>&)
    -> std::vector<std::vector<unsigned short>>;

} // namespace out

#endif // OUTPUT_UTILS_INTERPRET_PROMPT_H
