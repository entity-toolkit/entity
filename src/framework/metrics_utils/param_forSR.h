#ifndef FRAMEWORK_METRICS_UTILS_PARAM_FORSR_H
#define FRAMEWORK_METRICS_UTILS_PARAM_FORSR_H

[[nodiscard]]
auto getParameter(const std::string& parameter) const -> real_t override {
  NTTHostError("Unknown parameter: " + parameter);
};

#endif // FRAMEWORK_METRICS_UTILS_PARAM_FORSR_H