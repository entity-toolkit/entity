#ifndef FRAMEWORK_METRICS_UTILS_PARAM_FORGR_H
#define FRAMEWORK_METRICS_UTILS_PARAM_FORGR_H

[[nodiscard]] auto getParameter(const std::string& parameter) const -> real_t override {
  if (parameter == "spin") {
    return a;
  } else if (parameter == "rhorizon") {
    return rh_;
  } else if (parameter == "rg") {
    return rg_;
  } else {
    NTTHostError("Unknown parameter: " + parameter);
  }
};

#endif    // FRAMEWORK_METRICS_UTILS_PARAM_FORGR_H