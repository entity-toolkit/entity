template <>
ProblemGenerator<Dim3, TypePIC>::ProblemGenerator(const SimulationParams&) {}

template <>
void ProblemGenerator<Dim3, TypePIC>::UserInitFields(const SimulationParams&,
                                                     Meshblock<Dim3, TypePIC>&) {}

template <>
void ProblemGenerator<Dim3, TypePIC>::UserInitParticles(const SimulationParams&,
                                                        Meshblock<Dim3, TypePIC>&) {}

template <>
void ProblemGenerator<Dim3, TypePIC>::UserBCFields(const real_t&,
                                                   const SimulationParams&,
                                                   Meshblock<Dim3, TypePIC>&) {}

template <>
Inline auto
ProblemGenerator<Dim3, TypePIC>::UserTargetField_br_hat(const Meshblock<Dim3, TypePIC>&,
                                                        const coord_t<Dim3>&) const -> real_t {
  return ZERO;
}
