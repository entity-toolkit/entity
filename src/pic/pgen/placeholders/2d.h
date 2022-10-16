template <>
ProblemGenerator<Dim2, TypePIC>::ProblemGenerator(const SimulationParams&) {}

template <>
void ProblemGenerator<Dim2, TypePIC>::UserInitFields(const SimulationParams&,
                                                     Meshblock<Dim2, TypePIC>&) {}

template <>
void ProblemGenerator<Dim2, TypePIC>::UserInitParticles(const SimulationParams&,
                                                        Meshblock<Dim2, TypePIC>&) {}

template <>
void ProblemGenerator<Dim2, TypePIC>::UserBCFields(const real_t&,
                                                   const SimulationParams&,
                                                   Meshblock<Dim2, TypePIC>&) {}

template <>
Inline auto
ProblemGenerator<Dim2, TypePIC>::UserTargetField_br_hat(const Meshblock<Dim2, TypePIC>&,
                                                        const coord_t<Dim2>&) const -> real_t {
  return ZERO;
}

template <>
void ProblemGenerator<Dim2, TypePIC>::UserDriveParticles(const real_t&,
                                                         const SimulationParams&,
                                                         Meshblock<Dim2, TypePIC>&) {}
