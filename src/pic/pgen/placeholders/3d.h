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
void ProblemGenerator<Dim3, TypePIC>::UserDriveParticles(const real_t&,
                                                         const SimulationParams&,
                                                         Meshblock<Dim3, TypePIC>&) {}
