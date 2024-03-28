
namespace ntt {

#if defined(MINKOWSKI_METRIC) || defined(GRPIC_ENGINE)
  #define PrtlCoordD D
#else
  #define PrtlCoordD Dim3
#endif

  // Field IDs used for io
  enum class FieldID {
    E,      // Electric fields
    divE,   // Divergence of electric fields
    D,      // Electric fields (GR)
    divD,   // Divergence of electric fields (GR)
    B,      // Magnetic fields
    H,      // Magnetic fields (GR)
    J,      // Current density
    A,      // Vector potential
    T,      // Particle distribution moments
    Rho,    // Particle mass density
    Charge, // Charge density
    N,      // Particle number density
    Nppc    // Raw number of particles per each cell
  };

  enum class PrtlID {
    X, // Position
    U, // 4-Velocity / 4-Momentum
    W  // Weight
  };
} // namespace ntt