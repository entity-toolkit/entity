#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#if defined(MPI_ENABLED)
  #include <stdlib.h>
#endif // MPI_ENABLED

namespace user {
  using namespace ntt;

  // initializing guide field and curl(B) = J_ext at the initial time step
  template <Dimension D>
  struct InitFields {
    InitFields(array_t<real_t**>& k,
               array_t<real_t*>&  a_real,
               array_t<real_t*>&  a_imag,
               array_t<real_t*>&  a_real_inv,
               array_t<real_t*>&  a_imag_inv)
      : k { k }
      , a_real { a_real }
      , a_imag { a_imag }
      , a_real_inv { a_real_inv }
      , a_imag_inv { a_imag_inv }
      , n_modes { a_real.size() } {};

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      auto bx1_0 = ZERO;
      for (auto i = 0; i < n_modes; i++) {
        auto k_dot_r  = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1];
        bx1_0        -= TWO * k(1, i) *
                 (a_real(i) * math::sin(k_dot_r) + a_imag(i) * math::cos(k_dot_r));
        bx1_0 -= TWO * k(1, i) *
                 (a_real_inv(i) * math::sin(k_dot_r) +
                  a_imag_inv(i) * math::cos(k_dot_r));
      }
      return bx1_0;
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      auto bx2_0 = ZERO;
      for (auto i = 0; i < n_modes; i++) {
        auto k_dot_r  = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1];
        bx2_0        += TWO * k(0, i) *
                 (a_real(i) * math::sin(k_dot_r) + a_imag(i) * math::cos(k_dot_r));
        bx2_0 += TWO * k(0, i) *
                 (a_real_inv(i) * math::sin(k_dot_r) +
                  a_imag_inv(i) * math::cos(k_dot_r));
      }
      return bx2_0;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ONE;
    }

    array_t<real_t**> k;
    array_t<real_t*>  a_real;
    array_t<real_t*>  a_imag;
    array_t<real_t*>  a_real_inv;
    array_t<real_t*>  a_imag_inv;
    std::size_t       n_modes;
  };

  inline auto init_pool(int seed) -> unsigned int {
    if (seed < 0) {
      unsigned int new_seed = static_cast<unsigned int>(rand());
#if defined(MPI_ENABLED)
      MPI_Bcast(&new_seed, 1, MPI_UNSIGNED, MPI_ROOT_RANK, MPI_COMM_WORLD);
#endif // MPI_ENABLED
      return new_seed;
    } else {
      return static_cast<unsigned int>(seed);
    }
  }

  template <Dimension D>
  inline auto init_wavenumbers() -> std::vector<std::vector<real_t>> {
    if constexpr (D == Dim::_2D) {
      return {
        {  1, 0 },
        {  0, 1 },
        {  1, 1 },
        { -1, 1 }
      };
    } else if constexpr (D == Dim::_3D) {
      return {
        {  1,  0, 1 },
        {  0,  1, 1 },
        { -1,  0, 1 },
        {  0, -1, 1 }
      };
    } else {
      raise::Error("Invalid dimension", HERE);
      return {};
    }
  }

  // external current definition
  template <Dimension D>
  struct ExternalCurrent {
    ExternalCurrent(real_t                            dB,
                    real_t                            om0,
                    real_t                            g0,
                    std::vector<std::vector<real_t>>& wavenumbers,
                    random_number_pool_t&             random_pool,
                    real_t                            Lx,
                    real_t                            Ly,
                    real_t                            Lz,
                    bool                              restart)
      : wavenumbers { wavenumbers }
      , n_modes { wavenumbers.size() }
      , dB { dB }
      , Lx { Lx }
      , Ly { Ly }
      , Lz { Lz }
      , omega_0 { om0 }
      , gamma_0 { g0 }
      , k { "wavevector", D, n_modes }
      , a_real { "a_real", n_modes }
      , a_imag { "a_imag", n_modes }
      , a_real_inv { "a_real_inv", n_modes }
      , a_imag_inv { "a_imag_inv", n_modes }
      , A0 { "A0", n_modes }
      , restart { restart } {
      // initializing wavevectors
      auto k_host = Kokkos::create_mirror_view(k);
      if constexpr (D == Dim::_2D) {
        for (auto i = 0u; i < n_modes; i++) {
          k_host(0, i) = constant::TWO_PI * wavenumbers[i][0] / Lx;
          k_host(1, i) = constant::TWO_PI * wavenumbers[i][1] / Ly;
        }
      }
      if constexpr (D == Dim::_3D) {
        for (auto i = 0u; i < n_modes; i++) {
          k_host(0, i) = constant::TWO_PI * wavenumbers[i][0] / Lx;
          k_host(1, i) = constant::TWO_PI * wavenumbers[i][1] / Ly;
          k_host(2, i) = constant::TWO_PI * wavenumbers[i][2] / Lz;
        }
      }


auto a_real_host     = Kokkos::create_mirror_view(a_real);
auto a_imag_host     = Kokkos::create_mirror_view(a_imag);
auto a_real_inv_host = Kokkos::create_mirror_view(a_real_inv);
auto a_imag_inv_host = Kokkos::create_mirror_view(a_imag_inv);
auto A0_host         = Kokkos::create_mirror_view(A0);

if (!restart) {
  real_t prefac { ZERO };
  if constexpr (D == Dim::_2D) {
    prefac = HALF;
  } else if constexpr (D == Dim::_3D) {
    prefac = constant::SQRT2;
  }
  for (auto i = 0u; i < n_modes; i++) {
    auto k_perp = math::sqrt(
      k_host(0, i) * k_host(0, i) + k_host(1, i) * k_host(1, i));
    auto phase         = constant::TWO_PI / 6.;
    A0_host(i)         = dB / math::sqrt((real_t)n_modes) / k_perp * prefac;
    a_real_host(i)     = A0_host(i) * math::cos(phase);
    a_imag_host(i)     = A0_host(i) * math::sin(phase);
    phase              = constant::TWO_PI / 3;
    a_imag_inv_host(i) = A0_host(i) * math::cos(phase);
    a_real_inv_host(i) = A0_host(i) * math::sin(phase);
  }
} else {

  // Todo: Reading the (last) checkpoint file for amplitudes (#FRONTIER) 
  real_t prefac { ZERO };
  if constexpr (D == Dim::_2D) {
    prefac = HALF;
  } else if constexpr (D == Dim::_3D) {
    prefac = constant::SQRT2;
  }
  for (auto i = 0u; i < n_modes; i++) {
    auto k_perp = math::sqrt(
    k_host(0, i) * k_host(0, i) + k_host(1, i) * k_host(1, i));
    A0_host(i)         = dB / math::sqrt((real_t)n_modes) / k_perp * prefac;
  }


const std::string prefix = "antenna_amplitudes_step";
int max_step = -1;
std::string latest_file;

for (const auto& entry : std::filesystem::directory_iterator(".")) {
  auto fname = entry.path().filename().string();

  if (fname.rfind(prefix, 0) == 0 &&  // starts with prefix
      fname.size() > prefix.size() + 10 &&  // ensure room for "_stepX.csv"
      fname.substr(fname.size() - 4) == ".csv") {

    std::string suffix = fname.substr(prefix.size()); // what follows the prefix
    if (suffix.rfind("_step", 0) == 0) {
      // Extract the number between "_step" and ".csv"
      std::string step_str = suffix.substr(5, suffix.size() - 9); // 5 = len("_step"), 9 = 5 + 4 for ".csv"

      try {
        int step = std::stoi(step_str);
        if (step > max_step) {
          max_step = step;
          latest_file = entry.path().string();
        }
      } catch (const std::exception& e) {
        std::cerr << "Failed to parse step number from filename: " << fname
                  << " (" << e.what() << ")" << std::endl;
      }
    }
  }
}

if (latest_file.empty()) {
  throw std::runtime_error("No checkpoint file found to restore antenna amplitudes.");
}

int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

if (rank == 0) {
  std::cout << "Rank 0 reading file: " << latest_file << std::endl;

  std::ifstream infile(latest_file);
  if (!infile) throw std::runtime_error("Failed to open file: " + latest_file);

  std::string line;
  std::getline(infile, line); // skip header

  size_t i = 0;
  while (std::getline(infile, line) && i < a_real_host.extent(0)) {
    std::istringstream ss(line);
    std::string token;
    std::getline(ss, token, ','); // skip index
    std::getline(ss, token, ','); a_real_host(i)     = std::stod(token);
    std::getline(ss, token, ','); a_imag_host(i)     = std::stod(token);
    std::getline(ss, token, ','); a_real_inv_host(i) = std::stod(token);
    std::getline(ss, token, ','); a_imag_inv_host(i) = std::stod(token);
    ++i;
  }

  if (i != a_real_host.extent(0)) {
    throw std::runtime_error("Mismatch in number of antenna modes read from checkpoint.");
  }
}

// Then broadcast the arrays to other ranks
MPI_Bcast(a_real_host.data(),     a_real_host.extent(0), MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(a_imag_host.data(),     a_imag_host.extent(0), MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(a_real_inv_host.data(), a_real_inv_host.extent(0), MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(a_imag_inv_host.data(), a_imag_inv_host.extent(0), MPI_DOUBLE, 0, MPI_COMM_WORLD);

}

// Copy back to device
Kokkos::deep_copy(a_real,     a_real_host);
Kokkos::deep_copy(a_imag,     a_imag_host);
Kokkos::deep_copy(a_real_inv, a_real_inv_host);
Kokkos::deep_copy(a_imag_inv, a_imag_inv_host);
Kokkos::deep_copy(A0,         A0_host);
Kokkos::deep_copy(k,          k_host);

      
    };

    Inline auto jx1(const coord_t<D>& x_Ph) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return ZERO;
      }
      if constexpr (D == Dim::_3D) {
        real_t jx1_ant = ZERO;
        for (auto i = 0u; i < n_modes; i++) {
          auto k_dot_r = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1] + k(2, i) * x_Ph[2];
          jx1_ant -= TWO * k(0, i) * k(2, i) *
                     (a_real_inv(i) * math::cos(k_dot_r) -
                      a_imag_inv(i) * math::sin(k_dot_r));
        }
        return jx1_ant;
      }
    }

    Inline auto jx2(const coord_t<D>& x_Ph) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return ZERO;
      } else if constexpr (D == Dim::_3D) {
        real_t jx2_ant = ZERO;
        for (auto i = 0u; i < n_modes; i++) {
          auto k_dot_r = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1] + k(2, i) * x_Ph[2];
          jx2_ant -= TWO * k(1, i) * k(2, i) *
                     (a_real_inv(i) * math::cos(k_dot_r) -
                      a_imag_inv(i) * math::sin(k_dot_r));
        }
        return jx2_ant;
      }
    }

    Inline auto jx3(const coord_t<D>& x_Ph) const -> real_t {
      if constexpr (D == Dim::_2D) {
        real_t jx3_ant = ZERO;
        for (auto i = 0u; i < n_modes; i++) {
          auto k_perp_sq  = k(0, i) * k(0, i) + k(1, i) * k(1, i);
          auto k_dot_r    = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1];
          jx3_ant        += TWO * k_perp_sq *
                     (a_real(i) * math::cos(k_dot_r) -
                      a_imag(i) * math::sin(k_dot_r));
          jx3_ant += TWO * k_perp_sq *
                     (a_real_inv(i) * math::cos(k_dot_r) -
                      a_imag_inv(i) * math::sin(k_dot_r));
        }
        return jx3_ant;
      } else if constexpr (D == Dim::_3D) {
        real_t jx3_ant = ZERO;
        for (auto i = 0u; i < n_modes; i++) {
          auto k_perp_sq = k(0, i) * k(0, i) + k(1, i) * k(1, i);
          auto k_dot_r = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1] + k(2, i) * x_Ph[2];
          jx3_ant += TWO * k_perp_sq *
                     (a_real_inv(i) * math::cos(k_dot_r) -
                      a_imag_inv(i) * math::sin(k_dot_r));
        }
        return jx3_ant;
      }
    }

  private:
    const std::vector<std::vector<real_t>> wavenumbers;
    const std::size_t                      n_modes;
    const real_t                           dB, Lx, Ly, Lz;
    const bool                             restart;

  public:
    const real_t      omega_0, gamma_0;
    array_t<real_t**> k;
    array_t<real_t*>  a_real;
    array_t<real_t*>  a_imag;
    array_t<real_t*>  a_real_inv;
    array_t<real_t*>  a_imag_inv;
    array_t<real_t*>  A0;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t                     temperature, dB, omega_0, gamma_0;
    const real_t                     Lx, Ly, Lz, escape_dist;
    const int                        random_seed;
    std::vector<std::vector<real_t>> wavenumbers;
    random_number_pool_t             random_pool;
    int checkpoint_keep;
    simtime_t checkpoint_interval_time;
    timestep_t checkpoint_interval;
    tools::Tracker           m_tracker;
    bool m_enabled;
    bool restart; 

    // debugging, will delete later
    real_t total_sum           = ZERO;
    real_t total_sum_inv       = ZERO;
    real_t number_of_timesteps = ZERO;

    ExternalCurrent<D> ext_current;
    InitFields<D>      init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , temperature { p.template get<real_t>("setup.temperature") }
      , dB { p.template get<real_t>("setup.dB", ONE) }
      , omega_0 { p.template get<real_t>("setup.omega_0") }
      , gamma_0 { p.template get<real_t>("setup.gamma_0") }
      , wavenumbers { init_wavenumbers<D>() }
      , random_seed { p.template get<int>("setup.seed", -1) }
      , random_pool { init_pool(random_seed) }
      , checkpoint_interval { params.template get<timestep_t>("checkpoint.interval") }
      , checkpoint_interval_time { params.template get<simtime_t>("checkpoint.interval_time") }
      , checkpoint_keep { params.template get<int>("checkpoint.keep") }
      , Lx { global_domain.mesh().extent(in::x1).second -
             global_domain.mesh().extent(in::x1).first }
      , Ly { global_domain.mesh().extent(in::x2).second -
             global_domain.mesh().extent(in::x2).first }
      , Lz { global_domain.mesh().extent(in::x3).second -
             global_domain.mesh().extent(in::x3).first }
      , restart { p.template get<bool>("setup.restart_modes", false) }
      , escape_dist { p.template get<real_t>("setup.escape_dist", HALF * Lx) }
      , ext_current { dB, omega_0, gamma_0, wavenumbers, random_pool, Lx, Ly, Lz, restart }
      , init_flds { ext_current.k,
                    ext_current.a_real,
                    ext_current.a_imag,
                    ext_current.a_real_inv,
                    ext_current.a_imag_inv } {
                      m_tracker.init("checkpoint", checkpoint_interval, checkpoint_interval_time);
                      m_enabled = checkpoint_keep != 0;
                    }

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist  = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                      *local_domain.random_pool,
                                                      temperature);
      const auto spatial_dist = arch::UniformInjector<S, M, arch::Maxwellian>(
        energy_dist,
        { 1, 2 });
      arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
        params,
        local_domain,
        spatial_dist,
        ONE);

    };

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {
      // update amplitudes of antenna
      const auto  dt = params.template get<real_t>("algorithms.timestep.dt");
      const auto& ext_curr = ext_current;
      
      Kokkos::parallel_for(
        "Antenna amplitudes",
        wavenumbers.size(),
        ClassLambda(index_t i) {
          auto       generator  = random_pool.get_state();
          const auto u_imag     = Random<real_t>(generator) - HALF;
          const auto u_real     = Random<real_t>(generator) - HALF;
          const auto u_real_inv = Random<real_t>(generator) - HALF;
          const auto u_imag_inv = Random<real_t>(generator) - HALF;
          random_pool.free_state(generator);

          auto a_real_prev     = ext_curr.a_real(i);
          auto a_imag_prev     = ext_curr.a_imag(i);
          auto a_real_inv_prev = ext_curr.a_real_inv(i);
          auto a_imag_inv_prev = ext_curr.a_imag_inv(i);
          ext_curr.a_real(i) = (a_real_prev * math::cos(ext_curr.omega_0 * dt) +
                                a_imag_prev * math::sin(ext_curr.omega_0 * dt)) *
                                 math::exp(-ext_curr.gamma_0 * dt) +
                               ext_curr.A0(i) *
                                 math::sqrt(TWELVE * ext_curr.gamma_0 / dt) *
                                 u_real * dt;

          ext_curr.a_imag(i) = (a_imag_prev * math::cos(ext_curr.omega_0 * dt) -
                                a_real_prev * math::sin(ext_curr.omega_0 * dt)) *
                                 math::exp(-ext_curr.gamma_0 * dt) +
                               ext_curr.A0(i) *
                                 math::sqrt(TWELVE * ext_curr.gamma_0 / dt) *
                                 u_imag * dt;

          ext_curr.a_real_inv(
            i) = (a_real_inv_prev * math::cos(-ext_curr.omega_0 * dt) +
                  a_imag_inv_prev * math::sin(-ext_curr.omega_0 * dt)) *
                   math::exp(-ext_curr.gamma_0 * dt) +
                 ext_curr.A0(i) * math::sqrt(TWELVE * ext_curr.gamma_0 / dt) *
                   u_real_inv * dt;

          ext_curr.a_imag_inv(
            i) = (a_imag_inv_prev * math::cos(-ext_curr.omega_0 * dt) -
                  a_real_inv_prev * math::sin(-ext_curr.omega_0 * dt)) *
                   math::exp(-ext_curr.gamma_0 * dt) +
                 ext_curr.A0(i) * math::sqrt(TWELVE * ext_curr.gamma_0 / dt) *
                   u_imag_inv * dt;
        });

      // particle escape (resample velocities)
      const auto energy_dist = arch::Maxwellian<S, M>(domain.mesh.metric,
                                                      *domain.random_pool,
                                                      temperature);
      for (const auto& sp : { 0, 1 }) {
        if (domain.species[sp].npld() > 1) {
          const auto& ux1 = domain.species[sp].ux1;
          const auto& ux2 = domain.species[sp].ux2;
          const auto& ux3 = domain.species[sp].ux3;
          const auto& pld = domain.species[sp].pld;
          const auto& tag = domain.species[sp].tag;
          const auto  L   = escape_dist;
	  // printf("Entering the escape loop %d, L = %f\n", sp, L);
          Kokkos::parallel_for(
            "UpdatePld",
            domain.species[sp].npart(),
            Lambda(index_t p) {
              if (tag(p) == ParticleTag::dead) {
                return;
              }
              const auto gamma = math::sqrt(
                ONE + ux1(p) * ux1(p) + ux2(p) * ux2(p) + ux3(p) * ux3(p));
              pld(p, 0) += ux1(p) * dt / gamma;
              pld(p, 1) += ux2(p) * dt / gamma;

              if (math::abs(pld(p, 0) > L) or (math::abs(pld(p,1)) > L)) {
                coord_t<D>      x_Ph { ZERO };
                vec_t<Dim::_3D> u_Mxw { ZERO };
                energy_dist(x_Ph, u_Mxw);
                ux1(p)    = u_Mxw[0];
                ux2(p)    = u_Mxw[1];
                ux3(p)    = u_Mxw[2];
                pld(p, 0) = ZERO;
                pld(p, 1) = ZERO;
              }
            });
        }
      }

// Todo: Write checkpoint file for amplitudes (#FRONTIER) 
if (m_enabled && m_tracker.shouldWrite(step, time)) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    // Create mirror views and copy from device to host
    auto a_real_h       = Kokkos::create_mirror_view(ext_curr.a_real);
    auto a_imag_h       = Kokkos::create_mirror_view(ext_curr.a_imag);
    auto a_real_inv_h   = Kokkos::create_mirror_view(ext_curr.a_real_inv);
    auto a_imag_inv_h   = Kokkos::create_mirror_view(ext_curr.a_imag_inv);

    Kokkos::deep_copy(a_real_h, ext_curr.a_real);
    Kokkos::deep_copy(a_imag_h, ext_curr.a_imag);
    Kokkos::deep_copy(a_real_inv_h, ext_curr.a_real_inv);
    Kokkos::deep_copy(a_imag_inv_h, ext_curr.a_imag_inv);

    std::string prefix = "antenna_amplitudes";
    std::string filename = prefix + "_step" + std::to_string(step+1) + ".csv";

    // Write CSV
    std::ofstream outfile(filename);
    outfile << "k_index,a_real,a_imag,a_real_inv,a_imag_inv\n";
    for (std::size_t i = 0; i < a_real_h.extent(0); ++i) {
      outfile << i << ","
              << a_real_h(i)     << ","
              << a_imag_h(i)     << ","
              << a_real_inv_h(i) << ","
              << a_imag_inv_h(i) << "\n";
    }
    outfile.close();

    // Todo: Prune (all) old checkpoint files for amplitudes, have to keep previous checkpoints still (#FRONTIER) 
    namespace fs = std::filesystem;
    std::vector<std::pair<int, fs::path>> files;

    for (const auto& entry : fs::directory_iterator(".")) {
      const auto& path = entry.path();
      std::string fname = path.filename().string();
      if (fname.find(prefix + "_step") == 0 && fname.size() > prefix.size() + 5) {
        std::size_t pos1 = prefix.size() + 5;
        std::size_t pos2 = fname.find(".csv");
        if (pos2 != std::string::npos) {
          std::string num_str = fname.substr(pos1, pos2 - pos1);
          try {
            int s = std::stoi(num_str);
            files.emplace_back(s, path);
          } catch (...) {
            // Ignore malformed filenames
          }
        }
      }
    }

    std::sort(files.begin(), files.end());

    while (files.size() > static_cast<std::size_t>(checkpoint_keep)) {
      fs::remove(files.front().second);
      files.erase(files.begin());
    }
  }
}

    }
  };
} // namespace user

#endif
