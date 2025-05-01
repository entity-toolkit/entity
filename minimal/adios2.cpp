#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#if defined(MPI_ENABLED)
  #include <mpi.h>
  #define MPI_ROOT_RANK 0
#endif

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

auto pad(const std::string&, std::size_t, char, bool = false) -> std::string;

template <typename Func, typename... Args>
void CallOnce(Func, Args&&...);

template <typename T>
auto define_constdim_array(adios2::IO&,
                           const std::vector<std::size_t>&,
                           const std::vector<std::size_t>&,
                           const std::vector<std::size_t>&) -> std::string;

template <typename T>
auto define_unknowndim_array(adios2::IO&) -> std::string;

template <typename T, typename A>
void put_constdim_array(adios2::IO&, adios2::Engine&, const A&, const std::string&);

template <typename T>
void put_unknowndim_array(adios2::IO&,
                          adios2::Engine&,
                          const Kokkos::View<T*>&,
                          std::size_t,
                          const std::string&);

auto main(int argc, char** argv) -> int {
  try {
    Kokkos::initialize(argc, argv);
#if defined(MPI_ENABLED)
    MPI_Init(&argc, &argv);
    adios2::ADIOS adios { MPI_COMM_WORLD };
#else
    adios2::ADIOS adios;
#endif

    std::string engine = "hdf5";
    if (argc > 1) {
      engine = std::string(argv[1]);
      if (engine != "hdf5" && engine != "bp") {
        throw std::invalid_argument("Engine must be either 'hdf5' or 'bp'");
      }
    }
    const std::string format = (engine == "hdf5") ? "h5" : "bp";

    auto io = adios.DeclareIO("Test::Output");
    io.SetEngine(engine);

    io.DefineAttribute("Attr::Int", 42);
    io.DefineAttribute("Attr::Float", 42.0f);
    io.DefineAttribute("Attr::Double", 42.0);
    io.DefineAttribute("Attr::String", engine);

    io.DefineVariable<int>("Var::Int");
    io.DefineVariable<std::size_t>("Var::Size_t");

    int rank = 0, size = 1;
#if defined(MPI_ENABLED)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    // global sizes
    const std::size_t Sx_1d = (size - 1) * 1000 + 230;
    const std::size_t Sx_2d = 100, Sy_2d = (size - 1) * 100 + 23;
    const std::size_t Sx_3d = 10, Sy_3d = 10, Sz_3d = (size - 1) * 10 + 3;

    // local sizes
    const std::size_t sx_1d = (rank != size - 1) ? 1000 : 230;
    const std::size_t sx_2d = 100, sy_2d = (rank != size - 1) ? 100 : 23;
    const std::size_t sx_3d = 10, sy_3d = 10, sz_3d = (rank != size - 1) ? 10 : 3;

    // displacements
    const std::size_t ox_1d = rank * 1000;
    const std::size_t ox_2d = 0, oy_2d = rank * 100;
    const std::size_t ox_3d = 0, oy_3d = 0, oz_3d = rank * 10;

    CallOnce(
      [](auto&& size) {
        std::cout << "Running ADIOS2 test" << std::endl;
#if defined(MPI_ENABLED)
        std::cout << "- Number of MPI ranks: " << size << std::endl;
#else
        (void)size;
        std::cout << "- No MPI" << std::endl;
#endif
      },
      size);

    std::vector<std::string> vars;

    {
      vars.push_back(
        define_constdim_array<float>(io, { Sx_1d }, { ox_1d }, { sx_1d }));
      vars.push_back(define_constdim_array<float>(io,
                                                  { Sx_2d, Sy_2d },
                                                  { ox_2d, oy_2d },
                                                  { sx_2d, sy_2d }));
      vars.push_back(define_constdim_array<float>(io,
                                                  { Sx_3d, Sy_3d, Sz_3d },
                                                  { ox_3d, oy_3d, oz_3d },
                                                  { sx_3d, sy_3d, sz_3d }));
      vars.push_back(
        define_constdim_array<double>(io, { Sx_1d }, { ox_1d }, { sx_1d }));
      vars.push_back(define_constdim_array<double>(io,
                                                   { Sx_2d, Sy_2d },
                                                   { ox_2d, oy_2d },
                                                   { sx_2d, sy_2d }));
      vars.push_back(define_constdim_array<double>(io,
                                                   { Sx_3d, Sy_3d, Sz_3d },
                                                   { ox_3d, oy_3d, oz_3d },
                                                   { sx_3d, sy_3d, sz_3d }));
    }

    {
      vars.push_back(define_unknowndim_array<float>(io));
      vars.push_back(define_unknowndim_array<double>(io));
      vars.push_back(define_unknowndim_array<int>(io));
    }

    Kokkos::View<float*>  constdim_1d_f { "constdim_1d_f", sx_1d };
    Kokkos::View<float**> constdim_2d_f { "constdim_2d_f", sx_2d, sy_2d };
    Kokkos::View<float***> constdim_3d_f { "constdim_3d_f", sx_3d, sy_3d, sz_3d };

    Kokkos::View<double*>  constdim_1d_d { "constdim_1d_d", sx_1d };
    Kokkos::View<double**> constdim_2d_d { "constdim_2d_d", sx_2d, sy_2d };
    Kokkos::View<double***> constdim_3d_d { "constdim_3d_d", sx_3d, sy_3d, sz_3d };

    {
      // fill 1d
      Kokkos::parallel_for(
        "fill_constdim_1d_f",
        Kokkos::RangePolicy<>(0, sx_1d),
        KOKKOS_LAMBDA(std::size_t i) {
          constdim_1d_f(i) = static_cast<float>(ox_1d + i);
          constdim_1d_d(i) = static_cast<double>(ox_1d + i);
        });

      // fill 2d
      Kokkos::parallel_for(
        "fill_constdim_2d_f",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { sx_2d, sy_2d }),
        KOKKOS_LAMBDA(std::size_t i, std::size_t j) {
          constdim_2d_f(i, j) = static_cast<float>(ox_2d + i + (oy_2d + j) * Sx_2d);
          constdim_2d_d(i, j) = static_cast<double>(ox_2d + i + (oy_2d + j) * Sx_2d);
        });

      // fill 3d
      Kokkos::parallel_for(
        "fill_constdim_3d_f",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { sx_3d, sy_3d, sz_3d }),
        KOKKOS_LAMBDA(std::size_t i, std::size_t j, std::size_t k) {
          constdim_3d_f(i, j, k) = static_cast<float>(
            ox_3d + i + (oy_3d + j + (oz_3d + k) * Sy_3d) * Sx_3d);
          constdim_3d_d(i, j, k) = static_cast<double>(
            ox_3d + i + (oy_3d + j + (oz_3d + k) * Sy_3d) * Sx_3d);
        });
    }

    {
      // test multiple file mode
      const std::string path = "steps";
      CallOnce(
        [](auto&& path) {
          const std::filesystem::path parent_path { path };
          if (std::filesystem::exists(parent_path)) {
            std::filesystem::remove_all(parent_path);
          }
          std::filesystem::create_directory(path);
        },
        path);
      for (auto step { 0u }; step < 5u; ++step) {
        const std::string filename = path + "/step_" +
                                     pad(std::to_string(step * 20u), 6, '0') +
                                     "." + format;
        auto writer = io.Open(filename, adios2::Mode::Write);
        writer.BeginStep();

        {
          // constant dim arrays
          put_constdim_array<float, decltype(constdim_1d_f)>(io,
                                                             writer,
                                                             constdim_1d_f,
                                                             vars[0]);
          put_constdim_array<float, decltype(constdim_2d_f)>(io,
                                                             writer,
                                                             constdim_2d_f,
                                                             vars[1]);
          put_constdim_array<float, decltype(constdim_3d_f)>(io,
                                                             writer,
                                                             constdim_3d_f,
                                                             vars[2]);
          put_constdim_array<double, decltype(constdim_1d_d)>(io,
                                                              writer,
                                                              constdim_1d_d,
                                                              vars[3]);
          put_constdim_array<double, decltype(constdim_2d_d)>(io,
                                                              writer,
                                                              constdim_2d_d,
                                                              vars[4]);
          put_constdim_array<double, decltype(constdim_3d_d)>(io,
                                                              writer,
                                                              constdim_3d_d,
                                                              vars[5]);
        }

        {
          // unknown dim arrays
          const std::size_t nelems = static_cast<std::size_t>(
            (std::sin((step + 1 + rank) * 0.25) + 2.0) * 1000.0);

          Kokkos::View<float*>  unknowndim_f { "unknowndim_f", nelems };
          Kokkos::View<double*> unknowndim_d { "unknowndim_d", nelems };
          Kokkos::View<int*>    unknowndim_i { "unknowndim_i", nelems };

          // fill unknown dim arrays
          Kokkos::parallel_for(
            "fill_unknowndim",
            Kokkos::RangePolicy<>(0, nelems),
            KOKKOS_LAMBDA(std::size_t i) {
              unknowndim_f(i) = static_cast<float>(i + step * 1000);
              unknowndim_d(i) = static_cast<double>(i + step * 1000);
              unknowndim_i(i) = static_cast<int>(i + step * 1000);
            });

          put_unknowndim_array<float>(io, writer, unknowndim_f, nelems, vars[6]);
          put_unknowndim_array<double>(io, writer, unknowndim_d, nelems, vars[7]);
          put_unknowndim_array<int>(io, writer, unknowndim_i, nelems, vars[8]);
        }

        writer.EndStep();
        writer.Close();
      }
    }
    {
      // test single file mode
      const std::string filename = "allsteps." + format;
      adios2::Mode      mode     = adios2::Mode::Write;
      for (auto step { 0u }; step < 5u; ++step) {
        auto writer = io.Open(filename, mode);
        writer.BeginStep();

        {
          // constant dim arrays
          put_constdim_array<float, decltype(constdim_1d_f)>(io,
                                                             writer,
                                                             constdim_1d_f,
                                                             vars[0]);
          put_constdim_array<float, decltype(constdim_2d_f)>(io,
                                                             writer,
                                                             constdim_2d_f,
                                                             vars[1]);
          put_constdim_array<float, decltype(constdim_3d_f)>(io,
                                                             writer,
                                                             constdim_3d_f,
                                                             vars[2]);
          put_constdim_array<double, decltype(constdim_1d_d)>(io,
                                                              writer,
                                                              constdim_1d_d,
                                                              vars[3]);
          put_constdim_array<double, decltype(constdim_2d_d)>(io,
                                                              writer,
                                                              constdim_2d_d,
                                                              vars[4]);
          put_constdim_array<double, decltype(constdim_3d_d)>(io,
                                                              writer,
                                                              constdim_3d_d,
                                                              vars[5]);
        }

        {
          // unknown dim arrays
          const std::size_t nelems = static_cast<std::size_t>(
            (std::sin((step + 1 + rank) * 0.25) + 2.0) * 1000.0);

          Kokkos::View<float*>  unknowndim_f { "unknowndim_f", nelems };
          Kokkos::View<double*> unknowndim_d { "unknowndim_d", nelems };
          Kokkos::View<int*>    unknowndim_i { "unknowndim_i", nelems };

          // fill unknown dim arrays
          Kokkos::parallel_for(
            "fill_unknowndim",
            Kokkos::RangePolicy<>(0, nelems),
            KOKKOS_LAMBDA(std::size_t i) {
              unknowndim_f(i) = static_cast<float>(i + step * 1000);
              unknowndim_d(i) = static_cast<double>(i + step * 1000);
              unknowndim_i(i) = static_cast<int>(i + step * 1000);
            });

          put_unknowndim_array<float>(io, writer, unknowndim_f, nelems, vars[6]);
          put_unknowndim_array<double>(io, writer, unknowndim_d, nelems, vars[7]);
          put_unknowndim_array<int>(io, writer, unknowndim_i, nelems, vars[8]);
        }

        writer.EndStep();
        writer.Close();
        mode = adios2::Mode::Append;
      }
    }
  } catch (const std::exception& e) {
#if defined(MPI_ENABLED)
    if (MPI_COMM_WORLD != MPI_COMM_NULL) {
      MPI_Finalize();
    }
#endif
    if (Kokkos::is_initialized()) {
      Kokkos::finalize();
    }
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

#if defined(MPI_ENABLED)
  MPI_Finalize();
#endif
  Kokkos::finalize();
  return 0;
}

auto pad(const std::string& str, std::size_t n, char c, bool right) -> std::string {
  if (n <= str.size()) {
    return str;
  }
  if (right) {
    return str + std::string(n - str.size(), c);
  }
  return std::string(n - str.size(), c) + str;
}

#if !defined(MPI_ENABLED)

template <typename Func, typename... Args>
void CallOnce(Func func, Args&&... args) {
  func(std::forward<Args>(args)...);
}

#else

template <typename Func, typename... Args>
void CallOnce(Func func, Args&&... args) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == MPI_ROOT_RANK) {
    func(std::forward<Args>(args)...);
  }
}
#endif

template <typename T>
auto define_constdim_array(adios2::IO&                     io,
                           const std::vector<std::size_t>& glob_shape,
                           const std::vector<std::size_t>& loc_corner,
                           const std::vector<std::size_t>& loc_shape) -> std::string {
  const std::string arrname = "ConstantDimArr" +
                              std::to_string(glob_shape.size()) +
                              "D::" + std::string(typeid(T).name());
  io.DefineVariable<T>(arrname, glob_shape, loc_corner, loc_shape, adios2::ConstantDims);
  return arrname;
}

template <typename T>
auto define_unknowndim_array(adios2::IO& io) -> std::string {
  const std::string arrname = "UnknownDimArr::" + std::string(typeid(T).name());
  io.DefineVariable<T>(arrname,
                       { adios2::UnknownDim },
                       { adios2::UnknownDim },
                       { adios2::UnknownDim });
  return arrname;
}

template <typename T, typename A>
void put_constdim_array(adios2::IO&        io,
                        adios2::Engine&    writer,
                        const A&           array,
                        const std::string& varname) {
  auto var = io.InquireVariable<T>(varname);
  if (!var) {
    throw std::runtime_error("Variable not found: " + varname);
  }
  auto array_h = Kokkos::create_mirror_view(array);
  Kokkos::deep_copy(array_h, array);
  writer.Put<T>(var, array_h);
}

template <typename T>
void put_unknowndim_array(adios2::IO&             io,
                          adios2::Engine&         writer,
                          const Kokkos::View<T*>& array,
                          std::size_t             nelems,
                          const std::string&      varname) {
  auto var = io.InquireVariable<T>(varname);
  if (!var) {
    throw std::runtime_error("Variable not found: " + varname);
  }
  std::size_t glob_nelems   = nelems;
  std::size_t offset_nelems = 0u;
#if defined(MPI_ENABLED)
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<std::size_t> all_nelems(size);
  MPI_Allgather(&nelems,
                1,
                MPI_UNSIGNED_LONG,
                all_nelems.data(),
                1,
                MPI_UNSIGNED_LONG,
                MPI_COMM_WORLD);
  glob_nelems = 0u;
  for (int r = 0; r < size; ++r) {
    if (r < rank) {
      offset_nelems += all_nelems[r];
    }
    glob_nelems += all_nelems[r];
  }
#endif
  var.SetShape({ glob_nelems });
  var.SetSelection(adios2::Box<adios2::Dims>({ offset_nelems }, { nelems }));
  auto array_h = Kokkos::create_mirror_view(array);
  Kokkos::deep_copy(array_h, array);
  writer.Put<T>(var, array_h);
}
