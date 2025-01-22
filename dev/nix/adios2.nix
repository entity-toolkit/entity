{
  pkgs ? import <nixpkgs> { },
  hdf5 ? false,
  mpi ? false,
}:

let
  name = "adios2";
  version = "2.10.2";
in
pkgs.stdenv.mkDerivation {
  pname = "${name}${if hdf5 then "-hdf5" else ""}${if mpi then "-mpi" else ""}";
  version = "${version}";
  src = pkgs.fetchgit {
    url = "https://github.com/ornladios/ADIOS2/";
    rev = "v${version}";
    sha256 = "sha256-NVyw7xoPutXeUS87jjVv1YxJnwNGZAT4QfkBLzvQbwg=";
  };

  nativeBuildInputs =
    with pkgs;
    [
      cmake
      libgcc
      perl
      breakpointHook
    ]
    ++ (if mpi then [ openmpi ] else [ ]);

  buildInputs = if hdf5 then (if mpi then [ pkgs.hdf5-mpi ] else [ pkgs.hdf5 ]) else [ ];

  configurePhase = ''
    cmake -B build $src \
      -D CMAKE_CXX_STANDARD=17 \
      -D CMAKE_CXX_EXTENSIONS=OFF \
      -D CMAKE_POSITION_INDEPENDENT_CODE=TRUE \
      -D BUILD_SHARED_LIBS=ON \
      -D ADIOS2_USE_HDF5=${if hdf5 then "ON" else "OFF"} \
      -D ADIOS2_USE_Python=OFF \
      -D ADIOS2_USE_Fortran=OFF \
      -D ADIOS2_USE_ZeroMQ=OFF \
      -D BUILD_TESTING=OFF \
      -D ADIOS2_BUILD_EXAMPLES=OFF \
      -D ADIOS2_USE_MPI=${if mpi then "ON" else "OFF"} \
      -D ADIOS2_HAVE_HDF5_VOL=OFF \
      -D CMAKE_BUILD_TYPE=Release
  '';

  buildPhase = ''
    cmake --build build -j
  '';

  installPhase = ''
    sed -i '/if(CMAKE_INSTALL_COMPONENT/,/^[[:space:]]&endif()$/d' build/cmake/install/post/cmake_install.cmake
    cmake --install build --prefix $out
    chmod +x build/cmake/install/post/generate-adios2-config.sh
    sh build/cmake/install/post/generate-adios2-config.sh $out
  '';

  enableParallelBuilding = true;
}
