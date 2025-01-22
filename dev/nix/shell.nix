{
  pkgs ? import <nixpkgs> { },
  mpi ? false,
  hdf5 ? false,
}:

let
  name = "entity-dev";
  compilerPkg = pkgs.gcc13;
  compilerCXX = "g++";
  compilerCC = "gcc";
  adios2Pkg = (pkgs.callPackage ./adios2.nix { inherit pkgs mpi hdf5; });
in
pkgs.mkShell {
  name = "${name}-env";
  nativeBuildInputs =
    with pkgs;
    [
      zlib
      cmake

      compilerPkg

      clang-tools

      adios2Pkg
      python312
      python312Packages.jupyter

      cmake-format
      neocmakelsp
      black
      pyright
      taplo
      vscode-langservers-extracted
    ]
    ++ (if mpi then [ pkgs.openmpi ] else [ ])
    ++ (if hdf5 then (if mpi then [ pkgs.hdf5-mpi ] else [ pkgs.hdf5 ]) else [ ]);

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath ([
    pkgs.clang19Stdenv.cc.cc
    pkgs.zlib
  ]);

  shellHook = ''
    BLUE='\033[0;34m'
    NC='\033[0m'
    export CC=$(which ${compilerCC})
    export CXX=$(which ${compilerCXX})
    export CMAKE_CXX_COMPILER=$(which ${compilerCXX})
    export CMAKE_C_COMPILER=$(which ${compilerCC})

    echo ""
    echo -e "${name} nix-shell activated: ''\${BLUE}$(which ${compilerCXX})''\${NC}"
  '';
}
