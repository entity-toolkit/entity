# devenv counterpart of `shell.nix`; run from this directory:
#   devenv shell                                            # cpu-only
#   devenv shell -P cuda -O entity.arch:string AMPERE80      # cuda
#   devenv shell -P hip -O entity.arch:string AMD_GFX90A     # hip
#   devenv shell -P mpi -P hdf5                              # adios2 with mpi + hdf5
# persistent settings can be put into `devenv.local.nix` (gitignored).
{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

let
  cfg = config.entity;

  gpu = lib.toUpper cfg.gpu;
  arch = lib.toUpper cfg.arch;

  # `shell.nix` imports nixpkgs with `allowUnfree`/`cudaSupport` decided by the
  # requested backend. devenv instantiates its own `pkgs` before this module is
  # evaluated, so it cannot be reconfigured from here -- import the same input
  # ourselves and build everything from that instance.
  nixpkgs = import inputs.nixpkgs {
    inherit (pkgs.stdenv.hostPlatform) system;
    config = {
      allowUnfree = true;
      cudaSupport = gpu == "CUDA";
    };
  };

  adios2Pkg = nixpkgs.callPackage ./adios2.nix {
    pkgs = nixpkgs;
    inherit (cfg) hdf5 mpi;
  };

  kokkosPkg = nixpkgs.callPackage ./kokkos.nix {
    pkgs = nixpkgs;
    stdenv = nixpkgs.stdenv;
    inherit arch gpu;
  };

  extraPkgs = map (name: nixpkgs.${name}) (lib.filter (s: s != "") (lib.splitString "," cfg.extra));

  # compilers are picked by the backend; CUDA goes through kokkos' nvcc_wrapper
  compilerEnv =
    {
      NONE = {
        CXX = "g++";
        CC = "gcc";
      };
      HIP = {
        CXX = "clang++";
        CC = "clang";
      };
      CUDA = { };
    }
    .${gpu};
in
{
  options.entity = {
    gpu = lib.mkOption {
      # case-insensitive, as in `shell.nix`
      type = lib.types.enum [
        "NONE"
        "none"
        "CUDA"
        "cuda"
        "HIP"
        "hip"
      ];
      default = "NONE";
      description = "GPU backend to build Kokkos with.";
    };

    arch = lib.mkOption {
      type = lib.types.str;
      default = "NATIVE";
      example = "AMPERE80";
      description = ''
        Kokkos architecture; mandatory when `gpu` is not `NONE`. See
        https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#gpu-architectures
      '';
    };

    hdf5 = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Build ADIOS2 with HDF5 support.";
    };

    mpi = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Build ADIOS2 with MPI support.";
    };

    extra = lib.mkOption {
      type = lib.types.str;
      default = "";
      example = "gdb,valgrind";
      description = ''
        Comma-separated nixpkgs attributes to add to the environment, kept for
        parity with `shell.nix`. `-O packages:pkgs "gdb valgrind"` does the same
        without going through this option.
      '';
    };
  };

  config = {
    name =
      "nt2" + (if gpu != "NONE" then "-${lib.toLower gpu}" else "") + (if cfg.mpi then "-mpi" else "");

    profiles = {
      cuda.module = {
        entity.gpu = "CUDA";
      };
      hip.module = {
        entity.gpu = "HIP";
      };
      mpi.module = {
        entity.mpi = true;
      };
      hdf5.module = {
        entity.hdf5 = true;
      };
    };

    packages =
      (with nixpkgs; [
        zlib
        cmake

        adios2Pkg
        kokkosPkg

        python314

        cmake-format
        cmake-lint
        neocmakelsp
        black
        pyright
        taplo
        vscode-langservers-extracted
      ])
      ++ extraPkgs;

    env = compilerEnv // {
      LD_LIBRARY_PATH = lib.makeLibraryPath [
        nixpkgs.stdenv.cc.cc
        nixpkgs.zlib
      ];
    };

    enterShell = ''
      BLUE='\033[0;34m'
      NC='\033[0m'

      echo "following environment variables are set:"
    ''
    + lib.concatStringsSep "" (
      lib.mapAttrsToList (name: value: ''
        echo -e "  ''${BLUE}${name}''${NC}=${value}"
      '') compilerEnv
    )
    + ''
      echo ""
      echo -e "${config.name} devenv activated"
    '';
  };
}
