#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

source ${SCRIPT_DIR}/aux/aux.sh

declare -r modulename="ADIOS2"
declare -r has_modulefile="ON"

default_debug="OFF"
declare debug="${default_debug}"
default_arch="AUTO"
declare arch="${default_arch}"
default_mpi_path="module:ompi"
declare with_mpi="${default_mpi_path}"
default_hdf5_path="module:hdf5"
declare hdf5="${default_hdf5_path}"
default_kokkos_path="module:kokkos"
declare kokkos="${default_kokkos_path}"

source ${SCRIPT_DIR}/aux/default.sh
source ${SCRIPT_DIR}/aux/globals.sh

function usage {
  common_help
  echo "  --with-mpi <path|module>       MPI path or modulename (\`module:<modulename>\`)"
  echo "                                 set to OFF to disable MPI support"
  echo "                                 (default: ${default_mpi_path})"
  echo ""
  echo "  --arch <string>                Hardware architecture for Kokkos"
  echo "                                 comma-separated list of CPU and/or GPU archs"
  echo "                                 for example: \`SKX,Volta70\`"
  echo "                                 (default: ${default_arch})"
  echo ""
  echo "  --kokkos <path|module>         Kokkos path or modulename (\`module:<modulename>\`)"
  echo "                                 default: ${default_kokkos_path}/<automatic>"
  echo ""
  echo "  --hdf5 <path|module>           HDF5 path or modulename (\`module:<modulename>\`)"
  echo "                                 default: ${default_hdf5_path}/<automatic>"
  echo ""
  echo "  --debug                        Build in debug mode"
  echo "                                 (default: $debug)"
  echo ""
}

source ${SCRIPT_DIR}/aux/argparse.sh
source ${SCRIPT_DIR}/aux/config.sh

if [[ $hdf5 = module:* ]]; then
  hdf5_module=${hdf5#module:}
fi

if [[ $kokkos = module:* ]]; then
  kokkos_module=${kokkos#module:}
fi

if [ $arch = "AUTO" ]; then
  printf "${RED}Automatic architecture detection is not supported for ADIOS2${NC}\n"
  exit 1
fi

if [ $debug = "ON" ]; then
  adios2_module+="/debug"
  install_path+="/debug"
  kokkos_module+="/debug"
fi

if [ $with_mpi != "OFF" ]; then
  adios2_module+="/mpi"
  install_path+="/mpi"
  if [ $hdf5 == $default_hdf5_path ]; then
    hdf5_module+="/mpi"
    if [ $enable_cuda = "ON" ]; then
      hdf5_module+="/cuda"
    else
      hdf5_module+="/cpu"
    fi
  fi
  kokkos_module+="/mpi"
else
  if [ $hdf5 == $default_hdf5_path ]; then
    hdf5_module+="/serial"
  fi
fi

if [ $enable_cuda = "ON" ]; then
  adios2_module+="/cuda"
  install_path+="/cuda"
  kokkos_module+="/cuda"
fi

suffix=$(define_kokkos_suffix $arch)
adios2_module+=$suffix
install_path+=$suffix
kokkos_module+=$suffix

flags=()

if [ $enable_cuda = "ON" ]; then
  flags+=(
    ADIOS2_USE_CUDA=ON
  )
fi

if [ $with_mpi != "OFF" ]; then
  flags+=(
    ADIOS2_USE_MPI=ON
  )
else
  flags+=(
    ADIOS2_USE_MPI=OFF
    ADIOS2_HAVE_HDF5_VOL=OFF
  )
fi

compile_args=(
  -D CMAKE_CXX_STANDARD=17
  -D CMAKE_CXX_EXTENSIONS=OFF
  -D CMAKE_POSITION_INDEPENDENT_CODE=TRUE
  -D BUILD_SHARED_LIBS=ON

  -D ADIOS2_USE_HDF5=ON
  -D ADIOS2_USE_Kokkos=ON

  -D ADIOS2_USE_Python=OFF
  -D ADIOS2_USE_Fortran=OFF
  -D ADIOS2_USE_ZeroMQ=OFF
  -D BUILD_TESTING=OFF
  -D ADIOS2_BUILD_EXAMPLES=OFF

  -D CMAKE_INSTALL_PREFIX=$install_path
)

if [ $debug = "ON" ]; then
  compile_args+=(
    -D CMAKE_BUILD_TYPE=Debug
  )
fi
for flag in "${flags[@]}"; do
  compile_args+=(
    -D $flag
  )
done

source ${SCRIPT_DIR}/aux/run.sh

function prebuild {
  if [ $use_modules = "ON" ]; then
    runcommand "module purge"
    runcommand "module load $cc_module"
    if [ $enable_cuda = "ON" ]; then
      runcommand "module load $cuda_module"
    fi
    if [ ! $with_mpi = "OFF" ]; then
      runcommand "module load $mpi_module"
    fi
    runcommand "module load $hdf5_module"
    runcommand "module load $kokkos_module"
  fi
}

function configure {
  prebuild
  runcommand "cd $adios2_src_path"
  runcommand "rm -rf build"
  local args=$(printf " %s" "${compile_args[@]}")
  runcommand "cmake -B build$args"
}

function compile {
  prebuild
  runcommand "cd $adios2_src_path"
  runcommand "cmake --build build -j"
}

function install {
  runcommand "cd $adios2_src_path"
  runcommand "cmake --install build"
}

function cleanup {
  runcommand "cd $adios2_src_path"
  runcommand "rm -rf build"
}

function report {
  if [ ! $with_mpi = "OFF" ]; then
    REPORT_VARS+=(
      "MPI"
    )
    REPORT_VALS+=(
      "${with_mpi}"
    )
  fi
  REPORT_VARS+=(
    "Kokkos"
    "HDF5"
  )

  REPORT_VALS+=(
    "${kokkos}"
    "${hdf5}"
  )
  REPORT_VARS+=(
    "Architecture(s)"
    "Debug mode"
  )
  REPORT_VALS+=(
    "${arch}"
    "${debug}"
  )
}

function modulefile {
  fname=$adios2_module
  description="ADIOS2"
  if [ $with_mpi != "OFF" ]; then
    description=$description" @ MPI"
  fi
  if [ $enable_cuda = "ON" ]; then
    description=$description" @ CUDA"
  fi
  for ar in "${archs[@]}"; do
    if [ $ar = "AUTO" ]; then
      break
    fi
    description=$description" @ ${ar}"
  done
  prereqs=""
  if [ $use_modules = "ON" ]; then
    prereqs+="prereq\t\t$cc_module"
    if [ $enable_cuda = "ON" ]; then
      prereqs+=" $cuda_module"
    fi
    if [ ! $with_mpi = "OFF" ]; then
      prereqs+=" $mpi_module"
    fi
    prereqs+=" $hdf5_module $kokkos_module"
  fi
  local setflags=""
  for flag in "${flags[@]}"; do
    local setflag=$(echo $flag | sed 's/=/\t\t/')
    setflags+="\nsetenv\t$setflag"
  done
  runcommand "mkdir -p $(dirname $fname)"
  runcommand "rm -f $fname"
  runcommand "echo \"Writing modulefile to $fname\""
  modulecontent='''#%Module1.0######################################################################
##
## $description
##
proc ModulesHelp { } {
    puts stderr \t\"$description\"\n
}
module-whatis   \"$description\"

conflict        adios2
$prereqs

set             basedir         $install_path
append-path     PATH            \$basedir/bin
setenv          adios2_DIR      \$basedir
$setflags
    '''
  runcommand "echo -e \"$modulecontent\" >>$fname"
}

run configure compile install cleanup modulefile report
