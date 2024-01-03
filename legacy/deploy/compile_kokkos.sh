#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source ${SCRIPT_DIR}/aux/aux.sh

declare -r modulename="Kokkos"
declare -r has_modulefile="ON"

default_debug="OFF"
declare debug="${default_debug}"
default_arch="AUTO"
declare arch="${default_arch}"
default_mpi_path="module:ompi"
declare with_mpi="${default_mpi_path}"

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
  echo "  --debug                        Build in debug mode"
  echo "                                 (default: $debug)"
  echo ""
}

source ${SCRIPT_DIR}/aux/argparse.sh
source ${SCRIPT_DIR}/aux/config.sh

declare -a archs
IFS=',' read -ra archs <<<"$arch"

if [ $debug = "ON" ]; then
  kokkos_module+="/debug"
  install_path+="/debug"
fi

if [ $with_mpi != "OFF" ]; then
  kokkos_module+="/mpi"
  install_path+="/mpi"
fi

if [ $enable_cuda = "ON" ]; then
  kokkos_module+="/cuda"
  install_path+="/cuda"
fi

suffix=$(define_kokkos_suffix $arch)
kokkos_module+=$suffix
install_path+=$suffix

flags=()

if [ $with_mpi = "OFF" ]; then
  flags+=(
    Kokkos_ENABLE_OPENMP=ON
  )
fi

if [ $enable_cuda = "ON" ]; then
  flags+=(
    Kokkos_ENABLE_CUDA=ON
  )
fi

for ar in "${archs[@]}"; do
  if [ $ar = "AUTO" ]; then
    break
  fi
  flags+=(
    Kokkos_ARCH_${ar}=ON
  )
done

if [ $debug = "ON" ]; then
  flags+=(
    Kokkos_ENABLE_DEBUG=ON
    Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON
  )
fi

compile_args=(
  -D CMAKE_CXX_STANDARD=17
  -D CMAKE_CXX_EXTENSIONS=OFF
  -D CMAKE_POSITION_INDEPENDENT_CODE=TRUE
  -D BUILD_SHARED_LIBS=ON
  -D CMAKE_INSTALL_PREFIX=$install_path
)
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
  fi
}

function configure {
  prebuild
  runcommand "cd $kokkos_src_path"
  runcommand "rm -rf build"
  local args=$(printf " %s" "${compile_args[@]}")
  runcommand "cmake -B build$args"
}

function compile {
  prebuild
  runcommand "cd $kokkos_src_path"
  runcommand "cmake --build build -j"
}

function install {
  runcommand "cd $kokkos_src_path"
  runcommand "cmake --install build"
}

function cleanup {
  runcommand "cd $kokkos_src_path"
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
    "Architecture(s)"
    "Debug mode"
  )
  REPORT_VALS+=(
    "${arch}"
    "${debug}"
  )
}

function modulefile {
  fname=$kokkos_module
  description="Kokkos"
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

conflict        kokkos
$prereqs

set             basedir                 $install_path
append-path     PATH                    \$basedir/bin
setenv          Kokkos_DIR              \$basedir
$setflags
    '''
  runcommand "echo -e \"$modulecontent\" >>$fname"
}

run configure compile install cleanup modulefile report
