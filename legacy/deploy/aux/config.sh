declare -x writing_modulefile="ON"
declare -x use_modules="OFF"
declare -x enable_cuda="OFF"
declare -x install_path=${install_prefix}/${modulename_lower}

if [ -z $has_modulefile ]; then
  writing_modulefile="OFF"
else
  modnm=$(eval echo "\${${modulename_lower}_module}")
  if [[ $has_modulefile = "OFF" || $modnm = "OFF" ]]; then
    writing_modulefile="OFF"
  fi
fi

if [ ! $with_cuda = "OFF" ]; then
  enable_cuda="ON"
  if [[ $with_cuda == module:* ]]; then
    cuda_module=$(echo $with_cuda | cut -d':' -f2)
    use_modules="ON"
  else
    cuda_path=$with_cuda
  fi
fi

if [[ $with_cc == module:* ]]; then
  cc_module=$(echo $with_cc | cut -d':' -f2)
  use_modules="ON"
fi

if [ ! $with_mpi = "OFF" ]; then
  if [[ $with_mpi == module:* ]]; then
    mpi_module=$(echo $with_mpi | cut -d':' -f2)
    use_modules="ON"
    if [ ! $with_cuda = "OFF" ]; then
      mpi_module=$mpi_module/cuda
    else
      mpi_module=$mpi_module/cpu
    fi
  else
    mpi_path=$with_mpi
  fi
fi

function define_kokkos_suffix {
  local arch_raw=$1
  local suffix=""
  declare -xa archs
  IFS=',' read -ra archs <<<"$arch_raw"

  for ar in "${archs[@]}"; do
    if [ $ar = "AUTO" ]; then
      break
    fi
    is_gpu=$(is_gpu_arch $ar)
    if [ $is_gpu = "TRUE" ]; then
      if [ $enable_cuda != "ON" ]; then
        printf "\n${RED}GPU architecture $ar is specified but CUDA is not enabled${NC}\n"
        exit 1
      fi
      suffix+="/${ar,,}"
    fi
  done
  for ar in "${archs[@]}"; do
    if [ $ar = "AUTO" ]; then
      break
    fi
    is_gpu=$(is_gpu_arch $ar)
    if [ $is_gpu != "TRUE" ]; then
      suffix+="/${ar,,}"
    fi
  done
  echo $suffix
}
