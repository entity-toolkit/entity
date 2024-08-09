while [ $# -gt 0 ]; do
  if [[ $1 == "--help" ]] || [[ $1 == "-h" ]]; then
    usage
    exit 0
  elif [[ $1 == "--"* ]]; then
    v="${1/--/}"
    v=$(echo $v | sed 's/-/_/g')
    if [[ -z "$2" ]] || [[ "$2" == --* ]] || [[ "$2" == -* ]]; then
      if [[ "$1" != "--deploy" ]] && [[ "$1" != "--verbose" ]]; then
        printf "\n${RED}Invalid option: $1${NC}\n"
        exit 1
      fi
      declare "$v"="ON"
    else
      declare "$v"="$2"
      shift
    fi
  elif [[ $1 == "-"* ]]; then
    if [[ -z "$2" ]] || [[ "$2" == --* ]] || [[ "$2" == -* ]]; then
      if [[ $1 == "-v" ]]; then
        verbose="ON"
      elif [[ $1 == "-d" ]]; then
        deploy="ON"
      else
        printf "\n${RED}Invalid option: $1${NC}\n"
        exit 1
      fi
    else
      printf "\n${RED}Invalid option: $1${NC}\n"
      exit 1
    fi
  fi
  shift
done

# manage invalid options
if [ $with_cuda = "ON" ]; then
  printf "\n${RED}Please specify CUDA path or modulename${NC}\n"
  exit 1
fi

if [ $with_mpi = "ON" ]; then
  printf "\n${RED}Please specify MPI path or modulename${NC}\n"
  exit 1
fi