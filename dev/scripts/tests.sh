#!/usr/bin/env bash

build_dir=""
extra_flags=""
nproc=$(nproc)
with_pgens=false
make_plots=false
with_tests=false

while [[ $# -gt 0 ]]; do
	case "$1" in
	--build)
		if [[ -z "${2:-}" || "$2" == --* ]]; then
			echo "Error: --build requires a value"
			exit 1
		fi
		build_dir="$2"
		shift 2
		;;
	--with_pgens)
		with_pgens=true
		shift
		;;
	--with_tests)
		with_tests=true
		shift
		;;
	--make_plots)
		make_plots=true
		shift
		;;
	--flags)
		extra_flags="$2"
		shift 2
		;;
	--nproc)
		if [[ -z "${2:-}" || "$2" == --* ]]; then
			echo "Error: --nproc requires a value"
			exit 1
		fi
		nproc="$2"
		shift 2
		;;
	*)
		echo "Unknown option: $1"
		echo "Usage: $0 --build <build_dir> [--flags <extra_cmake_flags>] [--with_pgens] [--with_tests] [--make_plots] [--nproc <num_threads>]"
		exit 1
		;;
	esac
done

if [ "${with_pgens}" = false ] && [ "${with_tests}" = false ]; then

	echo "Error: At least one of --with_pgens or --with_tests must be specified"
	exit 1

fi

if [ "${make_plots}" = true ] && [ "${with_pgens}" = false ]; then

	echo "Error: --make_plots requires --with_pgens"
	exit 1

fi

if [ "${build_dir}" != "" ]; then

	pgens=$(find pgens/ -mindepth 2 -name "pgen.hpp" -exec dirname {} \; | sed 's|^pgens/||' | paste -sd ";" -)

	if [ "${with_pgens}" = true ]; then
		extra_flags="${extra_flags} -D pgens=${pgens}"
	fi

	if [ "${with_tests}" = true ]; then
		extra_flags="${extra_flags} -D TESTS=ON"
	fi

	(
		cmake -B ${build_dir} ${extra_flags} &&
			cmake --build ${build_dir} -j ${nproc}
	) || exit 1

	if [ "${with_tests}" = true ]; then
		ctest --test-dir ${build_dir} --output-on-failure
	fi

	if [ "${with_pgens}" = true ]; then
		temp_dir="$(realpath "${build_dir}")/runs"

		if [ "${make_plots}" = true ]; then
			if ! python3 -c "import nt2" &>/dev/null; then
				echo "Error: nt2py python package is not installed. Please install it with 'pip install nt2py' and try again."
				exit 1
			fi
		fi

		for pgen in $(echo ${pgens} | tr ";" "\n"); do
			if [[ "${pgen}" == *"examples/"* ]]; then
				if [[ "${pgen}" == *"tutorial"* ]]; then
					continue
				fi
				if [[ "${pgen}" == *"particle_update"* ]]; then
					continue
				fi
				pgen_alt=$(echo "${pgen}" | sed 's|examples/|examples_|')
				mkdir -p "${temp_dir}/${pgen_alt}"
				cp "${build_dir}/${pgen_alt}/src/entity_${pgen_alt}.xc" "${temp_dir}/${pgen_alt}/"
				find "pgens/${pgen}" -type f -name "*.py" -exec cp {} "${temp_dir}/${pgen_alt}/" \;
				find "pgens/${pgen}" -type f -name "*.toml" -exec cp {} "${temp_dir}/${pgen_alt}/" \;
			fi
		done

		for pgen in $(echo ${pgens} | tr ";" "\n"); do
			if [[ "${pgen}" == *"examples/"* ]]; then
				if [[ "${pgen}" == *"tutorial"* ]]; then
					continue
				fi
				if [[ "${pgen}" == *"particle_update"* ]]; then
					continue
				fi
				pgen_alt=$(echo "${pgen}" | sed 's|examples/|examples_|')
				inputs=$(find "${temp_dir}/${pgen_alt}" -type f -name "*.toml")
				for toml_file in ${inputs}; do
					toml_basename=$(basename "${toml_file}")
					echo "Running pgen: ${pgen} with input ${toml_basename}"
					if [[ "${extra_flags}" == *"mpi=ON"* ]]; then
						(cd "${temp_dir}/${pgen_alt}" && mpiexec -n 2 ./entity_${pgen_alt}.xc -input "${toml_basename}")
					else
						(cd "${temp_dir}/${pgen_alt}" && ./entity_${pgen_alt}.xc -input "${toml_basename}")
					fi
				done
				if [ "${make_plots}" = true ]; then
					python_scripts=$(find "${temp_dir}/${pgen_alt}" -type f -name "*.py")
					for py_file in ${python_scripts}; do
						py_basename=$(basename "${py_file}")
						echo "Running python script: ${py_basename} for pgen ${pgen}"
						(cd "${temp_dir}/${pgen_alt}" && python3 "${py_basename}")
					done
				fi
			fi
		done

		if [ "${make_plots}" = true ]; then
			mkdir -p "${temp_dir}/results"
			find "${temp_dir}" -maxdepth 2 -type f \( -name "*.png" -o -name "*.mp4" \) -exec mv {} "${temp_dir}/results/" \;
		fi
	fi

fi
