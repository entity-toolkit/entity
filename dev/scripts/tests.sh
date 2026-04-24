#!/usr/bin/env bash

build_dir=""
extra_flags=""
with_pgens=false
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
	--flags)
		extra_flags="$2"
		shift 2
		;;
	*)
		echo "Unknown option: $1"
		echo "Usage: $0 --build <build_dir> [--flags <extra_cmake_flags>] [--with_pgens] [--with_tests]"
		exit 1
		;;
	esac
done

if [ "${with_pgens}" = false ] && [ "${with_tests}" = false ]; then

	echo "Error: At least one of --with_pgens or --with_tests must be specified"
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
			cmake --build ${build_dir} -j $(nproc)
	) || exit 1

	if [ "${with_tests}" = true ]; then
		ctest --test-dir ${build_dir} --output-on-failure
	fi

fi
