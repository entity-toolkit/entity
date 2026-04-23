#!/usr/bin/env bash

build_dir=""
extra_flags=""

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
	--flags)
		extra_flags="$2"
		shift 2
		;;
	*)
		echo "Unknown option: $1"
		echo "Usage: $0 --build <build_dir> [--flags <extra_cmake_flags>]"
		exit 1
		;;
	esac
done

if [ "${build_dir}" != "" ]; then

	pgens=$(find pgens/ -mindepth 2 -name "pgen.hpp" -exec dirname {} \; | sed 's|^pgens/||' | paste -sd ";" -)
	(
		cmake -B ${build_dir}/${pgen_name} -D pgens="${pgens}" -D TESTS=ON ${extra_flags} &&
			cmake --build ${build_dir}/${pgen_name} -j $(nproc --ignore 8) &&
			ctest --test-dir ${build_dir}
	) || exit 1

else

	echo "Usage: $0 --build <build_dir> [--flags <extra_cmake_flags>]"

fi
