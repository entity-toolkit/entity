if command -v cmake-format &>/dev/null; then
	find cmake/ src/ minimal/ -type f -name "*.cmake" -o -name "*.txt" | xargs cmake-format -i
fi

if command -v clang-format &>/dev/null; then
	find pgens/ src/ minimal/ -type f -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format --style=file -i
fi

build_dir="$1"
extra_flags="$2"

if [ "${build_dir}" != "" ]; then

	pgens=$(find pgens/ -mindepth 2 -name "pgen.hpp" -exec dirname {} \; | sed 's|^pgens/||' | paste -sd ";" -)
	(
		cmake -B ${build_dir}/${pgen_name} -D pgens="${pgens}" -D TESTS=ON ${extra_flags} &&
			cmake --build ${build_dir}/${pgen_name} -j $(nproc) &&
			ctest --test-dir ${build_dir}
	) || exit 1

fi
