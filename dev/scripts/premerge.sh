# code formatting
find cmake/ -type f -name "*.cmake" -o -name "*.txt" | xargs cmake-format -i
find src/ -type f -name "*.cmake" -o -name "*.txt" | xargs cmake-format -i
find pgens/ -type f -name "*.cmake" -o -name "*.txt" | xargs cmake-format -i
find minimal/ -type f -name "*.cmake" -o -name "*.txt" | xargs cmake-format -i

find pgens/ -type f -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format --style=file -i
find src/ -type f -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format --style=file -i
find minimal/ -type f -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format --style=file -i

build_dir="$1"
extra_flags="$2"

if [ "${build_dir}" != "" ]; then

	# compile all problem generators
	find pgens/ -mindepth 2 -name "pgen.hpp" -exec dirname {} \; | sed 's|^pgens/||' | while read pgen; do
		pgen_name=$(basename ${pgen})
		(
			cmake -B ${build_dir}/${pgen_name} -D pgen=${pgen} ${extra_flags} &&
				cmake --build ${build_dir}/${pgen_name} -j $(nproc)
		) || exit 1
	done

	(
		cmake -B ${build_dir}/tests -D TESTS=ON ${extra_flags} &&
			cmake --build ${build_dir}/tests -j $(nproc) &&
			ctest --test-dir ${build_dir}/tests
	) || exit 1
fi
