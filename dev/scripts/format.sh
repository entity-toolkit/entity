if command -v cmake-format &>/dev/null; then
	find cmake/ src/ minimal/ tests/ -type f -name "*.cmake" -o -name "*.txt" | xargs cmake-format -i
fi

if command -v clang-format &>/dev/null; then
	find pgens/ src/ minimal/ tests/ -type f -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format --style=file -i
fi