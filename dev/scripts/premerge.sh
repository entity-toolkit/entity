find cmake/ -type f -name "*.cmake" -o -name "*.txt" | xargs cmake-format -i
find src/ -type f -name "*.cmake" -o -name "*.txt" | xargs cmake-format -i
find pgens/ -type f -name "*.cmake" -o -name "*.txt" | xargs cmake-format -i
find minimal/ -type f -name "*.cmake" -o -name "*.txt" | xargs cmake-format -i

find pgens/ -type f -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format --style=file -i
find src/ -type f -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format --style=file -i
find minimal/ -type f -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format --style=file -i
