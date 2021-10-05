.PHONY: clang-all clang-tidy-naming clang-format-fix clang-format clang-tidy clang-tidy-bugprone

SOURCES := $(SRCS) $(NTT_SRCS) $(TEST_SRCS) $(EXAMPLES_SRCS)
ALLCODE := $(SOURCES) $(call rwildcard, ${SRC_DIR}, *.hpp) $(call rwildcard, ${SRC_DIR}, *.h) $(call rwildcard, ${NTT_DIR}, *.hpp) $(call rwildcard, ${NTT_DIR}, *.h) $(call rwildcard, ${TEST_DIR}, *.hpp) $(call rwildcard, ${TEST_DIR}, *.h) $(call rwildcard, ${EXAMPLES_DIR}, *.hpp) $(call rwildcard, ${EXAMPLES_DIR}, *.h)
flags := $(INCFLAGS) $(DEFINITIONS) $(CFLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(INCFLAGS) -include ${PGEN_DIR}/${PGEN}.hpp

clang-all : clang-tidy-naming clang-format clang-tidy

clang-tidy-naming:
	@for src in $(SOURCES) ; do \
		echo "checking namings in $$src:" ;\
		clang-tidy -quiet -checks='-*,readability-identifier-naming' \
		    -config="{CheckOptions: [ \
		    { key: readability-identifier-naming.NamespaceCase, value: lower_case },\
		    { key: readability-identifier-naming.ClassCase, value: CamelCase  },\
		    { key: readability-identifier-naming.StructCase, value: CamelCase  },\
		    { key: readability-identifier-naming.FunctionCase, value: camelBack },\
		    { key: readability-identifier-naming.VariableCase, value: lower_case },\
		    { key: readability-identifier-naming.GlobalConstantCase, value: UPPER_CASE }\
		    ]}" "$$src" -extra-arg=${CXXSTANDARD} -- $(flags);\
	done
	@echo "clang-tidy-naming -- done"

clang-format:
	@for src in $(ALLCODE) ; do \
		var=`clang-format $$src | diff $$src - | wc -l` ; \
		if [ $$var -ne 0 ] ; then \
			diff=`clang-format $$src | diff $$src -` ; \
			echo "$$src:" ; \
			echo "$$diff" ; \
			echo ; \
		fi ; \
	done
	@echo "clang-format -- done"

clang-format-fix:
	@for src in $(ALLCODE) ; do \
		var=`clang-format $$src | diff $$src - | wc -l` ; \
		if [ $$var -ne 0 ] ; then \
			echo "formatting $$src:" ;\
			diff=`clang-format $$src | diff $$src -` ; \
			clang-format -i "$$src" ; \
			echo "$$diff" ; \
			echo ; \
		fi ; \
	done
	@echo "clang-format-fix -- done"

# TODO: get rid of KOKKOS and other files in tidy

clang-tidy:
	@for src in $(SOURCES) ; do \
		echo "tidying $$src:" ; \
		clang-tidy -quiet -checks="-*,\
			clang-diagnostic-*,clang-analyzer-*,modernize-*,-modernize-avoid-c-arrays*,\
			readability-*,performance-*,openmp-*,mpi-*,-performance-no-int-to-ptr" \
			-header-filter=".*\\b(ntt|src|test|examples)\\b\\/(?!lib).*" \
			"$$src" -extra-arg=${CXXSTANDARD} -- $(flags); \
	done
	@echo "clang-tidy -- done"

clang-tidy-bugprone:
	@for src in $(SOURCES) ; do \
		echo "tidying $$src:" ; \
		clang-tidy -quiet -checks="-*,bugprone-*",\
			-header-filter=".*\\b(ntt|src|test|examples)\\b\\/(?!lib).*" \
			"$$src" -extra-arg=${CXXSTANDARD} -- $(flags); \
	done
	@echo "clang-tidy-bugprone -- done"
