TEST_DIR := ${ROOT_DIR}/tests
BUILD_TEST_DIR := ${ROOT_DIR}/${BUILD_DIR}/tests/

.PHONY: test test_pic test_grpic

TEST_CFLAGS := $(filter-out -DSIMTYPE=%, $(filter-out -DMETRIC=%, $(CFLAGS)))
test_compile_command := ${CXX} ${CXXSTANDARD} $(INCFLAGS) $(DEFINITIONS) $(TEST_CFLAGS) -MMD

MINKOWSKI_FLAG := -DMETRIC=MINKOWSKI_METRIC
SPHERICAL_FLAG := -DMETRIC=SPHERICAL_METRIC
QSPHERICAL_FLAG := -DMETRIC=QSPHERICAL_METRIC
PIC_FLAG := -DSIMTYPE=PIC_SIMTYPE

TEST_MINK_PIC_OBJS := $(subst ${SRC_DIR},${BUILD_TEST_DIR}PIC_mnk,$(SRCS:%=%.o))
TEST_SPH_PIC_OBJS := $(subst ${SRC_DIR},${BUILD_TEST_DIR}PIC_sph,$(SRCS:%=%.o))
TEST_QSPH_PIC_OBJS := $(subst ${SRC_DIR},${BUILD_TEST_DIR}PIC_qsph,$(SRCS:%=%.o))

TEST_DEPS := ${BUILD_TEST_DIR}PIC_mnk/test.d

test_instance_depends := pgenCopy $(KOKKOS_LINK_DEPENDS) 

test: test_pic

test_pic: test_pic_minkowski test_pic_qspherical

test_pic_minkowski: $(test_instance_depends) ${BIN_DIR}/testPICMink.exec

test_pic_qspherical: $(test_instance_depends) ${BIN_DIR}/testPICQsph.exec

# test_sph_pic test_qsph_pic

# test_grpic: test_ks_grpic test_qks_grpic

# ---------------------------------------------------------------------------- #
#                                 PIC Minkowski                                #
# ---------------------------------------------------------------------------- #
${BIN_DIR}/testPICMink.exec: $(TEST_MINK_PIC_OBJS) ${BUILD_TEST_DIR}PIC_mnk/test.o
	@echo [L]inking $(notdir $@) from $<
	$(HIDE)mkdir -p ${BIN_DIR}
	$(HIDE)${link_command} $^ -o $@ $(LIBS)

${BUILD_TEST_DIR}PIC_mnk/%.o: ${SRC_DIR}/%
	@echo [C]ompiling \`src\` for \`PIC Mink\` test: $(subst ${ROOT_DIR}/,,$<)
	$(HIDE)mkdir -p $(dir $@)
	$(HIDE)${test_compile_command} ${MINKOWSKI_FLAG} ${PIC_FLAG} -c $< -o $@

${BUILD_TEST_DIR}PIC_mnk/test.o: ${TEST_DIR}/test.cpp
	@echo [C]ompiling \`test\` for \`PIC Mink\` test: $(subst ${ROOT_DIR}/,,$<)
	$(HIDE)mkdir -p $(dir $@)
	$(HIDE)${test_compile_command} ${MINKOWSKI_FLAG} ${PIC_FLAG} -c $< -o $@

# ---------------------------------------------------------------------------- #
#                                 PIC Spherical                                #
# ---------------------------------------------------------------------------- #
${BUILD_TEST_DIR}PIC_sph/%.o: ${SRC_DIR}/%
	@echo [C]ompiling \`src\` for \`PIC Sph\` test: $(subst ${ROOT_DIR}/,,$<)
	$(HIDE)mkdir -p $(dir $@)
	$(HIDE)${test_compile_command} ${SPHERICAL_FLAG} ${PIC_FLAG} -c $< -o $@

# ---------------------------------------------------------------------------- #
#                                PIC QSpherical                                #
# ---------------------------------------------------------------------------- #
${BIN_DIR}/testPICQsph.exec: $(TEST_QSPH_PIC_OBJS) ${BUILD_TEST_DIR}PIC_mnk/test.o
	@echo [L]inking $(notdir $@) from $<
	$(HIDE)mkdir -p ${BIN_DIR}
	$(HIDE)${link_command} $^ -o $@ $(LIBS)

${BUILD_TEST_DIR}PIC_qsph/%.o: ${SRC_DIR}/%
	@echo [C]ompiling \`src\` for \`PIC Qsph\` test: $(subst ${ROOT_DIR}/,,$<)
	$(HIDE)mkdir -p $(dir $@)
	$(HIDE)${test_compile_command} ${QSPHERICAL_FLAG} ${PIC_FLAG} -c $< -o $@

${BUILD_TEST_DIR}PIC_qsph/test.o: ${TEST_DIR}/test.cpp
	@echo [C]ompiling \`test\` for \`PIC Qsph\` test: $(subst ${ROOT_DIR}/,,$<)
	$(HIDE)mkdir -p $(dir $@)
	$(HIDE)${test_compile_command} ${QSPHERICAL_FLAG} ${PIC_FLAG} -c $< -o $@

-include $(TEST_DEPS)

export BUILD_TEST_DIR
