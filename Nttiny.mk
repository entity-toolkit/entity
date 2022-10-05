# to compile visualization app
#
COMPILER := ${HOST_CXX}
VIS_TARGET := ${BIN_DIR}/vis.exec
BUILD_VIS_DIR := $(subst ${ROOT_DIR}/,,${VIS_DIR})

VIS_SRC := ${VIS_DIR}/nttiny.cpp
VIS_OBJ := $(subst ${VIS_DIR},${BUILD_VIS_DIR},$(VIS_SRC:%=%.o))

ENV_VAR_FILES := ${BUILD_DIR}/lib/NTTINY_INC_DIRS ${BUILD_DIR}/lib/NTTINY_LIB_DIRS ${BUILD_DIR}/lib/NTTINY_LIBS

nttiny : ${BUILD_DIR}/lib/libnttiny.a $(ENV_VAR_FILES)

vis : pgenCopy nttiny ${VIS_TARGET}

${BUILD_DIR}/lib/libnttiny.a $(ENV_VAR_FILES) : ${NTTINY_DIR}
	@echo ${BUILD_DIR}
	$(HIDE)cd ${NTTINY_DIR} && cmake -B build -D CMAKE_CXX_COMPILER=${HOST_CXX} -D CMAKE_LD_FLAGS="" -D CMAKE_INSTALL_PREFIX=${BUILD_DIR}/lib && cd build && make nttiny -j && make install

nttiny_clean :
	$(HIDE)rm -rf ${BUILD_DIR}/lib/*nttiny* $(ENV_VAR_FILES)
	$(HIDE)cd ${NTTINY_DIR}/build && make clean

${VIS_TARGET} : ${BUILD_DIR}/lib/libnttiny.a $(OBJS) $(VIS_OBJ)
	$(eval NTTINY_LIB_DIRS := $(subst ",,$(subst ;, ,"$(shell cat ${BUILD_DIR}/lib/NTTINY_LIB_DIRS)")) ${BUILD_DIR}/lib)
	$(eval NTTINY_LIBS := $(subst ",,$(subst ;, ,"$(shell cat ${BUILD_DIR}/lib/NTTINY_LIBS)")) nttiny)
	$(eval NTTINY_LIB_FLAGS := $(addprefix -l, $(NTTINY_LIBS)))
	$(eval NTTINY_LD_FLAGS := $(addprefix -L, $(NTTINY_LIB_DIRS)) $(LIBS))
	@echo [L]inking $@ from $^
	$(HIDE)mkdir -p ${BIN_DIR}
	$(HIDE)${link_command} $^ $(NTTINY_LIB_FLAGS) -o $@ $(NTTINY_LD_FLAGS)

${BUILD_VIS_DIR}/%.o : ${VIS_DIR}/%
	$(eval NTTINY_INC_DIRS := $(subst ",,$(subst ;, ,"$(shell cat ${BUILD_DIR}/lib/NTTINY_INC_DIRS)")))
	$(eval NTTINY_INC_FLAGS := $(addprefix -I, $(NTTINY_INC_DIRS) ${NTTINY_DIR}/src ${NTTINY_DIR}/extern $(dir $(wildcard ${NTTINY_DIR}/src/**/))))
	@echo [C]ompiling \`vis\`: $(subst ${ROOT_DIR}/,,$<)
	$(HIDE)mkdir -p $(dir $@)
	$(HIDE)${compile_command} $(NTTINY_INC_FLAGS) -c $^ -o $@
