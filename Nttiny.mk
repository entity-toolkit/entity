# to compile visualization app
#
COMPILER := ${CXX}
VIS_TARGET := ${BIN_DIR}/vis.exec
BUILD_VIS_DIR := $(subst ${ROOT_DIR}/,,${VIS_DIR})

VIS_SRC := $(subst ntt,nttiny,${VIS_DIR}/${PGEN}.cpp)
VIS_OBJ := $(subst ${VIS_DIR},${BUILD_VIS_DIR},$(VIS_SRC:%=%.o))
-include ${NTTINY_DIR}/Makefile

vis : nttiny_static ${VIS_TARGET}

${VIS_TARGET}: ${NTTINY_DIR}/build/libnttiny.a $(OBJS) $(PGEN_OBJS) $(VIS_OBJ)
	@echo [L]inking $@ from $^
	$(HIDE)mkdir -p ${BIN_DIR}
	$(HIDE)${link_command} $^ $(NTTINY_LIBS) -o $@ $(NTTINY_LINKFLAGS) $(LIBS)

${BUILD_VIS_DIR}/%.o : ${VIS_DIR}/%
	@echo [C]ompiling \`vis\`: $(subst ${ROOT_DIR}/,,$<)
	$(HIDE)mkdir -p $(dir $@)
	$(HIDE)${compile_command} $(NTTINY_INCFLAGS) -include ${PGEN_DIR}/${PGEN}.hpp -c $^ -o $@
