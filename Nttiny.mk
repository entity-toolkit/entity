VIS_TARGET := ${BIN_DIR}/vis.exec
BUILD_VIS_DIR := $(subst ${ROOT_DIR}/,,${VIS_DIR})

VIS_SRC := $(wildcard ${VIS_DIR}/*.cpp)
VIS_OBJ := $(subst ${VIS_DIR},${BUILD_VIS_DIR},$(VIS_SRC:%=%.o))
-include ${NTTINY_DIR}/Makefile
print :
	@echo ${VIS_SRC}

vis : ${VIS_TARGET}

${VIS_TARGET}: $(OBJS) $(PGEN_OBJS) $(VIS_OBJ)
	@echo [L]inking $@ from $^
	$(HIDE)${link_command} $^ -o $@ $(LIBS)

${BUILD_VIS_DIR}/%.o : ${VIS_DIR}/%
	$(HIDE)mkdir -p $(dir $@)
	$(HIDE)${compile_command} -include ${PGEN_DIR}/${PGEN}.hpp -c $^ -o $@
