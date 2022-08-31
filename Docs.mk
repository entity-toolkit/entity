DOCS_DIR := ${ROOT_DIR}/docs

DOXYGENMD_DIR := ${EXT_DIR}/doxygenmd
DOXYGENMD_BIN := ${DOXYGENMD_DIR}/build/bin/doxygenmd

docs_command := ${DOXYGENMD_BIN}

# HPP_HEADERS := $(filter-out ${SRC_DIR}/.temp/problem_generator.hpp, $(shell find ${SRC_DIR} -name "*.hpp"))
# H_HEADERS := $(shell find ${SRC_DIR} -name "*.h")

# HPP_DOCS := $(subst ${SRC_DIR},${DOCS_DIR}, $(HPP_HEADERS:%=%.md))
# H_DOCS := $(subst ${SRC_DIR},${DOCS_DIR}, $(H_HEADERS:%=%.md))

# docs : $(HPP_DOCS) $(H_DOCS)

HEADERS := $(shell find ${SRC_DIR} -name "*.hpp" -o -name "*.h")

HEADERS := $(filter-out ${SRC_DIR}/.temp/problem_generator.hpp, $(HEADERS))
HEADERS := $(filter-out ${SRC_DIR}/defaults.h, $(HEADERS))
HEADERS := $(filter-out ${SRC_DIR}/definitions.h, $(HEADERS))

HEADERS := $(filter-out ${SRC_DIR}/framework/utils/%, $(HEADERS))
HEADERS := $(filter-out ${SRC_DIR}/framework/metrics/utils/%, $(HEADERS))
HEADERS := $(filter-out ${SRC_DIR}/framework/metric.h, $(HEADERS))
HEADERS := $(filter-out ${SRC_DIR}/framework/io/%, $(HEADERS))

HEADERS := $(filter-out ${SRC_DIR}/pic/pic_filter_currents.hpp, $(HEADERS))

DOCS := $(subst ${SRC_DIR},${DOCS_DIR}, $(HEADERS:%=%.md))

docs: doxygenmd $(DOCS)

doxygenmd: ${DOXYGENMD_BIN}

${DOXYGENMD_BIN}:
	@echo [C]ompiling doxygenmd
	$(HIDE)cd ${DOXYGENMD_DIR} && mkdir -p build && cd build && cmake .. && make -s -j

${DOCS_DIR}/%.md: ${SRC_DIR}/%
	@echo [D]ocumenting \`$(subst ${ROOT_DIR}/,,$<)\`
	$(HIDE)mkdir -p $(dir $@)
	$(HIDE)${docs_command} $< ${subst .h,,${subst .hpp,,$@}}

print:
	@echo $(DOCS)

clean_docs:
	@echo [C]leaning docs
	$(HIDE)rm -rf $(DOCS_DIR)