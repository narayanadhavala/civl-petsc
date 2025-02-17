###################################################################################################
# Super Makefile: Mapping from function names (directory names) to extra inputs.
#
# All functions use:        -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=2
# VecGetValues:              All functions inputs, -inputB=2
# VecConcatenate:            All functions inputs, -inputB=2
# *_Seq:                    -input_mpi_nprocs=1 -inputN_MIN=2 -inputN_MAX=2
# VecSetValues:              All functions inputs, -inputB=2
###################################################################################################

# Find all function directories (exclude hidden directories and CIVLREP)
SUBDIRS := $(shell find functions -mindepth 1 -maxdepth 1 -type d -not -name ".*" -not -name "CIVLREP" -exec basename {} \;)

# List of subdirectories for demonstration.
# SUBDIRS := VecNorm_Seq VecWAXPY

# All functions inputs (default for everything unless overridden)
COMMON_INPUTS := -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=2

# Specialized inputs for specific functions
VecGetValues    := $(COMMON_INPUTS) -inputB=2
VecConcatenate  := $(COMMON_INPUTS) -inputB=2
VecSetValues    := $(COMMON_INPUTS) -inputB=2

# Specialized inputs for anything ending in "_Seq"
_Seq            := -input_mpi_nprocs=1 -inputN_MIN=2 -inputN_MAX=2

# Define a lookup function that returns the right extra inputs.
# 1) If the directory name ends with "_Seq", return _Seq inputs.
# 2) Else if it's VecGetValues / VecConcatenate / VecSetValues, return those.
# 3) Otherwise, just return the common inputs.
define get_super_inputs
$(if $(findstring _Seq,$(1)),$(_Seq), \
  $(if $(filter VecGetValues,$(1)),$(VecGetValues), \
    $(if $(filter VecConcatenate,$(1)),$(VecConcatenate), \
      $(if $(filter VecSetValues,$(1)),$(VecSetValues), \
        $(COMMON_INPUTS)))))
endef

.PHONY: all clean

all:
	@for dir in $(SUBDIRS); do \
	  extra_inputs="$(call get_super_inputs,$$dir)"; \
	  echo "\n\033[1;34m==============================================\033[0m"; \
	  echo "\033[1;34m=== Processing directory: $$dir\033[0m"; \
	  echo "\033[1;34m=== Using extra inputs: $$extra_inputs\033[0m"; \
	  echo "\033[1;34m==============================================\033[0m\n"; \
	  $(MAKE) -C functions/$$dir all EXTRA_INPUTS="$$extra_inputs" || exit 1; \
	done

clean:
	@for d in $(SUBDIRS); do \
	  $(MAKE) -C functions/$$d clean; \
	done; \
	rm -f test_results.log