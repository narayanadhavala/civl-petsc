# This makefile looks for all the subdirectories inside the "functions" folder
# (it skips hidden folders and the folder called "CIVLREP").
SUBDIRS := $(shell find functions -mindepth 1 -maxdepth 1 -type d -not -name ".*" -not -name "CIVLREP" -exec basename {} \;)

# The "small" target runs tests in each subfolder.
# It runs the verification tests with vector sizes from 1..3 and processor counts from 1 and 2,
# covering both real and complex cases.
small:
	@for dir in $(SUBDIRS); do \
	  echo "\n\033[1;34m==============================================\033[0m"; \
	  echo "\033[1;34m=== Verifying: $$dir\033[0m"; \
	  echo "\033[1;34m==============================================\033[0m\n"; \
	  $(MAKE) -C functions/$$dir small; \
	done

# The "big" target runs tests in each subfolder.
# It runs the verification tests with vector sizes from 1..5 and processor counts from 1..5,
# covering both real and complex cases.
big:
	@for dir in $(SUBDIRS); do \
	  echo "\n\033[1;34m==============================================\033[0m"; \
	  echo "\033[1;34m=== Verifying: $$dir\033[0m"; \
	  echo "\033[1;34m==============================================\033[0m\n"; \
	  $(MAKE) -C functions/$$dir big; \
	done

# The "clean" target goes into every subfolder and runs its clean target
clean:
	@for d in $(SUBDIRS); do \
	  $(MAKE) -C functions/$$d clean; \
	done; \

.phony: small big clean
