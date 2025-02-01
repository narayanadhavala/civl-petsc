# Dynamically gather subdirectories in the "functions" folder.
# The 'find' command returns only immediate subdirectories,
# excluding those that begin with a dot or named "CIVLREP".
SUBDIRS := $(shell find functions -mindepth 1 -maxdepth 1 -type d \
             -not -name ".*" -not -name "CIVLREP" -exec basename {} \;)

.PHONY: all verify clean

# Default target: run verification of all function directories.
all: verify

verify:
	@echo "==============================="
	@echo " Verifying civl-petsc functions"
	@echo "==============================="
	@rm -f test_results.log
	@for d in $(SUBDIRS); do \
	    echo "==== Verifying $$d ===="; \
	    (cd functions/$$d && $(MAKE) all > /dev/null 2>&1); \
	    if [ $$? -eq 0 ]; then \
	        echo "$$d : TEST SUCCESS"; \
	        echo "$$d : SUCCESS" >> test_results.log; \
	    else \
	        echo "$$d : TEST FAIL"; \
	        echo "$$d : FAIL" >> test_results.log; \
	    fi; \
	done
	@echo ""
	@echo "===== Summary ====="
	@echo "Detailed log saved to test_results.log"

clean:
	@for d in $(SUBDIRS); do \
	    $(MAKE) -C functions/$$d clean; \
	done
	@rm -f test_results.log

