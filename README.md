# CIVL-PETSc Verification Project

This repository is being used to explore the application of CIVL to PETSc (Portable, Extensible Toolkit for Scientific Computation). The structure may change frequently as the project evolves.

## Current Structure
```
.
├── common.mk
├── examples
│   ├── ex11.c
│   ├── ex1.c
│   └── Makefile
├── functions
│   ├── VecAXPBY
│   │   ├── Makefile
│   │   ├── VecAXPBY.c
│   │   ├── VecAXPBY_driver.cvl
│   │   └── VecAXPBY_test.cvl
│   ├── VecAXPBYPCZ
│   │   ├── Makefile
│   │   ├── VecAXPBYPCZ.c
│   │   ├── VecAXPBYPCZ_driver.cvl
│   │   └── VecAXPBYPCZ_test.cvl
│   ├── VecAXPY
│   │   ├── Makefile
│   │   ├── VecAXPY.c
│   │   ├── VecAXPY_driver.cvl
│   │   └── VecAXPY_test.cvl
│   ├── VecAYPX
│   │   ├── Makefile
│   │   ├── VecAYPX.c
│   │   ├── VecAYPX_driver.cvl
│   │   └── VecAYPX_test.cvl
│   ├── VecConjugate_Seq
│   │   ├── Makefile
│   │   ├── VecConjugate_Seq.c
│   │   ├── VecConjugate_Seq_driver.cvl
│   │   └── VecConjugate_Seq_test.c
│   ├── VecCopy
│   │   ├── Makefile
│   │   ├── VecCopy.c
│   │   ├── VecCopy_driver.cvl
│   │   └── VecCopy_test.cvl
│   ├── VecCopy_Seq
│   │   ├── Makefile
│   │   ├── VecCopy_Seq.c
│   │   ├── VecCopy_Seq_driver.cvl
│   │   └── VecCopy_Seq_test.c
│   ├── VecDot
│   │   ├── Makefile
│   │   ├── VecDot.c
│   │   ├── VecDot_driver.cvl
│   │   └── VecDot_test.cvl
│   ├── VecDotRealPart
│   │   ├── Makefile
│   │   ├── VecDotRealPart.c
│   │   ├── VecDotRealPart_driver.cvl
│   │   └── VecDotRealPart_test.cvl
│   ├── VecGetValues
│   │   ├── Makefile
│   │   ├── VecGetValues.c
│   │   ├── VecGetValues_driver.cvl
│   │   └── VecGetValues_test.cvl
│   ├── VecMax
│   │   ├── Makefile
│   │   ├── VecMax.c
│   │   ├── VecMax_driver.cvl
│   │   └── VecMax_test.cvl
│   ├── VecMAXPBY
│   │   ├── Makefile
│   │   ├── VecMAXPBY.c
│   │   ├── VecMAXPBY_driver.cvl
│   │   └── VecMAXPBY_test.cvl
│   ├── VecMaxPointwiseDivide
│   │   ├── Makefile
│   │   ├── VecMaxPointwiseDivide.c
│   │   ├── VecMaxPointwiseDivide_driver.cvl
│   │   └── VecMaxPointwiseDivide_test.cvl
│   ├── VecMAXPY
│   │   ├── Makefile
│   │   ├── VecMAXPY.c
│   │   ├── VecMAXPY_driver.cvl
│   │   └── VecMAXPY_test.cvl
│   ├── VecMin
│   │   ├── Makefile
│   │   ├── VecMin.c
│   │   ├── VecMin_driver.cvl
│   │   └── VecMin_test.cvl
│   ├── VecMTDot
│   │   ├── Makefile
│   │   ├── VecMTDot.c
│   │   ├── VecMTDot_driver.cvl
│   │   └── VecMTDot_test.cvl
│   ├── VecNorm
│   │   ├── Makefile
│   │   ├── output.txt
│   │   ├── VecNorm.c
│   │   └── VecNorm_driver.cvl
│   ├── VecNormalize
│   │   ├── Makefile
│   │   ├── VecNormalize.c
│   │   ├── VecNormalize_driver.cvl
│   │   └── VecNormalize_test.cvl
│   ├── VecNormAvailable
│   │   ├── Makefile
│   │   ├── VecNormAvailable.c
│   │   ├── VecNormAvailable_driver.cvl
│   │   └── VecNormAvailable_test.c
│   ├── VecNorm_Seq
│   │   ├── Makefile
│   │   ├── VecNorm_Seq.c
│   │   ├── VecNorm_Seq_driver.cvl
│   │   └── VecNorm_Seq_test.c
│   ├── VecScale
│   │   ├── Makefile
│   │   ├── VecScale.c
│   │   ├── VecScale_driver.cvl
│   │   └── VecScale_test.cvl
│   ├── VecSet
│   │   ├── Makefile
│   │   ├── VecSet.c
│   │   ├── VecSet_driver.cvl
│   │   └── VecSet_test.cvl
│   ├── VecSetValues
│   │   ├── Makefile
│   │   ├── VecSetValues.c
│   │   ├── VecSetValues_driver.cvl
│   │   └── VecSetValues_test.cvl
│   ├── VecSetValuesBlocked
│   │   ├── Makefile
│   │   ├── VecSetValuesBlocked.c
│   │   ├── VecSetValuesBlocked_driver.cvl
│   │   └── VecSetValuesBlocked_test.cvl
│   ├── VecTDot
│   │   ├── Makefile
│   │   ├── VecTDot.c
│   │   ├── VecTDot_driver.cvl
│   │   └── VecTDot_test.cvl
│   └── VecWAXPY
│       ├── Makefile
│       ├── VecWAXPY.c
│       ├── VecWAXPY_driver.cvl
│       └── VecWAXPY_test.cvl
├── Makefile
├── README.md
└── scaffolding
    ├── include
    │   ├── civlcomplex.cvh
    │   ├── civlvec.cvh
    │   ├── matrix.cvh
    │   ├── petscvec.h
    │   └── scalars.cvh
    ├── src
    │   └── vec
    │       ├── civlcomplex.cvl
    │       ├── civlvec.cvl
    │       ├── Makefile
    │       └── petscvec.c
    └── test
        ├── civlcomplex_test.cvl
        ├── civlvec_test.cvl
        ├── equals.c
        ├── Makefile
        └── petscToCivl.cvl

```

## Directory Descriptions

- `scaffolding/`: Contains definitions of program elements (functions, type definitions, etc.) needed for verification. These are simplified definitions, not necessarily the same as in the actual PETSc code.
  - `include/`: Header files for scaffolding.
  - `src/`: Source files for scaffolding implementations.

- `examples/`: Contains examples (from PETSc or otherwise) that we can verify. These examples may use the scaffolding.

- `functions/`: Contains excerpts of actual PETSc code, with one function per subdirectory. Our goal is to verify these without modification.
  - Each function has its own subdirectory (e.g., `VecNorm/`) containing:
    - The original PETSc function definition (e.g., `VecNorm.c`)
    - A CIVL driver for verification (e.g., `VecNorm_driver.cvl`)
    - A test file (e.g., `VecNorm_test.c`)
    - A Makefile for building and running tests

- `original/`: Contains the actual unmodified PETSc code relevant to this project.

## Verification Process

The verification process relies on the scaffolding found in the `scaffolding/` directory to supply all the necessary definitions for the functions under test. Each function within the `functions/` directory is isolated for individual verification using CIVL.

A super Makefile is employed to automate the CIVL verification of these individual function implementations. Every function resides in its own subdirectory and has a dedicated Makefile (with an `all` target) that triggers its CIVL verification. The super Makefile dynamically detects these subdirectories, executes their verification routines, and produces an overall report.

### How It Works

1. **Automatic Discovery of Function Subdirectories**  
   The Makefile uses the `find` command to list all immediate subdirectories in the `functions/` folder, excluding any hidden directories (those beginning with a dot) and directories named `CIVLREP`. This ensures that any new function folder added to `functions/` is automatically incorporated into the verification process without requiring manual intervention.

2. **Verification Target (`verify`)**  
   The default target (`all`) triggers the verification process by:
   - Printing header messages to signal the start of verification.
   - Removing any pre-existing `test_results.log` file.
   - Iterating through each discovered subdirectory:
     - Changing into the subdirectory.
     - Running `make all` to initiate the CIVL verification for that function.
     - Checking the exit status:
       - If the verification is successful, it prints a "TEST SUCCESS" message and logs the result as `SUCCESS` in `test_results.log`.
       - If the verification fails, it prints a "TEST FAIL" message and logs it as `FAIL` in the same file.
   - Once all subdirectories have been processed, it prints a summary and saves the detailed log to `test_results.log`.

3. **Cleanup Process**  
   The `clean` target iterates through each function subdirectory, executing their respective `clean` targets, and also removes the generated log file.

### Report Generation

- **Console Output:**  
  As the Makefile runs, it outputs messages for each function indicating whether the test succeeded or failed.
  
- **Log File:**  
  The `test_results.log` file records the outcome for each function directory with an entry of either `SUCCESS` or `FAIL`, serving as a detailed report of the verification process.

Using this super Makefile, developers can quickly verify that all function implementations in the `functions/` directory pass their individual CIVL checks, with any failures immediately identifiable from both the on-screen summary and the log file.

## Note

This structure is subject to change as the project develops. Please refer to this README.md for the most up-to-date information on the repository structure and verification process.
