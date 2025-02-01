# CIVL-PETSc Verification Project

This repository is being used to explore the application of CIVL to PETSc (Portable, Extensible Toolkit for Scientific Computation). The structure may change frequently as the project evolves.

## Current Structure
```
.
├── build_functions.sh
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
│   ├── VecConcatenate
│   │   ├── Makefile
│   │   ├── VecConcatenate.c
│   │   ├── VecConcatenate_driver.cvl
│   │   └── VecConcatenate_test.cvl
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
├── README
├── scaffolding
│   ├── include
│   │   ├── civlcomplex.cvh
│   │   ├── civlvec.cvh
│   │   ├── matrix.cvh
│   │   ├── petscvec.h
│   │   └── scalars.cvh
│   ├── src
│   │   └── vec
│   │       ├── civlcomplex.cvl
│   │       ├── civlvec.cvl
│   │       ├── Makefile
│   │       └── petscvec.c
│   ├── svn-commit.tmp~
│   └── test
│       ├── civlcomplex_test.cvl
│       ├── civlvec_test.cvl
│       ├── equals.c
│       ├── Makefile
│       └── petscToCivl.cvl
└── test_results.log
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

Below is a sample description you can add to your README file:

---

## Verification Process

The verification process leverages the scaffolding (located in the `scaffolding/` directory) to provide the necessary definitions for the functions used by the function being verified. Each function in the `functions/` directory is isolated for individual verification using CIVL.

A bash script (`build_functions.sh`) has been provided to automate the verification process for all functions. This script dynamically locates every subdirectory under `functions/` (excluding hidden directories and directories such as `CIVLREP`), and in each one it runs the `make all` target. The target is expected to run the CIVL verification for that function.

### How to Run the Script

1. **Make the Script Executable**  
   In the root directory of the repository, run:
   ```bash
   chmod +x build_functions.sh
   ```

2. **Execute the Script**  
   Run the script by executing:
   ```bash
   ./build_functions.sh
   ```

### Report Generation

- **Logging:**  
  As the script processes each function subdirectory, it logs the outcome (either `SUCCESS` or `FAIL`) for each function into a log file named `test_results.log`.

- **Summary:**  
  At the end of the verification, the script displays a summary on the terminal listing:
  - **Working:** All function directories that passed verification.
  - **Not Working:** All function directories that failed verification.

This automated approach allows you to quickly assess the status of all functions and identify any that may require further attention.

## Note

This structure is subject to change as the project develops. Please refer to this README.md for the most up-to-date information on the repository structure and verification process.
