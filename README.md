# CIVL-PETSc Verification Project

This repository is being used to explore the application of CIVL to PETSc (Portable, Extensible Toolkit for Scientific Computation). The structure may change frequently as the project evolves.

## Current Structure
```
.
├── build_functions.sh
├── common.mk
├── examples
│   ├── ex11.c
│   ├── ex1.c
│   ├── ex1.o
│   └── Makefile
├── functions
│   ├── VecAXPBY
│   │   ├── Makefile
│   │   ├── VecAXPBY.c
│   │   ├── VecAXPBY_driver.cvl
│   │   └── VecAXPBY_test.cvl
│   ├── VecAXPBYPCZ
│   │   ├── Makefile
│   │   ├── VecAXPBYPCZ.c
│   │   ├── VecAXPBYPCZ_driver.cvl
│   │   └── VecAXPBYPCZ_test.cvl
│   ├── VecAXPY
│   │   ├── Makefile
│   │   ├── VecAXPY.c
│   │   ├── VecAXPY_driver.cvl
│   │   └── VecAXPY_test.cvl
│   ├── VecAYPX
│   │   ├── Makefile
│   │   ├── VecAYPX.c
│   │   ├── VecAYPX_driver.cvl
│   │   └── VecAYPX_test.cvl
│   ├── VecConcatenate
│   │   ├── Makefile
│   │   ├── VecConcatenate.c
│   │   ├── VecConcatenate_driver.cvl
│   │   └── VecConcatenate_test.cvl
│   ├── VecConjugate_Seq
│   │   ├── Makefile
│   │   ├── VecConjugate_Seq.c
│   │   ├── VecConjugate_Seq_driver.cvl
│   │   └── VecConjugate_Seq_test.c
│   ├── VecCopy
│   │   ├── Makefile
│   │   ├── VecCopy.c
│   │   ├── VecCopy_driver.cvl
│   │   └── VecCopy_test.cvl
│   ├── VecCopy_Seq
│   │   ├── Makefile
│   │   ├── VecCopy_Seq.c
│   │   ├── VecCopy_Seq_driver.cvl
│   │   └── VecCopy_Seq_test.c
│   ├── VecDot
│   │   ├── Makefile
│   │   ├── VecDot.c
│   │   ├── VecDot_driver.cvl
│   │   └── VecDot_test.cvl
│   ├── VecDotRealPart
│   │   ├── Makefile
│   │   ├── VecDotRealPart.c
│   │   ├── VecDotRealPart_driver.cvl
│   │   └── VecDotRealPart_test.cvl
│   ├── VecGetValues
│   │   ├── Makefile
│   │   ├── VecGetValues.c
│   │   ├── VecGetValues_driver.cvl
│   │   └── VecGetValues_test.cvl
│   ├── VecMax
│   │   ├── Makefile
│   │   ├── VecMax.c
│   │   ├── VecMax_driver.cvl
│   │   └── VecMax_test.cvl
│   ├── VecMAXPBY
│   │   ├── Makefile
│   │   ├── VecMAXPBY.c
│   │   ├── VecMAXPBY_driver.cvl
│   │   └── VecMAXPBY_test.cvl
│   ├── VecMaxPointwiseDivide
│   │   ├── Makefile
│   │   ├── VecMaxPointwiseDivide.c
│   │   ├── VecMaxPointwiseDivide_driver.cvl
│   │   └── VecMaxPointwiseDivide_test.cvl
│   ├── VecMAXPY
│   │   ├── Makefile
│   │   ├── VecMAXPY.c
│   │   ├── VecMAXPY_driver.cvl
│   │   └── VecMAXPY_test.cvl
│   ├── VecMin
│   │   ├── Makefile
│   │   ├── VecMin.c
│   │   ├── VecMin_driver.cvl
│   │   └── VecMin_test.cvl
│   ├── VecMTDot
│   │   ├── Makefile
│   │   ├── VecMTDot.c
│   │   ├── VecMTDot_driver.cvl
│   │   └── VecMTDot_test.cvl
│   ├── VecNorm
│   │   ├── Makefile
│   │   ├── output.txt
│   │   ├── VecNorm.c
│   │   └── VecNorm_driver.cvl
│   ├── VecNormalize
│   │   ├── Makefile
│   │   ├── VecNormalize.c
│   │   ├── VecNormalize_driver.cvl
│   │   └── VecNormalize_test.cvl
│   ├── VecNormAvailable
│   │   ├── Makefile
│   │   ├── VecNormAvailable.c
│   │   ├── VecNormAvailable_driver.cvl
│   │   └── VecNormAvailable_test.c
│   ├── VecNorm_Seq
│   │   ├── Makefile
│   │   ├── VecNorm_Seq.c
│   │   ├── VecNorm_Seq_driver.cvl
│   │   └── VecNorm_Seq_test.c
│   ├── VecScale
│   │   ├── Makefile
│   │   ├── VecScale.c
│   │   ├── VecScale_driver.cvl
│   │   └── VecScale_test.cvl
│   ├── VecSet
│   │   ├── Makefile
│   │   ├── VecSet.c
│   │   ├── VecSet_driver.cvl
│   │   └── VecSet_test.cvl
│   ├── VecSetValues
│   │   ├── Makefile
│   │   ├── VecSetValues.c
│   │   ├── VecSetValues_driver.cvl
│   │   └── VecSetValues_test.cvl
│   ├── VecSetValuesBlocked
│   │   ├── Makefile
│   │   ├── VecSetValuesBlocked.c
│   │   ├── VecSetValuesBlocked_driver.cvl
│   │   └── VecSetValuesBlocked_test.cvl
│   ├── VecTDot
│   │   ├── Makefile
│   │   ├── VecTDot.c
│   │   ├── VecTDot_driver.cvl
│   │   └── VecTDot_test.cvl
│   └── VecWAXPY
│       ├── Makefile
│       ├── VecWAXPY.c
│       ├── VecWAXPY_driver.cvl
│       └── VecWAXPY_test.cvl
├── Makefile
├── README.md
└── scaffolding
    ├── include
    │   ├── civlcomplex.cvh
    │   ├── civlvec.cvh
    │   ├── matrix.cvh
    │   ├── petscvec.h
    │   └── scalars.cvh
    ├── src
    │   └── vec
    │       ├── civlcomplex.cvl
    │       ├── civlvec.cvl
    │       ├── Makefile
    │       └── petscvec.c
    └── test
        ├── civlcomplex_test.cvl
        ├── civlvec_test.cvl
        ├── equals.c
        ├── Makefile
        ├── petscToCivl.cvl
        └── vectorOwnershipTest.cvl

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

## Verification Process

The verification process leverages a custom shell script to automatically verify the CIVL correctness of function implementations found in the `functions/` directory. Each function is isolated in its own subdirectory and has its own Makefile (with an `all` target) to trigger its CIVL verification. The script automates the discovery of these subdirectories, executes their individual verification routines, and compiles a detailed report.

> **Note:** This process replaces the previous super Makefile approach. Instead of a Makefile, the shell script `build_functions.sh` performs all the required tasks.

### How It Works

1. **Automatic Discovery of Function Subdirectories**
   The script uses the `find` command to list all immediate subdirectories in the `functions/` folder. It excludes hidden directories (those beginning with a dot), as well as directories named `CIVLREP`, ensuring that any new function added to `functions/` is automatically incorporated into the verification process without manual updates.

2. **Per-Function Verification**
   For each discovered subdirectory, the script:
   - Prints a colorful header to indicate the start of verification for that function.
   - Determines the appropriate superset of input flags based on the function's name. For example:
     - Functions with `_Seq` use sequential inputs.
     - Functions matching `VecGetValues`, `VecConcatenate`, or `VecSetValues` receive an extra `-inputB=2` flag.
     - All others use common inputs.
   - Changes into the function's directory and executes `make all` with the determined inputs.
   - Captures the complete output (both stdout and stderr) and appends it to a detailed log file (`Summary.log`).
   - Measures the time taken for the verification of each function and displays this information immediately.

3. **Timing and Statistics**
   The script records:
   - **Overall Execution Time:** Capturing start and end times for the entire verification process.
   - **Individual Function Times:** Each function's execution time is measured and logged.
   - **Pass/Fail Status:** The script checks for a failure pattern (specifically, the phrase "The program MAY NOT be correct") in the output to determine if the verification passed or failed.
   - A final summary report is generated in a tabular format, showing:
     - Total time taken
     - Total number of functions verified
     - Individual execution time for each function
     - The number of functions that passed and failed

### Report Generation

- **Console Output:**
  As the script runs, it outputs colorful, formatted messages for each function's verification. It also displays the final summary in the terminal.

- **Log File:**
  All details—including the output of each function’s verification, execution times, and the final summary—are saved to `Summary.log`. This log provides a persistent record of the verification process.

### How to Run

Ensure that the script is executable and that you are using Bash (the script will exit if not run with Bash):

```bash
chmod +x build_functions.sh
./build_functions.sh
```

Using this Bash script, developers can quickly verify that all function implementations in the `functions/` directory pass their individual CIVL checks, with any failures immediately identifiable from both the on-screen summary and the log file.

## Note

This structure is subject to change as the project develops. Please refer to this README.md for the most up-to-date information on the repository structure and verification process.
