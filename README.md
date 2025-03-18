# CIVL-PETSc Verification Project

This repository is being used to explore the application of CIVL to PETSc (Portable, Extensible Toolkit for Scientific Computation). The structure may change frequently as the project evolves.

## Current Structure
```
.
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
│   ├── VecAXPBYPCZ_Seq
│   │   ├── Makefile
│   │   ├── VecAXPBYPCZ_Seq.c
│   │   ├── VecAXPBYPCZ_Seq_driver.cvl
│   │   └── VecAXPBYPCZ_Seq_test.cvl
.   .
.   .
.   .
│   └── VecWAXPY_Seq
│       ├── Makefile
│       ├── VecWAXPY_Seq.c
│       ├── VecWAXPY_Seq_driver.cvl
│       └── VecWAXPY_Seq_test.cvl
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

This makefile is designed to automatically run tests on all the function implementations found inside the functions folder. It does this by:

### How It Works

1. **Finding Function Folders**  
   The makefile automatically looks inside the "functions" folder and finds every subfolder—ignoring any hidden folders and the "CIVLREP" folder. This means if we add a new function, it will be picked up automatically.

2. **Running Full Tests**  
   When user run `make all`, the makefile goes into each of those subfolders and runs a complete set of tests. These tests check different vector sizes (from 1 to 5) and different processor counts (from 1 to 5) for both real and complex cases.

3. **Running Quick Tests**  
   If user want faster results, running `make runsmall` does a quicker set of tests. It tests with smaller vector sizes (1 to 3) and fewer processor counts (1 and 2), while still checking both real and complex cases.

4. **Cleaning Up**  
   The `make clean` target goes into each function folder and runs its clean routine.

## Note

This structure is subject to change as the project develops. Please refer to this README.md for the most up-to-date information on the repository structure and verification process.
