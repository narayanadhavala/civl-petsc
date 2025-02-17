#!/usr/bin/env bash

# This script expects to be run with Bash.
if [ -z "$BASH_VERSION" ]; then
  echo "Error: This script must be run with Bash."
  exit 1
fi

# Predefined inputs (no command line arguments are used)
COMMON_INPUTS="-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=2"
VecGetValues="${COMMON_INPUTS} -inputB=2"
VecConcatenate="${COMMON_INPUTS} -inputB=2"
VecSetValues="${COMMON_INPUTS} -inputB=2"
_Seq="-input_mpi_nprocs=1 -inputN_MIN=2 -inputN_MAX=2"

# Find all subdirectories in the 'functions' directory
mapfile -t SUBDIRS < <(find functions -mindepth 1 -maxdepth 1 -type d \
  -not -name ".*" \
  -not -name "CIVLREP" \
  -exec basename {} \;)

# SUBDIRS=("VecAXPY" "VecCopy_Seq" "VecGetValues")

# Remove old log file
rm -f Summary.log

# Record overall start time
start_script=$(date +%s)

# Counters and arrays for stats
total_functions=0
pass_count=0
fail_count=0
declare -A func_times
declare -A func_results

echo "============================================================================================================================" | tee -a Summary.log
echo "                                               Verifying civl-petsc functions                                               " | tee -a Summary.log
echo "============================================================================================================================" | tee -a Summary.log

# Loop over each function directory
for d in "${SUBDIRS[@]}"; do
  total_functions=$((total_functions + 1))
  
  # Create a colorful separation line (blue bold text)
  separator="\033[1;34m==================================================== Verifying $d =====================================================\033[0m"
  # Print to terminal and append to Summary.log (without ANSI codes in the log, if desired, you can strip them)
  echo -e "$separator" | tee -a Summary.log
  
  # Determine which inputs to pass based on the directory name.
  if [[ "$d" == *_Seq* ]]; then
    inputs="$_Seq"
  elif [[ "$d" == *VecGetValues* ]]; then
    inputs="${VecGetValues}"
  elif [[ "$d" == *VecConcatenate* ]]; then
    inputs="${VecConcatenate}"
  elif [[ "$d" == *VecSetValues* ]]; then
    inputs="${VecSetValues}"
  else
    inputs="${COMMON_INPUTS}"
  fi
  
  # Record the start time for this function
  start_func=$(date +%s)
  
  # Change into the directory and run 'make all' with the determined inputs.
  # Capture both stdout and stderr.
  output=$(cd "functions/$d" && make all EXTRA_INPUTS="$inputs" 2>&1)
  
  # Record the end time for this function and calculate elapsed time
  end_func=$(date +%s)
  duration=$(( end_func - start_func ))
  func_times["$d"]=$duration
  
  # Append the captured output to Summary.log
  echo "$output" | tee -a Summary.log
  
  # Check if the output contains the failure pattern
  if echo "$output" | grep -q "The program MAY NOT be correct"; then
    func_results["$d"]="FAIL"
    fail_count=$((fail_count+1))
  else
    func_results["$d"]="PASS"
    pass_count=$((pass_count+1))
  fi

  # Log the time taken for this function
  echo -e "\n\033[1;32mTime taken for $d: ${duration} seconds\033[0m" | tee -a Summary.log
  echo "" | tee -a Summary.log
done

# Record overall end time and calculate total elapsed time
end_script=$(date +%s)
total_time=$(( end_script - start_script ))

# Log final statistics
{
  echo "============================================================================================================================"
  echo "Final Summary:"
  echo "Total Time Taken: ${total_time} second(s)"
  echo "Total Number of Functions: ${total_functions}"
  echo "Number of Functions Passed: ${pass_count}"
  echo "Number of Functions Failed: ${fail_count}"
  echo ""
  echo "Individual Function Stats:"
  printf "  %-20s %-20s %s\n" "--------------------" "----------------------"  "-------"
  printf "  %-20s %-20s   %s\n" "      Function"     "      Time Taken"         "Result"
  printf "  %-20s %-20s %s\n" "--------------------" "----------------------"  "-------"
  for d in "${SUBDIRS[@]}"; do
    printf "  %-20s : %-10d second(s) [%s]\n" "$d" "${func_times[$d]}" "${func_results[$d]}"
  done
  echo "============================================================================================================================"
} | tee -a Summary.log

echo "Detailed log saved to Summary.log"
