#!/usr/bin/env bash
set -euo pipefail

# Run through all 40 test cases for parent branch thread insertion. 

for c in 1 2 3 4 5; do
  for h in 1 2 3 4; do
    for hole in A B; do
      python parent_branch_thread_insertion.py \
        --config "$c" --height "$h" --hole "$hole" --align 0.05 --offset 0.05 
    done
  done
done