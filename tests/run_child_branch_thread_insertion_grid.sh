#!/usr/bin/env bash
set -euo pipefail

# Run through all 20 test cases for child branch thread insertion. 
# Unreachable cases are not included:
#   Franka configuration 2: skip all cases except --kinova 2 --target 1
#   Franka configuration 4: skip all --target 1 cases
#   Franka configuration 4: skip all --kinova 2 cases

for f in 1 2 3 4; do
  for k in 1 2 3 4; do
    for hole in 1 2; do
      # Unreachable cases:
      [[ "$f" == "2" && !("$k" == "2" && "$hole" == "1")]] && continue 
      [[ "$f" == "4" && "$hole" == "1" ]] && continue
      [[ "$f" == "4" && "$k" == "2"]] && continue     

      python child_branch_thread_insertion.py \
        --kinova "$k" --franka "$f" --target "$hole" --align 0.05 --offset 0.05 --vis-only
    done
  done
done
