"""
Basic usage example for DEFT.

This script demonstrates how to use the DEFT package for basic BDLO simulation.
"""

import sys
import os
import torch

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deft.core.DEFT_sim import DEFT_sim
from deft.utils.util import DEFT_initialization


def main():
    """Run a basic DEFT simulation example."""
    print("DEFT Basic Usage Example")
    print("========================")
    
    # Set up basic parameters
    torch.set_default_dtype(torch.float64)
    device = "cpu"
    
    # Example parameters for BDLO1
    n_parent_vertices = 13
    n_child1_vertices = 5
    n_child2_vertices = 4
    n_branch = 3
    
    print(f"Parent vertices: {n_parent_vertices}")
    print(f"Child1 vertices: {n_child1_vertices}")
    print(f"Child2 vertices: {n_child2_vertices}")
    print(f"Number of branches: {n_branch}")
    
    # This is a minimal example - in practice you would need to:
    # 1. Load or define the undeformed BDLO shape
    # 2. Initialize all required parameters
    # 3. Set up the simulation with proper boundary conditions
    # 4. Run the simulation loop
    
    print("\nFor a complete example, see the training script in scripts/DEFT_train.py")


if __name__ == "__main__":
    main()