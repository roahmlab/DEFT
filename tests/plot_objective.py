#!/usr/bin/env python3
"""
Plot IPOPT optimization objective history
"""

import matplotlib.pyplot as plt
import numpy as np

# Hardcoded objective values from IPOPT output
iterations = [0, 1, 2, 3, 4, 5]
objectives = [7.6757520e-01, 1.7656795e-01, 2.1445120e-02, 9.9651138e-03, 7.4241270e-03, 7.2763835e-03]

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, objectives, 'o-', linewidth=2, markersize=10, color='blue')
plt.xlabel('IPOPT Iteration', fontsize=14)
plt.ylabel('Objective Value', fontsize=14)
plt.title('IPOPT Optimization Progress', fontsize=16)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('objective_history.png', dpi=150)
print("Objective history plot saved to: objective_history.png")
plt.show()
