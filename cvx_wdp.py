#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp

# problem data
v1 = [2, 3, 3]
v2 = [4, 5, 10]
v = np.array(v1 + v2)
n = len(v)

# construct and solve the problem
x = cp.Variable(n)
objective = cp.Maximize(v @ x)
constraints = [
    0 <= x, x <= 1,         # continuous relaxation of x \in {0, 1}
    cp.sum(x[0::3]) <= 1,   # at-most-one constraint for agent 1
    cp.sum(x[3::])  <= 1,
    cp.sum(x[0] + x[2]) + cp.sum(x[3] + x[5]) <= 1, # pairwise-disjoint
    cp.sum(x[1] + x[2]) + cp.sum(x[4] + x[5]) <= 1
]
prob = cp.Problem(objective, constraints)
result = prob.solve(verbose=True)

# The optimal value for x is stored in `x.value` which is assigned when 
# prob.solve() is executed.
print(f"Optimal value for the problem is: {np.round(result, 4)}.")
print(f"Optimal solution for the problem is: {np.round(x.value, 4)}.")

# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(f"Dual variables for the constraint 0 <= x: " 
        f"{np.round(constraints[0].dual_value, 4)}.")
print(f"Dual variables for the constraint x <= 1: " 
        f"{np.round(constraints[1].dual_value, 4)}.")
print(f"Dual variables for the at-most-one constraints: " 
        f"{np.round(constraints[2].dual_value, 4)} for agent 1 and "
        f"{np.round(constraints[3].dual_value, 4)} for agent 2.")
print(f"Dual variables for the pairwise disjoint constraints: "
        f"{np.round(constraints[4].dual_value, 4)} and "
        f"{np.round(constraints[5].dual_value, 4)}.")
