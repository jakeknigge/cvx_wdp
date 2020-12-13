# Just Relax: A Real-Valued WDP

The winner determination problem (WDP) aims to find a welfare-maximizing 
allocation of goods for a combinatorial auction. The WDP can be expressed as the
optimization problem

```latex
\begin{array}{ll}
   \mbox{maximize}   & \displaystyle \sum_{i \in \mathcal{N}} 
                        \sum_{j \in \mathcal{B}} v_{ij} x_{ij} \\
   \mbox{subject to} & x_{ij} \in \{0, 1\}, \quad i \in \mathcal{N}, \, 
                        j \in \mathcal{M} \\
                     & \displaystyle \sum_{j \in \mathcal{B}} x_{ij} \leq 1, 
                        \quad i \in \mathcal{N} \\
                     & \displaystyle \sum_{i \in \mathcal{N}} \sum_{j \in 
                        \mathcal{S}} x_{ij} \leq 1, \quad 
                        \mathcal{S} \in \mathcal{C},
\end{array}
```

where $x \in \{0, 1\}^n$ are the optimization variables, $v_{ij}$ are the 
valuations assigned by agent $i$ to bundle $j$. The set $\mathcal{N}$ is the set
of agents, $\mathcal{M}$ is the set of marketable goods (*e.g.*, equities), 
$\mathcal{S} \subseteq \mathcal{M}$ is a subset representing bundles of goods,
and $\mathcal{B} = \{1, \ldots, 2^m-1\}$ gives the indices for a (total ordering
of the) power set of $\mathcal{S}$ (denoted as $\mathcal{P}(\mathcal{S})$), and
$\mathcal{C}$ is the collection (aka, set of sets) that contains the indices of
bids that share a common element.

## Time to Relax

The WDP looks like a linear program (LP) except for the binary-valued
optimization variable $x$. A common approach for handling binary- and
integer-valued variables is to *relax*—which is a fancy way of saying 
*make a constraint less of a constraint*—the integrality constraint to a 
continuous constraint. For the WDP, we swap $x_{ij} \in \{0, 1\}$ for 
$0 \le x_{ij} \le 1$ and *voila!* we have a LP. The *relaxed WDP* is the problem

```latex
\begin{array}{ll}
   \mbox{maximize}   & \displaystyle \sum_{i \in \mathcal{N}} 
                            \sum_{j \in \mathcal{B}} v_{ij} x_{ij} \\
   \mbox{subject to} & 0 \le x_{ij} \le 1, \quad i \in \mathcal{N}, \, 
                            j \in \mathcal{M} \\
                     & \displaystyle \sum_{j \in \mathcal{B}} x_{ij} \leq 1, 
                            \quad i \in \mathcal{N} \\
                     & \displaystyle \sum_{i \in \mathcal{N}} 
                            \sum_{j \in \mathcal{S}} x_{ij} \leq 1, 
                                \quad \mathcal{S} \in \mathcal{C}.
\end{array}
```

Modulo a transformation into a standard form, the relaxed WDP can be solved
(quickly and efficiently) with your favorite LP solver or algorithm.

## Example

We have items $A = \textsf{apples}$ and $B = \textsf{bananas}$, and their
combination $\{A, B\} = \textsf{apples and bananas}$. There are two bidders
interested in buying apples and/or bananas. Their valuation functions are 

```latex
\hat{v}_1 = \langle 2, 3, 3 \rangle, \quad \hat{v}_2 = \langle 4, 5, 10 \rangle,
```

meaning agent 1 values $A$ at 2, $B$ at 3 and $\{A, B\}$ at 3.

The relaxed WDP is

```latex
\begin{array}{ll}
    \mbox{maximize}     & 2x_{11} + 3x_{12} + 3x_{13} + 
                            4x_{21} + 5x_{22} + 10x_{23} \\    
    \mbox{subject to}   & 0 \le x_{ij} \le 1, \quad i \in \{1, 2\}, \, 
                            j \in \{1, 2, 3\} \\                        
                        & x_{11} + x_{12} + x_{13} \leq 1 \\                        
                        & x_{21} + x_{22} + x_{23} \leq 1 \\                        
                        & x_{11} + x_{13} + x_{21} + x_{23} \le 1 \\                        
                        & x_{12} + x_{13} + x_{22} + x_{23} \le 1.
\end{array}
```

The second and third constraints ensures that each agent receives *at most one*
bundle. The last two constraints ensure that no good gets allocated more than
once---*i.e.*, the goods are *pairwise disjoint*.

We can solve this problem in Python using `CVXPY` which is a modeling language
 for convex optimization problems. The chunk below specifies the problem data,
 constructs a problem instance and solves it. (See the file `cvx_wdp.py`.)
 
```python
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
```

CVXPY returns an optimal solution $x^\star$ of 
``x.value = [ 0. 0. -0. 0. 0. 1.]`` with an optimal value of 10, which was
agent 2's bid for $\{A, B\}$. (Although this problem is trivially small, it is
solvable on the order of ~100–200 microseconds.)
