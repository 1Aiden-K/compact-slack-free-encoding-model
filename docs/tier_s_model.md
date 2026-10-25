
## Tier S Model

### Parameters

```math
N = 4, \quad K = 2, \quad Q = 7
```

```math
d_i \in \{1,2,3,4\}, \qquad D_{ij} = \lVert r_i-r_j \rVert_2
```

```math
T=\sum_{i=1}^{N}d_i=10
```

where:

- `N` is the number of locations.
- `K` is the number of vehicles.
- `Q` is the capacity of each vehicle.
- `d_i` is the demand at location `i`.
- `D_ij` is the Euclidean distance between locations `i` and `j`.
- `T` is the total demand.

### Decision Variables

```math
z_i =
\begin{cases}
1 & \text{if assigned to Vehicle 1} \\
0 & \text{if assigned to Vehicle 2}
\end{cases}
```

Each location is assigned to exactly one vehicle.

### Clustering Objective

The objective minimizes the total distance between locations assigned to the same vehicle.

```math
C(z) = \sum_{i < j} D_{ij}(1-z_i-z_j+2z_iz_j)
```

The expression

```math
1-z_i-z_j+2z_iz_j
```

equals 1 when locations `i` and `j` are assigned to the same vehicle and 0 otherwise.

### Capacity Constraints

The total demand assigned to each vehicle must not exceed its capacity.

**Vehicle 1:**

```math
\sum_i d_i z_i \leq Q
```

**Vehicle 2:**

```math
\sum_i d_i(1-z_i) \leq Q
```

Since the total demand is

```math
T=\sum_i d_i
```

the capacity constraints can also be expressed as:

```math
T-Q \leq \sum_i d_i z_i \leq Q
```

The original constrained optimization problem is:

```math
\begin{aligned}
\min_z \quad & C(z) \\
\text{subject to}\quad
& \sum_i d_i z_i \leq Q \\
& \sum_i d_i(1-z_i) \leq Q
\end{aligned}
```

### Capacity Margins

Instead of introducing binary-encoded slack variables, define capacity margins for both vehicles.

**Vehicle 1:**

```math
h_1(z)=Q-\sum_i d_i z_i
```

**Vehicle 2:**

```math
h_2(z)=Q-\sum_i d_i(1-z_i)
```

The original capacity constraints require:

```math
h_1(z)\geq 0,\qquad h_2(z)\geq 0
```

Adding the capacity margins:

```math
h_1(z)+h_2(z)=2Q-T
```

Since `Q` and `T` are fixed, the sum of the capacity margins is constant.

### Capacity Penalty

The capacity constraints are approximated using an unbalanced quadratic penalty.

```math
P(z)=-\alpha\left(h_1(z)+h_2(z)\right)
+\lambda\left(h_1(z)^2+h_2(z)^2\right)
```

where:

- `α` is the linear penalty coefficient.
- `λ > 0` is the quadratic penalty weight.

Since

```math
h_1(z)+h_2(z)=2Q-T
```

the linear penalty is constant and does not affect optimization.

Expanding the quadratic penalty:

```math
\begin{aligned}
h_1(z)^2+h_2(z)^2
={}&2\left(\sum_i d_i z_i-\frac{T}{2}\right)^2\\
&+2\left(Q-\frac{T}{2}\right)^2
\end{aligned}
```

The second term is constant.

Therefore, removing terms that do not affect optimization gives the equivalent capacity penalty:

```math
P(z)=2\lambda\left(\sum_i d_i z_i-\frac{T}{2}\right)^2
```

This penalty encourages both vehicles to carry approximately equal total demand.

Unlike the original capacity constraints, this penalty does not explicitly enforce the capacity limit `Q`.

### Complete QUBO

The complete QUBO objective combines the clustering cost and the load-balancing penalty.

```math
F(z)=C(z)+P(z)
```

Expanding the objective:

```math
\begin{aligned}
F(z)={}&
\sum_{i < j}D_{ij}(1-z_i-z_j+2z_iz_j)\\
&+2\lambda\left(\sum_i d_i z_i-\frac{T}{2}\right)^2
\end{aligned}
```

The optimization problem is:

```math
\min_z F(z)
```

**Binary constraints:**

```math
z_i\in\{0,1\}
```

The QUBO objective minimizes within-vehicle distances while encouraging balanced demand between vehicles.

### Penalty Weight

The quadratic penalty weight controls the preference for balanced vehicle loads.

```math
\lambda>0
```

A larger `λ` places greater importance on balancing total demand, while a smaller `λ` places greater importance on minimizing the clustering objective.

Unlike the slack-variable formulation, the load-balancing penalty is not an exact encoding of the capacity constraints.

In particular, the capacity `Q` appears only in constant terms that are removed from the optimization objective.

Therefore, the condition

```math
\lambda>\sum_{i < j}D_{ij}
```

is not, in general, a sufficient guarantee of capacity feasibility for this surrogate.

Every sampled assignment must be evaluated against the original capacity constraints:

```math
\sum_i d_i z_i\leq Q
```

```math
\sum_i d_i(1-z_i)\leq Q
```

The original clustering objective `C(z)` is used to evaluate the quality of feasible assignments.

### Standard QUBO Form

The objective can be expanded into the standard quadratic unconstrained binary optimization form:

```math
F(x)=c+\sum_a h_a x_a+\sum_{a < b}J_{ab}x_ax_b
```

where:

- `x` represents the complete vector of binary vehicle assignment variables.
- `c` is the constant offset.
- `h_a` represents the linear QUBO coefficients.
- `J_ab` represents the quadratic interaction coefficients.

Since no slack variables are required:

```math
x=z
```

The constant offset is:

```math
c=\sum_{i < j}D_{ij}+\frac{\lambda T^2}{2}
```

The linear coefficients are:

```math
h_i=-\sum_{j\neq i}D_{ij}
+2\lambda(d_i^2-Td_i)
```

The quadratic interaction coefficients are:

```math
J_{ij}=2D_{ij}+4\lambda d_i d_j,
\qquad i<j
```

**Symmetric QUBO Matrix:**

The same objective can be written in matrix form:

```math
F(z)=c+z^{\mathsf T}\mathbf{M}z
```

where `M` is the symmetric QUBO matrix, distinguished from the vehicle capacity `Q`.

**Diagonal coefficients:**

```math
M_{ii}=-\sum_{j\neq i}D_{ij}
+2\lambda(d_i^2-Td_i)
```

**Off-diagonal coefficients:**

```math
M_{ij}=D_{ij}+2\lambda d_i d_j,
\qquad i\neq j
```

The matrix is symmetric:

```math
M_{ij}=M_{ji}
```

The constant offset `c` can be omitted when solving the QUBO because it does not affect the optimal assignment.

### Variable Count

The total number of binary variables consists only of the vehicle assignment variables.

Unlike the original slack-variable formulation, no additional binary variables are required to represent vehicle capacity.

```math
N_{\mathrm{bits}}=N
```

Substituting the problem parameters:

```math
N_{\mathrm{bits}}=4
```

Therefore:

```math
\boxed{N_{\mathrm{bits}}=4}
```

### Capacity Feasibility

The QUBO minimizes a load-balancing surrogate rather than enforcing the original vehicle capacity constraints exactly.

As a result, an assignment minimizing `F(z)` is not automatically guaranteed to satisfy both capacity limits.

For every sampled assignment, feasibility is determined by:

```math
\sum_i d_i z_i\leq Q
```

```math
\sum_i d_i(1-z_i)\leq Q
```

Only assignments satisfying both constraints are considered feasible solutions to the original clustering problem.

The original clustering cost is then calculated as:

```math
C(z)=\sum_{i < j}D_{ij}(1-z_i-z_j+2z_iz_j)
```

This separates the surrogate QUBO optimization from the evaluation of capacity feasibility and the original clustering objective.

## Ising Hamiltonian
Our QUBO matrix is of the form:
```math
F(z)=c+\sum_i M_{ii}z_i+2\sum_{i \lt j}M_{ij}z_iz_j
```

Represent each bit with one atom:
```math
|0\rangle=|g\rangle,\qquad |1\rangle=|r\rangle, \qquad \hat n_i=|r\rangle_i\langle r|
```

Ignoring c (not required for optimization) gives:  
```math
\frac{H_{\mathrm{target}}}{\hbar}
=
\gamma\sum_i M_{ii}\hat n_i
+
2\gamma\sum_{i<j}M_{ij}\hat n_i\hat n_j
```

### Neutral Atom Hamiltonian

```math
\frac{H(t)}{\hbar}
=
\frac{\Omega(t)}2\sum_i X_i
-\delta(t)\sum_i\hat n_i
+\sum_{i<j}V_{ij}\hat n_i\hat n_j
```

where:

```math
V_{ij}=\frac{\widetilde C_6}{R_{ij}^{6}}, \qquad \widetilde C_6=C_6/\hbar
```

*Note the factor of 1/2, as each off diagonal pair is counted twice in our QUBO matrix*
```math
R_{ij}^{*}
=
\left(\frac{\widetilde C_6}{2\gamma M_{ij}}\right)^{1/6}.
```

## Global Detuning
*Exact direct matching with global detuning requires equal $M_{ii}$*
We can fix the diagonals with mean target detuning.
```math
\bar m=\frac1N\sum_i M_{ii}
```

Pasqal’s solver also uses mean target detuning when DMM is disabled, with its own coefficient normalization.

```math
\delta_f=-\gamma\bar m
```

## Pulse Functions

### Rabi Envelope
```math
\Omega(t)=\Omega_{\max}\sin^2\left(\frac{\pi t}{\tau}\right)

```

Where $\tau$ is the pulse duration

```math
0\le t \le \tau
```

### Detuning Endpoints

```math
\delta_i=-\gamma\max_j|M_{jj}|,
\qquad
\delta_f=-\frac{\gamma}{N}\sum_jM_{jj}
```
Therefore the linear sweep is defined as 

```math
\delta(t)=\delta_i+(\delta_f-\delta_i)\frac{t}{T}
```
