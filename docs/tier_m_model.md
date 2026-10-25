## Tier M Model

### Parameters

```math
N = 10, \quad K = 3, \quad Q = 20
```

```math
d_i \in \{1,2,\ldots,8\}, \qquad D_{ij} = \lVert r_i-r_j \rVert_2
```

```math
T=\sum_{i=1}^{N}d_i
```

where:

- `N` is the number of locations.
- `K` is the number of vehicles.
- `Q` is the capacity of each vehicle.
- `d_i` is the demand at location `i`.
- `D_ij` is the Euclidean distance between locations `i` and `j`.
- `T` is the total demand.

Unlike Tier S, the total demand `T` is instance-dependent.

A necessary condition for capacity feasibility is:

```math
T\leq KQ=60
```

This condition does not guarantee that the individual demands can be partitioned into three feasible vehicle loads.

### Decision Variables

Each location is represented using two binary variables:

```math
a_i,b_i\in\{0,1\}
```

The vehicle assignments are encoded as:

| `a_i` | `b_i` | Assignment |
|---|---|---|
| 0 | 0 | Vehicle 3 |
| 1 | 0 | Vehicle 1 |
| 0 | 1 | Vehicle 2 |
| 1 | 1 | Invalid |

Vehicle 3 is represented implicitly when both assignment variables equal zero.

The corresponding assignment indicators are:

```math
x_{i1}=a_i
```

```math
x_{i2}=b_i
```

```math
x_{i3}=1-a_i-b_i
```

For valid assignments:

```math
a_i+b_i\leq1
```

Each location is therefore assigned to exactly one vehicle.

The invalid assignment occurs when:

```math
a_i=b_i=1
```

Rather than introducing an additional binary variable, this combination is discouraged through a quadratic penalty.

### Clustering Objective

The objective minimizes the total distance between locations assigned to the same vehicle.

```math
\begin{aligned}
C(a,b)=\sum_{i<j}D_{ij}\big[&
a_i a_j+b_i b_j\\
&+(1-a_i-b_i)(1-a_j-b_j)
\big]
\end{aligned}
```

For valid assignments, the expression

```math
a_i a_j+b_i b_j+(1-a_i-b_i)(1-a_j-b_j)
```

equals 1 when locations `i` and `j` are assigned to the same vehicle and 0 otherwise.

Expanding the clustering objective:

```math
\begin{aligned}
C(a,b)=\sum_{i<j}D_{ij}\big[&
1-a_i-b_i-a_j-b_j\\
&+2a_i a_j+2b_i b_j\\
&+a_i b_j+b_i a_j
\big]
\end{aligned}
```

The clustering objective remains quadratic despite representing three vehicles.

Unlike Tier S, the model contains interactions between two different sets of assignment variables.

### Capacity Constraints

The total demand assigned to each vehicle must not exceed its capacity.

Define the vehicle loads:

**Vehicle 1:**

```math
A=\sum_i d_i a_i
```

**Vehicle 2:**

```math
B=\sum_i d_i b_i
```

**Vehicle 3:**

```math
L_3=\sum_i d_i(1-a_i-b_i)
```

Since the total demand is:

```math
T=\sum_i d_i
```

the third vehicle load can also be written as:

```math
L_3=T-A-B
```

The original capacity constraints are:

**Vehicle 1:**

```math
A\leq Q
```

**Vehicle 2:**

```math
B\leq Q
```

**Vehicle 3:**

```math
T-A-B\leq Q
```

The original constrained optimization problem is:

```math
\begin{aligned}
\min_{a,b}\quad & C(a,b)\\
\text{subject to}\quad
&A\leq Q\\
&B\leq Q\\
&T-A-B\leq Q\\
&a_i+b_i\leq1
\end{aligned}
```

All assignment variables are binary.

### Capacity Margins

Instead of introducing binary-encoded slack variables, define capacity margins for all three vehicles.

**Vehicle 1:**

```math
h_1(a,b)=Q-A
```

**Vehicle 2:**

```math
h_2(a,b)=Q-B
```

**Vehicle 3:**

```math
h_3(a,b)=Q-(T-A-B)
```

The original capacity constraints require:

```math
h_1(a,b)\geq0
```

```math
h_2(a,b)\geq0
```

```math
h_3(a,b)\geq0
```

Adding the capacity margins:

```math
h_1(a,b)+h_2(a,b)+h_3(a,b)=3Q-T
```

Since `Q` and `T` are fixed for a given instance, the sum of the capacity margins is constant.

### Capacity Penalty

The capacity constraints are approximated using an unbalanced quadratic penalty.

```math
\begin{aligned}
P_{\mathrm{capacity}}(a,b)
={}&-\alpha\sum_{k=1}^{3}h_k(a,b)\\
&+\lambda\sum_{k=1}^{3}h_k(a,b)^2
\end{aligned}
```

where:

- `α` is the linear penalty coefficient.
- `λ > 0` is the quadratic penalty weight.

Since:

```math
\sum_{k=1}^{3}h_k(a,b)=3Q-T
```

the linear penalty is constant and does not affect optimization.

Expanding the quadratic penalty:

```math
\begin{aligned}
\sum_{k=1}^{3}h_k(a,b)^2
={}&(Q-A)^2\\
&+(Q-B)^2\\
&+(Q-T+A+B)^2
\end{aligned}
```

Using the identity:

```math
\begin{aligned}
\sum_{k=1}^{3}(Q-L_k)^2
={}&\sum_{k=1}^{3}\left(L_k-\frac{T}{3}\right)^2\\
&+3\left(Q-\frac{T}{3}\right)^2
\end{aligned}
```

the second term is constant.

Therefore, removing terms that do not affect optimization gives the equivalent capacity penalty:

```math
\begin{aligned}
P_{\mathrm{capacity}}(a,b)
=\lambda\bigg[&
\left(A-\frac{T}{3}\right)^2\\
&+\left(B-\frac{T}{3}\right)^2\\
&+\left(T-A-B-\frac{T}{3}\right)^2
\bigg]
\end{aligned}
```

This penalty encourages all three vehicles to carry approximately equal total demand.

Unlike the original capacity constraints, this penalty does not explicitly enforce the capacity limit `Q`.

### Assignment Penalty

The two-bit encoding introduces an invalid assignment when both variables are equal to 1.

```math
a_i=b_i=1
```

To discourage this combination, introduce the assignment penalty:

```math
P_{\mathrm{assignment}}(a,b)=\eta\sum_i a_i b_i
```

where:

```math
\eta>0
```

The expression:

```math
a_i b_i
```

equals 1 only when both assignment variables are active.

Therefore, valid combinations contribute zero assignment penalty, while invalid combinations contribute a positive penalty.

Unlike the capacity penalty, the assignment penalty can guarantee valid vehicle assignments when its weight is sufficiently large.

### Adaptive Penalty Weight     

Rather than using a fixed penalty weight, we compute an instance-dependent $\lambda$ using total demand and mean pairwise customer distance.

```math
T = \sum_{i=1}^{N} d_i
```

```math
\overline{D} = \frac{2}{N(N-1)}\sum_{i<j}D_{ij}
```

The adaptive penalty is defined as

```math
\lambda = \operatorname{clip}\left(
0.3\frac{\overline D}{50}e^{0.4(T-48)},
\ 0.001,\ 100
\right)
```

The exponential factor increases the penalty as total demand approaches the fleet's capacity, while the distance factor scales it with the clustering objective.

The coefficients were selected empirically using Tier M benchmark instances. Computing \(\lambda\) requires only instance parameters, not knowledge of the optimal clustering.

### Complete QUBO

The complete QUBO objective combines the clustering cost, load-balancing penalty, and assignment penalty.

```math
F(a,b)=C(a,b)+P_{\mathrm{capacity}}(a,b)+P_{\mathrm{assignment}}(a,b)
```

Expanding the objective:

```math
\begin{aligned}
F(a,b)={}&
\sum_{i<j}D_{ij}\big[
a_i a_j+b_i b_j\\
&\qquad+(1-a_i-b_i)(1-a_j-b_j)
\big]\\
&+\lambda\bigg[
\left(A-\frac{T}{3}\right)^2\\
&\qquad+\left(B-\frac{T}{3}\right)^2\\
&\qquad+\left(T-A-B-\frac{T}{3}\right)^2
\bigg]\\
&+\eta\sum_i a_i b_i
\end{aligned}
```

The optimization problem is:

```math
\min_{a,b}F(a,b)
```

**Binary constraints:**

```math
a_i,b_i\in\{0,1\}
```

The QUBO objective minimizes within-vehicle distances while encouraging balanced demand between vehicles and penalizing invalid assignments.

No additional slack variables are required.

### Penalty Weights

The QUBO contains two independent penalty weights.

**Capacity Penalty Weight:**

```math
\lambda>0
```

A larger `λ` places greater importance on balancing total demand, while a smaller `λ` places greater importance on minimizing the clustering objective.

As in Tier S, the load-balancing penalty is not an exact encoding of the capacity constraints.

In particular, the capacity `Q` appears only in constant terms that are removed from the optimization objective.

Therefore, increasing `λ` does not generally guarantee capacity feasibility.

**Assignment Penalty Weight:**

```math
\eta>0
```

The assignment penalty discourages the invalid combination:

```math
a_i=b_i=1
```

A sufficient, conservative condition ensuring that every ground state of the target QUBO has valid vehicle assignments is:

```math
\eta>\max_i(-h_i)
```

where `h_i` is the linear QUBO coefficient derived below.

This condition is sufficient because all other quadratic interaction coefficients are nonnegative.

For any invalid assignment, removing one active variable reduces the assignment penalty and cannot increase the positive pairwise interaction terms.

Under the stated bound, the resulting reduction in the objective is strictly positive.

Therefore, no invalid assignment can be a global minimum of the target QUBO.

This guarantee applies to assignment validity, not capacity feasibility.

It also assumes the target QUBO coefficients are implemented exactly.

Approximate neutral-atom embedding or imperfect optimization can produce assignments that violate the original constraints.

Every sampled assignment must therefore be evaluated against both assignment validity and capacity feasibility.

### Standard QUBO Form

The objective can be expanded into the standard quadratic unconstrained binary optimization form:

```math
F(x)=c+\sum_u h_u x_u+\sum_{u<v}J_{uv}x_ux_v
```

where:

- `x` represents the complete vector of binary vehicle assignment variables.
- `c` is the constant offset.
- `h_u` represents the linear QUBO coefficients.
- `J_uv` represents the quadratic interaction coefficients.

Since no slack variables are required:

```math
x=
\begin{pmatrix}
a_1&\cdots&a_N&b_1&\cdots&b_N
\end{pmatrix}^{\mathsf T}
```

The complete objective can be written as:

```math
\begin{aligned}
F(a,b)={}&c+\sum_i h_i(a_i+b_i)\\
&+\sum_i J_i^{\mathrm{within}}a_i b_i\\
&+\sum_{i<j}\big[
J_{ij}^{\mathrm{same}}(a_i a_j+b_i b_j)\\
&\qquad\qquad+
J_{ij}^{\mathrm{cross}}(a_i b_j+b_i a_j)
\big]
\end{aligned}
```

**Constant Offset:**

```math
c=\sum_{i<j}D_{ij}+\frac{2\lambda T^2}{3}
```

**Linear Coefficients:**

The linear coefficients are identical for the two assignment variables belonging to the same location.

```math
h_i=-\sum_{j\neq i}D_{ij}+2\lambda(d_i^2-Td_i)
```

**Within-Location Interaction Coefficients:**

The interaction between the two assignment variables of the same location is:

```math
J_i^{\mathrm{within}}=\eta+2\lambda d_i^2
```

**Same-Type Interaction Coefficients:**

For distinct locations:

```math
J_{ij}^{\mathrm{same}}=2D_{ij}+4\lambda d_i d_j,
\qquad i<j
```

These coefficients correspond to interactions between variables of the same type:

```math
a_i a_j,\qquad b_i b_j
```

**Cross-Type Interaction Coefficients:**

For distinct locations:

```math
J_{ij}^{\mathrm{cross}}=D_{ij}+2\lambda d_i d_j,
\qquad i<j
```

These coefficients correspond to interactions between variables of different types:

```math
a_i b_j,\qquad b_i a_j
```

**Interaction Signs:**

Since:

```math
D_{ij}\geq0,\qquad d_i>0,\qquad\lambda>0,\qquad\eta>0
```

all quadratic interaction coefficients are positive.

```math
J_i^{\mathrm{within}}>0
```

```math
J_{ij}^{\mathrm{same}}>0
```

```math
J_{ij}^{\mathrm{cross}}>0
```

This is compatible with the positive Rydberg interaction strengths used in the neutral-atom model.

**Symmetric QUBO Matrix:**

The same objective can be written in matrix form:

```math
F(x)=c+x^{\mathsf T}\mathbf Mx
```

where `M` is the symmetric QUBO matrix, distinguished from the vehicle capacity `Q`.

Since the variables are divided into two groups, the matrix can be expressed in block form:

```math
\mathbf M=
\begin{pmatrix}
\mathbf M^{aa}&\mathbf M^{ab}\\
\mathbf M^{ba}&\mathbf M^{bb}
\end{pmatrix}
```

Each block has dimensions:

```math
N\times N
```

**Diagonal Coefficients:**

```math
M_{ii}^{aa}=M_{ii}^{bb}
=-\sum_{j\neq i}D_{ij}
+2\lambda(d_i^2-Td_i)
```

**Same-Type Off-Diagonal Coefficients:**

```math
M_{ij}^{aa}=M_{ij}^{bb}
=D_{ij}+2\lambda d_i d_j,
\qquad i\neq j
```

**Within-Location Cross-Type Coefficients:**

```math
M_{ii}^{ab}=M_{ii}^{ba}
=\frac{\eta}{2}+\lambda d_i^2
```

**Different-Location Cross-Type Coefficients:**

```math
M_{ij}^{ab}=M_{ij}^{ba}
=\frac{D_{ij}}{2}+\lambda d_i d_j,
\qquad i\neq j
```

The matrix is symmetric:

```math
\mathbf M^{\mathsf T}=\mathbf M
```

The same-type blocks satisfy:

```math
\mathbf M^{aa}=\mathbf M^{bb}
```

The cross-type blocks satisfy:

```math
\mathbf M^{ab}=\mathbf M^{ba}
```

The full symmetric QUBO matrix has dimensions:

```math
\mathbf M\in\mathbb R^{20\times20}
```

The constant offset `c` can be omitted when solving the QUBO because it does not affect the optimal assignment.

### Variable Count

The total number of binary variables consists of two vehicle assignment variables per location.

Vehicle 3 is encoded implicitly and does not require an additional binary variable.

Unlike a slack-variable formulation, no additional binary variables are required to represent vehicle capacity.

```math
N_{\mathrm{bits}}=2N
```

Substituting the problem parameters:

```math
N_{\mathrm{bits}}=2(10)
```

Therefore:

```math
\boxed{N_{\mathrm{bits}}=20}
```

This represents a three-vehicle clustering problem using 20 binary variables.

### Capacity Feasibility

The QUBO minimizes a load-balancing surrogate rather than enforcing the original vehicle capacity constraints exactly.

As a result, an assignment minimizing `F(a,b)` is not automatically guaranteed to satisfy all three capacity limits.

For every sampled assignment, feasibility is determined by two sets of conditions.

**Assignment Validity:**

```math
a_i+b_i\leq1,\qquad\forall i
```

**Vehicle 1 Capacity:**

```math
\sum_i d_i a_i\leq Q
```

**Vehicle 2 Capacity:**

```math
\sum_i d_i b_i\leq Q
```

**Vehicle 3 Capacity:**

```math
\sum_i d_i(1-a_i-b_i)\leq Q
```

Only assignments satisfying all four conditions are considered feasible solutions to the original clustering problem.

The original clustering cost is then calculated as:

```math
\begin{aligned}
C(a,b)=\sum_{i<j}D_{ij}\big[&
a_i a_j+b_i b_j\\
&+(1-a_i-b_i)(1-a_j-b_j)
\big]
\end{aligned}
```

This separates the surrogate QUBO optimization from the evaluation of assignment validity, capacity feasibility, and the original clustering objective.

## Ising Hamiltonian

Our QUBO matrix is of the form:

```math
F(x)=c+\sum_u M_{uu}x_u+2\sum_{u<v}M_{uv}x_ux_v
```

Represent each bit with one atom:

```math
|0\rangle=|g\rangle,\qquad |1\rangle=|r\rangle,
\qquad \hat n_u=|r\rangle_u\langle r|
```

Each location requires two atoms corresponding to its two binary assignment variables.

```math
a_i\longleftrightarrow\hat n_{a_i}
```

```math
b_i\longleftrightarrow\hat n_{b_i}
```

Ignoring `c` (not required for optimization) gives:

```math
\frac{H_{\mathrm{target}}}{\hbar}
=
\gamma\sum_u M_{uu}\hat n_u
+
2\gamma\sum_{u<v}M_{uv}\hat n_u\hat n_v
```

where `γ > 0` is the energy scaling factor converting the QUBO coefficients into angular-frequency units.

The target Hamiltonian contains:

- Linear occupation terms for all 20 atoms.
- Interactions between assignment variables of the same type.
- Interactions between different assignment types.
- Interactions between the two variables representing the same location.

The binary objective remains quadratic and therefore requires only single-atom and pairwise occupation terms.

### Neutral Atom Hamiltonian

```math
\frac{H(t)}{\hbar}
=
\frac{\Omega(t)}2\sum_u X_u
-\delta(t)\sum_u\hat n_u
+\sum_{u<v}V_{uv}\hat n_u\hat n_v
```

where:

```math
V_{uv}=\frac{\widetilde C_6}{R_{uv}^{6}},
\qquad \widetilde C_6=C_6/\hbar
```

To match the target pair interactions:

```math
V_{uv}=2\gamma M_{uv}
```

*Note the factor of 2, as each off-diagonal pair is counted twice in our symmetric QUBO matrix.*

Therefore, the target interatomic distances are:

```math
R_{uv}^{*}
=
\left(\frac{\widetilde C_6}{2\gamma M_{uv}}\right)^{1/6}
```

The model contains three types of pair interactions.

**Within-Location Interactions:**

For atoms representing the same location:

```math
R_{a_i b_i}^{*}
=
\left(
\frac{\widetilde C_6}
{\gamma(\eta+2\lambda d_i^2)}
\right)^{1/6}
```

**Same-Type Interactions:**

For distinct locations:

```math
R_{a_i a_j}^{*}
=
R_{b_i b_j}^{*}
=
\left(
\frac{\widetilde C_6}
{\gamma(2D_{ij}+4\lambda d_i d_j)}
\right)^{1/6}
```

**Cross-Type Interactions:**

For distinct locations:

```math
R_{a_i b_j}^{*}
=
R_{b_i a_j}^{*}
=
\left(
\frac{\widetilde C_6}
{\gamma(D_{ij}+2\lambda d_i d_j)}
\right)^{1/6}
```

All target interaction strengths are positive.

However, positive interaction coefficients do not guarantee that all target distances can be realized simultaneously in a two-dimensional atom arrangement.

The planar embedding must therefore approximate the target interaction matrix while respecting the geometric constraints of the neutral-atom device.

Consequently, exact equivalence between the target QUBO and the physically implemented Hamiltonian is not guaranteed.

## Global Detuning

*Exact direct matching with global detuning requires equal diagonal coefficients across all atoms.*

For Tier M, both binary variables associated with the same location have identical diagonal coefficients:

```math
M_{a_i a_i}=M_{b_i b_i}=h_i
```

However, the coefficients generally differ between locations because demands and distances vary.

We can approximate the diagonals using the mean target detuning.

```math
\bar m=\frac{1}{2N}\sum_{u=1}^{2N}M_{uu}
```

Since each location contributes two identical diagonal coefficients:

```math
\bar m=\frac1N\sum_i h_i
```

Pasqal's solver also uses mean target detuning when DMM is disabled, with its own coefficient normalization.

The final global detuning is:

```math
\delta_f=-\gamma\bar m
```

Equivalently:

```math
\delta_f=-\frac{\gamma}{N}\sum_i h_i
```

Using a global detuning replaces the nonuniform target diagonal coefficients with a uniform effective value.

Therefore, the implemented Hamiltonian generally approximates rather than exactly reproduces the original QUBO linear terms.

## Pulse Functions

### Rabi Envelope

```math
\Omega(t)=\Omega_{\max}\sin^2\left(\frac{\pi t}{\tau}\right)
```

Where `τ` is the pulse duration:

```math
0\leq t\leq\tau
```

The Rabi frequency begins and ends at zero, reaching its maximum value at the midpoint of the pulse.

### Detuning Endpoints

```math
\delta_i=-\gamma\max_u|M_{uu}|
```

```math
\delta_f=-\frac{\gamma}{2N}\sum_uM_{uu}
```

Since the diagonal coefficients are repeated for both assignment variables:

```math
\delta_i=-\gamma\max_i|h_i|
```

```math
\delta_f=-\frac{\gamma}{N}\sum_i h_i
```

Therefore, the linear sweep is defined as:

```math
\delta(t)=\delta_i+(\delta_f-\delta_i)\frac{t}{\tau}
```

where:

```math
0\leq t\leq\tau
```

The Rabi envelope and detuning sweep define the time-dependent evolution used to approximate the target QUBO ground state.

After measurement, the resulting 20-bit configurations are decoded into three vehicle assignments and evaluated against the original assignment and capacity constraints.