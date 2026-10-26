# Finding Pauli webs with F2 linear algebra

There are two ways the code finds webs:

- **Directly**: a web is a vector in the nullspace of the diagram's constraint
  matrix $A$ (section 2).
  This is what `ZXDiagram.web_space` and `detector_space` compute. They are
  dense and pick an arbitrary basis, so they are only used for checks.
- **Incrementally**: every construction (`compose`, `close_ports`,
  `QECExperiment._find_webs`) starts from pieces whose webs are already known
  and solves for *combinations* of them. The same nullspace step is used, but on
  much smaller matrices.

## 1. A web is an F2 vector

`PauliWeb` (`src/dizx/zx_diagram.py`). The diagram has a fixed edge order,
`ZXDiagram.edges`, with $n = |E|$ edges. A web is the vector

$$
w = (x_0, \dots, x_{n-1} \mid z_0, \dots, z_{n-1}) \in \mathbb{F}_2^{2n},
$$

where $(x_i, z_i)$ gives the Pauli on edge $i$: $(1,0) = X$, $(0,1) = Z$,
$(1,1) = Y$.

- A web is stored sparsely as `idx`, the sorted positions of its 1s: $i$ for
  $x_i$ and $n + i$ for $z_i$.
- Adding webs is XOR (`__add__` uses `np.setxor1d`).
- A set of $k$ webs is a sparse $k \times 2n$ matrix with one web per row
  (`web_matrix`, `webs_from_matrix`).

## 2. The web rules form a linear system $Aw = 0$

Every web rule is a linear equation over $\mathbb{F}_2$ in the bits of $w$,
of the form "the sum of some of the $x_i$ and $z_i$ is $0$". The **constraint
matrix** $A \in \mathbb{F}_2^{r \times 2n}$ has one row per rule ($r$ rules in
total) and the same $2n$ columns as a web. Row $j$ has a $1$ in each column
whose bit appears (an odd number of times) in rule $j$, so

$$
(Aw)_j = \sum_{c\,:\,A_{jc} = 1} w_c ,
$$

and $Aw = 0$ says exactly that $w$ satisfies every rule. Each row belongs to
the node whose rule it is. In code, $A$ is the sparse matrix returned by
`web_constraints()`; `constraints_at(nodes)` builds the rows of only the given
nodes.

The rules are as follows. For a node $v$ let $E(v)$ be its legs.

| Node | Rules (one row each) |
|---|---|
| Z spider, legs $l_0, l_1, \dots$ | x is the same on every leg: $x_{l_0} + x_e = 0$ for each other leg $e$; z has even parity: $\sum_{e \in E(v)} z_e = 0$ |
| X spider | the same rules with $x$ and $z$ swapped (`const, parity = (0, n)` vs `(n, 0)`) |
| H node, legs $e_1, e_2$ | $x_{e_1} + z_{e_2} = 0$ and $z_{e_1} + x_{e_2} = 0$ |
| Port | no rule |

`add()` keeps only the columns that appear an odd number of times
(`Counter … k % 2`). This is how self-loops, which `legs()` lists twice, cancel
correctly over $\mathbb{F}_2$. `web_constraints()` caches $A$ (built from all
nodes) and also returns the node each row belongs to. The web space is then

$$
W(D) = \ker A = \{\, w \in \mathbb{F}_2^{2n} : Aw = 0 \,\}.
$$

### Worked example

The diagram `i — Z(a) =two edges= X(b) — o`, with edges in the order
$e_0 = (i,a)$, $e_1 = (a,b)_0$, $e_2 = (a,b)_1$, $e_3 = (b,o)$:

$$
A = \begin{array}{c|cccc|cccc|l}
 & x_0 & x_1 & x_2 & x_3 & z_0 & z_1 & z_2 & z_3 & \\ \hline
a & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & x \text{ constant at } a \\
a & 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & \\
a & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 0 & z \text{ even at } a \\
b & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & z \text{ constant at } b \\
b & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 1 & \\
b & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & x \text{ even at } b
\end{array}
$$

The rank is $6$ with $8$ columns, so $\dim W(D) = 8 - 6 = 2$.
`web_space()` returns these two webs:

- $X \to I$: $X$ on $e_0, e_1, e_2$;
- $I \to Z$: $Z$ on $e_1, e_2, e_3$.

To reproduce it:

```python
import networkx as nx
from dizx.zx_diagram import ZXDiagram
from dizx._f2 import rref

g = nx.MultiGraph()
g.add_edges_from([("i", "a"), ("a", "b"), ("b", "o"), ("a", "b")])
d = ZXDiagram(g, {"i": "P", "o": "P", "a": "Z", "b": "X"}, inputs=["i"], outputs=["o"])
A, owners = d.web_constraints()
print(A.toarray(), owners)
print(rref(A.toarray())[1])          # pivots [0, 1, 3, 4, 5, 6]
for w in d.web_space():
    print(w.boundary(), w.support())
```

`violations()` runs the same check on many webs at once. For a web matrix $W$
it computes $A W^\top \bmod 2$, and each nonzero entry is a broken rule at the
node that owns that row.

## 3. The F2 tools (`src/dizx/_f2.py`)

### Why Gauss–Jordan elimination works

**$\mathbb{F}_2$ is a field.** Its only nonzero element, $1$, is its own
inverse. So everything from linear algebra over $\mathbb{R}$ that relies only
on the field axioms also holds over $\mathbb{F}_2$: row reduction, rank, bases,
and rank–nullity. Nothing about webs needs special machinery. Finding them is
solving a homogeneous linear system $Aw = 0$.

**Row operations do not change the solutions.** Over $\mathbb{F}_2$ there are
only two elementary row operations:

- swap two rows;
- add one row to another (XOR).

Scaling a row is trivial, because the only nonzero scalar is $1$. Each
operation is left multiplication by an invertible matrix $E$, so

$$
EAw = 0 \iff Aw = 0, \qquad\text{i.e.}\qquad \ker(EA) = \ker A .
$$

The row space is also unchanged, since $E$ is invertible. So we can replace
$A$ by a simpler matrix with the same solutions.

**Reduced row echelon form makes the solutions readable.** Gauss–Jordan
elimination applies row operations until the matrix is in reduced row echelon
form (RREF) $R$:

- each nonzero row has a leading $1$, its *pivot*, strictly to the right of
  the pivot in the row above;
- each pivot column is a unit vector: the pivot is the only $1$ in its column.

Zero rows are dropped (`rref` returns only the first `row` rows). Split the
columns into pivot columns $p_0, p_1, \dots$ and *free* columns. Row $i$ of
$Rw = 0$ involves exactly one pivot variable, because no other pivot column has
a $1$ in that row:

$$
w_{p_i} + \sum_{f \text{ free}} R_{if}\, w_f = 0
\quad\Longrightarrow\quad
w_{p_i} = \sum_{f \text{ free}} R_{if}\, w_f .
$$

So the free variables can be chosen arbitrarily, and they determine the pivot
variables uniquely. That is a bijection
$\ker A \cong \mathbb{F}_2^{\#\text{free}}$, which gives:

- $\operatorname{rank} A = \#\text{pivots}$ (`rank` is `len(rref(A)[1])`);
- $\dim \ker A = 2n - \operatorname{rank} A$, which is rank–nullity;
- a basis of $\ker A$: set one free variable to $1$ and the rest to $0$. This
  is what `nullspace` does. The vectors are independent because, restricted to
  the free columns, they form an identity matrix. There are
  $\dim \ker A$ of them, so they span $\ker A$.

**Why Gauss–Jordan rather than plain Gaussian elimination.** Plain Gaussian
elimination clears only *below* each pivot, giving an echelon form. A pivot
variable can then still depend on later pivot variables, so you need back
substitution. Gauss–Jordan also clears *above* each pivot. Then every pivot
variable is written directly in terms of free variables, and the whole basis is
the single line `N[:, pivots] = R[:, free].T`.

**Pivots also pick out independent vectors.** Row operations do not change
linear dependencies between *columns*: if $\sum_k c_k A_{:,k} = 0$ then
$\sum_k c_k (EA)_{:,k} = 0$. In $R$ the pivot columns are distinct unit
vectors, so they are independent, and every free column is a combination of
earlier pivot columns. So the pivot columns of $A$ form a maximal independent
set of columns, chosen greedily from left to right. `_complete` uses this on
$\mathrm{cand}^\top$ to keep a maximal independent set of the rows of
$\mathrm{cand}$, preferring earlier rows.

**Exact and cheap over $\mathbb{F}_2$.** Over $\mathbb{R}$, elimination
divides by pivots, and you have to pick pivots carefully to avoid rounding
errors. Over $\mathbb{F}_2$ there is no division and no rounding: any $1$ is a
valid pivot, and adding rows is XOR. `rref` therefore packs 8 columns into each
byte (`np.packbits`) and XORs whole packed rows at once
(`P[column] ^= P[row]`). The cost is still about $r \cdot 2n \cdot
\operatorname{rank}$ bit operations for an $r \times 2n$ matrix, which is why
dense elimination is only used on small matrices: the global $A$ only in
checks, and the small boundary matrices $M$ and $C$ during construction
(section 4).

**When a full RREF is not needed.** To test whether a vector lies in the span
of some rows, or to add rows one at a time, an echelon form with distinct
pivots is enough. Reduce $v$ by every stored row whose pivot is set in $v$. If
the result is $0$ then $v$ is in the span, and otherwise it is independent of
the stored rows and becomes a new row. `Echelon` and `SparseEchelon` work this
way. Their tags record which rows were added, which is the same as row reducing
the augmented matrix $[\,A \mid I\,]$: the right-hand block keeps track of the
combination of original rows. `sp_solver` uses exactly this, with tag $2^r$
for row $r$.

### Dense

- **`rref(A)`**: Gauss–Jordan elimination over $\mathbb{F}_2$ on rows packed
  into bytes. For each column it finds a pivot row, swaps it up, and XORs it
  into every other row with a 1 in that column (`P[column] ^= P[row]`). It
  returns the reduced form $R$ and the pivot columns.
- **`nullspace(A)`**: the standard free-variable construction. Let $R$ be the
  reduced form with pivot columns $p_0, p_1, \dots$. Each free (non-pivot)
  column $f$ gives one basis vector $v^{(f)}$ with
  $$
  v^{(f)}_f = 1, \qquad v^{(f)}_{p_i} = R_{i f}, \qquad \text{all other entries } 0.
  $$
  This works because row $i$ of $Rv = 0$ reads $x_{p_i} + \sum_f R_{if} x_f = 0$,
  and over $\mathbb{F}_2$ minus equals plus, so $x_{p_i} = \sum_f R_{if} x_f$.
  In code it is the single line `N[:, pivots] = R[:, free].T`.
- **`solve(A, t)`**: finds $x$ with $xA = t$ using a nullspace trick. It looks
  for $y = (x, 1)$ in the nullspace of $\begin{pmatrix} A \\ t \end{pmatrix}^{\!\top}$,
  i.e. with last entry 1. Then
  $$
  y \begin{pmatrix} A \\ t \end{pmatrix} = xA + t = 0 \quad\Longrightarrow\quad xA = t.
  $$
- **`Echelon`**: an echelon form you add rows to one at a time. Each row carries
  a **tag** that is XORed along with it. After you reduce a vector, the tag
  tells you which combination of the stored rows you used.

### Sparse

- **`sp_rows`, `sp_canonical`, `sp_matmul`**: CSR matrices mod 2.
- **`SparseEchelon`**: the sparse version of `Echelon`. It pivots on each
  row's lowest column, so stored rows have distinct lowest columns. A vector
  whose lowest column is not a pivot is therefore outside the span, and
  reduction can stop early. Its tag is a Python int used as a bitmask.
- **`sp_rank`**: rank, by adding every row to a `SparseEchelon`.
- **`sp_solver(M)`**: returns a function `solve(t)` giving $x$ with $xM = t$.
  Row $r$ is stored with the tag $2^r$ (`1 << r`), so the tag left after
  reducing $t$ lists the rows used.

### The global spaces

- `web_space()` computes a basis of $\ker A$.
- `detector_space()` adds rows that force every boundary bit to 0, so it
  computes
  $$
  \{\, w \in \mathbb{F}_2^{2n} : Aw = 0,\ w|_{\partial} = 0 \,\},
  $$
  where $w|_\partial$ is the restriction of $w$ to the $x$ and $z$ bits of the
  boundary edges.

Both are used for checks such as `webs_complete()` and `detectors_complete()`.

## 4. How webs are actually built

The key idea is that you never solve $Aw = 0$ on a big diagram. Every operation
has a basis matrix $V \in \mathbb{F}_2^{G \times 2n}$ of $G$ known webs. A web
of the result is a combination $aV$ with $a \in \mathbb{F}_2^G$ that meets a few
boundary conditions. So the unknown lives in $\mathbb{F}_2^G$, and the
constraint matrix only has columns for the edges that matter.

### (a) `compose(d1, d2)`: gluing $d_1 ; d_2$

A web of the composite is a $d_1$ web and a $d_2$ web that agree on the joined
edges. Let $V_1$ ($n_1$ webs) and $V_2$ ($n_2$ webs) be the two bases, and
$G = n_1 + n_2$. Let `out1` $= O_1$ be the bits of $d_1$'s webs on the joined
outputs, and `in2` $= I_2$ the bits of $d_2$'s webs on the joined inputs.
For $a = (a_1, a_2)$ the condition is

$$
a_1 O_1 = a_2 I_2
\quad\Longleftrightarrow\quad
(a_1, a_2) \begin{pmatrix} O_1 \\ I_2 \end{pmatrix} = 0,
$$

so the composite's web space is the left nullspace of
$M = \begin{pmatrix} O_1 \\ I_2 \end{pmatrix}$ (`np.vstack([out1, in2])`), and

$$
\dim = G - \operatorname{rank} M.
$$

Before any general algebra, it takes the cheap combinations:

1. Webs with zero bits on the seam pass straight through.
2. A $d_1$ web and a $d_2$ web with identical seam bits are glued as a pair.
   These are matched by hashing the bit pattern (`by_in`). A stabiliser glued
   to its co-stabiliser gives a detector.

If that falls short of $\dim$, `_complete_glue` finds the rest. It keeps two
echelon forms:

- a `SparseEchelon` over combinations $a$, to test that each new combination is
  independent in $\mathbb{F}_2^G$;
- an `Echelon` over the composite boundaries $\partial(a)$, tagged with $a$. If
  a candidate's boundary is already a sum of chosen boundaries,
  $\partial(a) = \sum_{b \in S} \partial(b)$, then `reduce` returns the tag
  $\sum_{b \in S} b$, and $a + \sum_{b \in S} b$ has trivial boundary: it is a
  detector. This removes duplicates such as $ZZ \to I$ next to $I \to ZZ$.

Candidates are tried in three steps:

1. **Straight carries.** `_straight_web` solves a small `solve` restricted to
   one qubit's wire to get a straight sheet $s$. `sp_solver(V)` then finds the
   combination $y$ with $yV = s$. Logicals go first.
2. **Consuming partners.** `solve(in2[live2], out1[i])` finds $y$ with
   $y I_2 = (O_1)_i$, i.e. $d_2$ webs that cancel a $d_1$ web's seam bits, and
   the same the other way round.
3. **General solutions.** Whatever is still missing comes from
   `nullspace(M[live].T)`, a basis of $\{a : aM = 0\}$ over the webs that
   touch the seam.

### (b) `close_ports` and `_complete`

Closing a port in basis $b$ forbids one component on its edge ($z$ for a Z
spider, $x$ for an X spider). Let $C \in \mathbb{F}_2^{G \times c}$ hold, for
each web, the bits in those $c$ forbidden columns. The surviving space is

$$
\{\, aV : a \in \mathbb{F}_2^G,\ aC = 0 \,\}.
$$

1. Webs with zero row in $C$ survive on their own.
2. Webs with equal rows in $C$ are paired (their sum has a zero row).
3. `_complete` stacks those pairs on top of `nullspace(C[live].T)` and runs
   `rref(cand.T)`. The pivot columns of $\mathrm{cand}^\top$ are a maximal
   independent set of rows of $\mathrm{cand}$, chosen greedily with the pairs
   first.

### (c) `QECExperiment._find_webs` (`src/dizx/realisation.py`)

The same pattern again:

- `_port_constraints` gives $C$, one bit per prepared or measured port.
- Surviving webs (zero row in $C$) are kept as they are.
- The lost webs $L$ are recombined with a basis of $\{a : a\,C_L = 0\}$
  (`nullspace(C[lost].T)`).
- A combination is filed as an observable if it uses any logical, otherwise as
  a detector.

### Turning combinations back into webs

All three end the same way. Stack the combinations as the rows of a 0/1 matrix
$K \in \mathbb{F}_2^{m \times G}$ and compute

$$
W = K V, \qquad W' = W P,
$$

where $P \in \mathbb{F}_2^{2n \times 2n'}$ relabels columns from the old edge
order to the new one:

```python
W = sp_matmul(sp_rows(combos, G), V)                # K V: combinations → edge vectors
vecs = sp_matmul(W, edge_transport(src, dst, ...))  # W P: move into the new edge order
```

- `edge_transport` and `column_transport` build $P$, with $P_{jk} = 1$ when old
  column $j$ becomes new column $k$.
- `SEDiagram._check_webs` then checks validity with one product (via
  `violations`), each web's class from its boundary bits, and independence
  with `sp_rank`.

## 5. Suggested reading order

1. `_f2.rref` → `nullspace` → `solve`. Check them by hand on the $6 \times 8$
   matrix above.
2. `ZXDiagram.constraints_at` → `violations` → `web_space` / `detector_space`.
3. `SEDiagram._from_code_unrolled`. The webs here are written down directly,
   with no linear algebra. Everything else is built from them.
4. `compose` → `_complete_glue`, with `Echelon`, `SparseEchelon` and
   `sp_solver`.
5. `close_ports` / `_complete` → `QECExperiment._find_webs`.
