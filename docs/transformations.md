# Web-preserving transformations

Builds on [definitions.md](definitions.md) and [sed_definitions.md](sed_definitions.md). All diagrams here are phase-free with no classical outputs, and ports keep their labels, so boundary edges of $D$ and $D'$ can be compared directly.

## Web-preserving

A transformation $D \mapsto D'$ is **web-preserving** if it induces a linear bijection $\tau : \mathcal{W}(D) \to \mathcal{W}(D')$ with

$$
O_{\tau(w)} = O_w \qquad \forall\, w \in \mathcal{W}(D).
$$

- Equality of the represented maps does not imply this: a rewrite can add or remove internal webs.
- Compositions and inverses of web-preserving transformations are web-preserving.
- Web preservation does not imply distance preservation, which depends on a lower-level decomposition and a noise model.

## Matching components

At a spider $v$, split each leg's label $(x_e, z_e)$ into

| $t(v)$ | matching $m_e$ | non-matching $n_e$ |
|---|---|---|
| $Z$ | $x_e$ | $z_e$ |
| $X$ | $z_e$ | $x_e$ |

so the web rules at $v$ are: $m_e$ is constant on $E(v)$ and $\sum_{e \in E(v)} n_e = 0$.

## Rules

Each rule acts locally on a spider or an edge. $\tau$ leaves labels on unaffected edges unchanged and is given below on new edges. Every rule also holds with $Z$ and $X$ exchanged.

### Duality $\mathcal{D}_v$

Exchange the colour of spider $v$ and replace each leg $e = (v, u)$ by $v - h_e - u$ with a new $H$ node $h_e$.

$$
\tau(w)_{(h_e, u)} = (x_e, z_e), \qquad \tau(w)_{(v, h_e)} = (z_e, x_e).
$$

$\mathcal{D}_v^2$ is the identity after cancelling adjacent $H$ pairs.

### Identity insertion $\mathcal{I}_e^c$

Insert a phase-free degree-2 spider $s$ of colour $c \in \{Z, X\}$ on edge $e = (u, v)$, giving edges $(u, s)$ and $(s, v)$.

$$
\tau(w)_{(u, s)} = \tau(w)_{(s, v)} = w_e.
$$

### Split $\mathcal{S}_v(A, B)$

For a partition $E(v) = A \sqcup B$, replace $v$ by two spiders $v_A, v_B$ of the same colour with legs $A$ and $B$ respectively, joined by a single new edge $f = (v_A, v_B)$.

$$
m_f = m_e \ (\text{any } e \in E(v)), \qquad n_f = \sum_{e \in A} n_e = \sum_{e \in B} n_e .
$$

The inverse is spider fusion along a single edge.

**Non-example.** Joining $v_A$ and $v_B$ by two edges is not web-preserving: it adds a loop web in the non-matching colour (and fusing along two edges removes one). Splits must use exactly one joining edge.

## Loop insertion (not web-preserving)

### Parallel edge $\mathcal{P}_{u,v}$

For same-colour spiders $u, v$ joined by a single edge $f$, add a second edge $f'$. The linear map is unchanged up to scalar, and the web space gains one generator, the **loop web** $\ell$:

$$
\tau(w)_{f'} = (m_f, 0), \qquad \ell_f = \ell_{f'} = (0, 1),
$$

in (matching, non-matching) components. $\ell$ has trivial boundary, so it is a new detecting region.

For opposite-colour spiders joined through one H node, the copy $f'$ also goes through a new H node; $\ell$ is in $u$'s colour on $u$'s legs and $v$'s colour on $v$'s legs.

### Loop edge

For spiders $u, v$ joined by $f$, apply $\mathcal{P}_{u,v}$, then split $u$ so that $u$ keeps $A_u \cup \{f\}$ and a new spider $u'$ takes the rest and $f'$; likewise for $v$. The result is the cycle $u - v - v' - u'$, with $f, f'$ as two of its edges. Applied to every boundary edge of a region $S$ (with $v' \in S$), the cut crosses each loop twice.

### Loop generator

For a generator spider $g$ with legs $a, b, c, d$ in cyclic order:

$$
\mathcal{S}_g(\{a, b\}, \{c, d\}) \;\to\; \mathcal{P}_{g_{ab}, g_{cd}} \;\to\; \mathcal{S}_{g_{ab}}(\{a, f\}, \{b, f'\}),\ \mathcal{S}_{g_{cd}}(\{d, f\}, \{c, f'\}).
$$

The result is a 4-cycle of spiders, one per leg, with one new detecting region $\ell$ around the cycle.

## Boundary splitting

Let $S$ be a set of spiders in an SED with $\partial V(S)$ consisting only of generator spiders. For each $v \in \partial V(S)$, apply $\mathcal{S}_v(A, B)$ with

$$
A = \{(v, u) \in E(v) : u \in S\}, \qquad B = \{(v, u) \in E(v) : u \in \bar S\},
$$

keep $v_A$ in $S$ and $v_B$ in $\bar S$. Afterwards $\partial E(S) = \{(v_A, v_B)\}$, one edge per original boundary spider, so $D[S]$ has a matching boundary.

## In code

The rules are methods of `dizx.transformations.SEDEditor`, which edits an SED in place and returns the result with `freeze()`:

| Rule | Method |
|---|---|
| $\mathcal{D}_v$ (several spiders at once) | `dualise(nodes)` |
| $\mathcal{I}_e^c$ | `insert_identity(e, colour)` |
| $\mathcal{S}_v(A, B)$ | `split(v, A, name=None, direction=None)` |
| $\mathcal{P}_{u,v}$ | `add_parallel_edge(u, v)` |
| loop edge / loop generator | `loop_edge(u, v)`, `loop_generator(g, order=None)` |
| boundary splitting | `split_boundary(S)`, `split_by_side(side, spiders)` |
| inverses | `fuse(u, v)`, `remove_identity(s)`, `cancel_hadamards(h1, h2)` |

```python
ed = SEDEditor(sed)
ed.dualise(region)
f = ed.split(g, A)
sed2 = ed.freeze()
```

The editor indexes webs by the edges they sit on, so a rule only touches the webs on the edges it rewrites. Applying several rules to one editor gives exactly the SED that applying them one at a time gives. The module-level functions of the same names (`dualise(sed, nodes)`, `split(sed, v, A)`, …) each run one editor session and suit single rewrites.

`freeze()` checks the webs it produces according to `transformations.CHECK`:
- `"local"` (the default) checks the web rules at every node a rule touched;
- `"full"` re-checks the whole diagram;
- `"none"` skips the check.

Use `with checking("full"):` to change the mode temporarily.
