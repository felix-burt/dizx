# Definitions

## ZX diagram

A **ZX diagram** is a graph $D = (V, E)$ with

- a **type map** $t : V \to \{Z, X, H, P\}$,
- a **phase map** $\alpha(v) = a_v \pi$ on $t^{-1}(\{Z, X\})$, with $a_v \in \mathbb{F}_2$ either fixed or a measurement outcome,

subject to $t(v) = H \Rightarrow \deg v = 2$ and $t(v) = P \Rightarrow \deg v = 1$.

- **Legs** of $v$: $E(v) = \{e \in E : v \in e\}$, so $\deg v = |E(v)|$.
- **Ports** ($t = P$) are split into inputs and outputs.
- **Boundary edges**: $\mathcal{B}(D) = \{(u, v) \in E : t(u) = P \text{ or } t(v) = P\} = \mathcal{B}_\text{in} \sqcup \mathcal{B}_\text{out}$. All other edges are **internal**.
- **Classical output**: a degree-one spider $c$ with the same type as its sole neighbour and phase $m_c \pi$, where $m_c$ is a measurement outcome.

## Pauli web

Encode a Pauli on edge $e$ as $w_e = X^{x_e} Z^{z_e}$ (up to phase), $(x_e, z_e) \in \mathbb{F}_2^2$:

| Pauli | $(x, z)$ | colour  |
|-------|----------|---------|
| $I$   | $(0, 0)$ | none    |
| $X$   | $(1, 0)$ | green   |
| $Z$   | $(0, 1)$ | red     |
| $Y$   | $(1, 1)$ | both    |

A **Pauli web** is a map $w : E \to \mathbb{F}_2^2$ satisfying, at every node $v$:

| $t(v)$ | condition |
|--------|-----------|
| $Z$    | $x_e = x_{e'}\ \forall e, e' \in E(v)$, and $\sum_{e \in E(v)} z_e = 0$ |
| $X$    | $z_e = z_{e'}\ \forall e, e' \in E(v)$, and $\sum_{e \in E(v)} x_e = 0$ |
| $H$    | for legs $e_1, e_2$: $(x_{e_1}, z_{e_1}) = (z_{e_2}, x_{e_2})$ |
| $P$    | none |

(Consequence: a $Z$ (resp. $X$) classical output's leg can carry only $I$ or $X$ (resp. $I$ or $Z$).)

Webs add edgewise over $\mathbb{F}_2^2$; the valid webs $\mathcal{W}(D)$ form an $\mathbb{F}_2$-vector space.

### Boundary operator

$$
P_\text{in}(w) = \bigotimes_{e \in \mathcal{B}_\text{in}} w_e, \qquad
P_\text{out}(w) = \bigotimes_{e \in \mathcal{B}_\text{out}} w_e, \qquad
O_w : P_\text{in}(w) \mapsto P_\text{out}(w) \quad (\text{unsigned}).
$$

| $P_\text{in}$ | $P_\text{out}$ | class |
|---|---|---|
| $I$ | $\neq I$ | stabiliser |
| $\neq I$ | $I$ | co-stabiliser |
| $\neq I$ | $\neq I$ | logical operator |
| $I$ | $I$ | detecting region |

### Outcome parity

For webs with no $Y$ edges on diagrams whose only non-zero phases are on classical outputs, let $\mathcal{C}(w)$ be the classical outputs whose leg is highlighted by $w$. Then

$$
\mu(w) = \sum_{c \in \mathcal{C}(w)} m_c \bmod 2.
$$

- Detecting region: $\mu(w) = 0$ absent errors (a parity check).
- Stabiliser / logical: signed map $O_w^{(m)} : P_\text{in}(w) \mapsto (-1)^{\mu(w)} P_\text{out}(w)$ (a frame update).
