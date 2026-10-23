# Realisations

Builds on [definitions.md](definitions.md), [sed_definitions.md](sed_definitions.md) and [transformations.md](transformations.md).

A **realisation** of an SED $D$ adds two annotations: **classical outputs** and **world-lines**. It need not fix gate-level decompositions, but must be enough to commit to a circuit-based, measurement-based or hybrid implementation.

## World-lines

A **world-line** is a directed path $P = (v_1 \to v_2 \to \cdots \to v_k)$ of distinct nodes, with consecutive nodes joined by quantum edges. $v_i \to v_{i+1}$ means the qubit acted on at $v_i$ persists and is next acted on at $v_{i+1}$.

A set of world-lines $\{P_i\}$ must satisfy:

- each spider and each edge lies on at most one $P_i$ (paths are vertex-disjoint);
- a port at the tail of a directed edge is an input; a port at the head is an output;
- classical outputs lie on no world-line.

Edges on no world-line are interactions between different qubits. A world-line containing exactly one spider is a **singleton**; it may also run to a single output port.

A **resource state** is a connected sub-diagram with no input ports in which every spider lies on its own singleton world-line.

## Classical outputs

For a spider $v$ with $\deg v \geq 1$ and no classical output, $\mathcal{C}_v$ attaches a degree-1 spider $c$ of the same colour with phase $m_c \pi$, where $m_c \in \mathbb{F}_2$ is a measurement outcome. This is not an equality of maps: $m_c = 0$ recovers $D$ and $m_c = 1$ puts a $\pi$ phase on $v$.

$\mathcal{C}_v$ is web-preserving. Each web extends uniquely to the new leg $f = (v, c)$ by

$$
n_f = 0, \qquad m_f = m_e \ (\text{any } e \in E(v)),
$$

so $c \in \mathcal{C}(w)$ iff $w$ is highlighted in $v$'s colour at $v$, and $\mu(w)$ gains $m_c$ for those webs.

On a generator spider for $S$, outcome $m$ selects the $(-1)^m$ eigenspace of $S$. A detecting region between consecutive occurrences of $S$ with outcomes $a, b$ then reports $a \oplus b = 0$ in the absence of errors.

## Circuit-based realisation

- **Data world-lines:** for each qubit, a world-line from its input port to its output port along the shortest path between them.
- **Ancillas:** each generator spider lies on its own singleton world-line and gets a classical output. Each ancilla represents one preparation and measurement; hardware may reuse it.

Arrows fix the order of the projector gadgets along each data qubit, but not the gates within a gadget.

## Measurement-based realisation

1. Apply duality $\mathcal{D}_v$ to every $X$ spider, making the diagram **graph-like** (only $Z$ spiders, joined via $H$ nodes).
2. Put every spider on its own singleton world-line and give it a classical output. Spiders adjacent to output ports may be left unmeasured if the aim is to prepare a state.

The result is a **cluster state**: each spider is a qubit prepared in $|+\rangle$ and each internal edge is a $CZ$. Compared with the circuit-based realisation, each bulk detecting region also includes the outcomes of the data spiders it passes through (four for a weight-4 generator), and each logical gains $d$ outcomes per round (surface code).

## QEC experiment

A **QEC experiment** is a realisation with every port closed, so that every qubit is initialised and measured:

- each input $p$ is replaced by a degree-1 **preparation** spider: $X$ for $|0\rangle$, $Z$ for $|+\rangle$;
- each output $p$ is replaced by a **measurement** spider ($X$ for a $Z$ measurement, $Z$ for an $X$ measurement) with a classical output;
- every spider lies on a world-line, and every world-line ends at a measured spider.

Closing port $p$ with a spider of type $T$ forces the non-matching component of $T$ on its boundary edge $e$ to vanish:

| basis | closing spider | constraint | surviving boundary Pauli |
|---|---|---|---|
| $Z$ | $X$ | $x_e = 0$ | $I, Z$ |
| $X$ | $Z$ | $z_e = 0$ | $I, X$ |

The surviving webs are the subspace of $\mathcal{W}(D)$ satisfying every constraint; all have trivial boundary. A basis of them splits into **detectors** (combinations of SED detectors, stabilisers and co-stabilisers) and **observables** (combinations involving a logical), defined up to adding detectors. Webs violating a constraint are **lost**, e.g. the $X$-type stabilisers and $\bar X$ in a $Z$-basis memory experiment, though combinations of lost webs may survive.
