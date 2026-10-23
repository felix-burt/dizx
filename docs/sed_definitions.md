# Syndrome extraction diagrams

Builds on [definitions.md](definitions.md). All diagrams here are phase-free and have no classical outputs.

## Projector gadget

For a CSS stabiliser generator $S$ of type $P \in \{X, Z\}$, the **projector gadget** is a phase-free ZX diagram representing

$$
\Pi_S = \tfrac{1}{2}(I + S)
$$

(up to scalar): a **generator spider** $g_S$ joined to one **data spider** per qubit in $\operatorname{supp}(S)$.

| generator type | generator spider | data spiders |
|---|---|---|
| $Z$ | $X$ (red) | $Z$ (green) |
| $X$ | $Z$ (green) | $X$ (red) |

Data spiders do not necessarily correspond to data qubits; that depends on the realisation.

## SED

A **layer** of type $P$ is all type-$P$ projector gadgets composed together, with neighbouring data spiders on the same qubit fused: one type-$P$ data spider per qubit, joined to every type-$P$ generator spider acting on it.

A **syndrome extraction diagram (SED)** is a composition of alternating $Z$- and $X$-type layers, with all neighbouring same-colour spiders fused, and with input and output ports on each qubit. It is a codespace projector diagram and fixes no **realisation**.

## SED webs

Each generator $S$ occurs once per layer of its type. An SED carries three families of Pauli web:

| family | web | class |
|---|---|---|
| detecting region | two consecutive occurrences of $S$ | $O_w : I \mapsto I$ |
| stabiliser | last occurrence of $S$, carried to the outputs | $O_w : I \mapsto S$ |
| co-stabiliser | first occurrence of $S$, carried to the inputs | $O_w : S \mapsto I$ |
| logical | each chosen logical generator $L$, carried inputs to outputs | $O_w : L \mapsto L$ |

A logical may have one trivial side. For example, a $ZZ$ merge (with the seam closed) composed between two memory phases has the logicals $I \mapsto \bar Z \bar Z$ (the merge outcome: the merge's stabilisers, carried out by the patches' $\bar Z$ logicals), $\bar Z I \mapsto \bar Z I$, $I \bar Z \mapsto I \bar Z$ and $\bar X \bar X \mapsto \bar X \bar X$. Each is a minimal sheet: the single-patch logicals run straight up the patches' outer boundaries, and the merge outcome rises from the merge's last layer (its lid) straight up the inner boundaries. $\bar Z \bar Z \mapsto I$ is their sum, so it is not a separate web.

## Composition

For diagrams $D_1, D_2$ with $|\mathcal{B}_\text{out}(D_1)| = |\mathcal{B}_\text{in}(D_2)|$, the **sequential composition** $D_1 ; D_2$ joins output $k$ of $D_1$ to input $k$ of $D_2$ and removes both ports; neighbouring same-colour spiders joined by a single new edge are fused.

A web of $D_1 ; D_2$ is a pair $(w_1, w_2) \in \mathcal{W}(D_1) \times \mathcal{W}(D_2)$ agreeing on every joined edge, i.e. $P_\text{out}(w_1) = P_\text{in}(w_2)$. In particular:

| $w_1$ | $w_2$ | composite |
|---|---|---|
| detector / co-stabiliser | — | unchanged |
| — | detector / stabiliser | unchanged |
| stabiliser $S$ | co-stabiliser $S$ | detector |
| logical $L$ | logical $L$ | logical |

Webs not glued this way are found in turn: webs carried straight through, logicals first ($A \mapsto B$ continued as $B \mapsto B$, confined to those qubits' wires), then webs glued to partners that consume their boundary, then general solutions. A web whose boundary is already a sum of the chosen webs' boundaries is reduced by them to a detector. A composite web built from a logical is a logical; any other is filed by which boundary is trivial. The $S \mapsto S$ webs of a partial round (stabilisers carried through) are filed with the logicals but marked as *carried*, and do not count as logicals here.

An SED with $R$ rounds is $B ; B ; \cdots ; B$ for a single-round body $B$.

## Closing ports

Closing a port $p$ in basis $b$ prepares (input) or measures (output) its wire in $b$: the wire must end at a spider of colour $c(b)$, with $c(X) = Z$ and $c(Z) = X$. If $p$'s neighbour has colour $c(b)$, $p$ is removed; otherwise $p$ becomes a degree-one spider of colour $c(b)$.

A web survives iff its label on $p$'s edge has no non-matching component for $c(b)$. The surviving webs are the webs of the closed diagram, refiled by their new boundary class.

A merge whose first and last layers have the seam basis's colour can close the seam by removing its ports; otherwise explicit preparation or projection spiders are needed.

## Sub-diagram boundaries

For a non-empty, proper, connected set of spiders $S$ with complement $\bar S$, $D[S]$ is the induced sub-diagram, with

$$
\partial E(S) = \{(u, v) \in E : u \in S,\ v \in \bar S\}, \qquad
\partial V(S) = \{u \in S : \exists\, v \in \bar S,\ (u, v) \in E\}.
$$

$D[S]$ has a **matching boundary** if the cut $\partial E(S)$ is a matching:

$$
|E(v) \cap \partial E(S)| = 1 \qquad \forall\, v \in \partial V(S) \cup \partial V(\bar S).
$$

## Realisation (not part of an SED)

A realisation is a separate layer over an SED that adds classical outputs (and worldlines), turning detecting regions into parity checks and fixing the outcome signs of stabilisers and logicals.
