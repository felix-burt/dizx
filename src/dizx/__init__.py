"""dizx: distributed quantum error correction from syndrome extraction ZX diagrams.

Pipeline: ``code`` (stabiliser codes) -> ``zx_diagram`` (SEDs and Pauli webs)
-> ``transformations`` (web-preserving rewrites) -> ``realisation``
(world-lines, links, QEC experiments) -> ``circuit_extractor`` (Stim
circuits, tagged for ``noise_models``). ``surgery`` and ``builders`` assemble
lattice-surgery experiments; ``erasure_matching`` and ``sinter_decoders``
decode them.

Modules are imported on demand; nothing heavy is loaded here.
"""
