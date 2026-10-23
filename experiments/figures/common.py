"""Shared figure helpers: the `_data.csv` written next to each snippet, and the
standalone wrapper that compiles a snippet to its own PDF.

Each generator that writes `<dir>/<name>.tex` also writes `<dir>/<name>_data.csv`:
one row per plotted value at full precision (the snippet rounds to 6
significant figures), in the plotted quantity's own units (rates, not the
percent the axes are labelled in), so the text can quote exact numbers and
their errors. Nothing else: no comments, no provenance columns.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Sequence


def data_path_for(tex_path: Path) -> Path:
    return tex_path.with_name(f"{tex_path.stem}_data.csv")


def plain_label(label: str) -> str:
    r"""A LaTeX legend label as plain text: `$3d$-\textsc{Chain}` -> `3d-Chain`."""
    text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", label)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    return text.replace("$", "").replace("{", "").replace("}", "").strip()


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)   # shortest round-trip form: every digit the fit produced
    return str(value)


def write_figure_data(tex_path: Path, columns: Sequence[str], rows: Iterable[Sequence]) -> Path:
    out = data_path_for(tex_path)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(v) for v in row])
    print(f"Wrote {out}")
    return out


# Covers every generator's preamble, groupplots included. The snippets size
# themselves against \linewidth / \textheight, which the standalone class
# leaves at meaningless values, so pin them to an A4 article text block -- the
# standalone PDF then matches the proportions the figure has in the paper.
_STANDALONE_TEMPLATE = r"""\documentclass[tikz,border=5pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\usepgfplotslibrary{{fillbetween}}
\usepgfplotslibrary{{groupplots}}
\usepackage[dvipsnames]{{xcolor}}
\begin{{document}}
\setlength{{\textwidth}}{{16cm}}\setlength{{\linewidth}}{{16cm}}\setlength{{\textheight}}{{24cm}}
\input{{{snippet}}}
\end{{document}}
"""


def compile_standalone(snippet: Path, out_dir: Path) -> Path | None:
    """Compile `snippet` to `<out_dir>/<snippet stem>.pdf` with latexmk.

    Returns None (with a message) when latexmk is not installed: the snippet
    is the paper's input, the PDF only a preview."""
    import shutil
    import subprocess
    if shutil.which("latexmk") is None:
        print("latexmk not found: skipping the standalone PDF")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    tex = out_dir / f"{snippet.stem}.tex"
    tex.write_text(_STANDALONE_TEMPLATE.format(snippet=snippet.resolve()))
    result = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                            cwd=out_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"latexmk failed on {tex}:\n{(result.stdout + result.stderr)[-1500:]}")
    subprocess.run(["latexmk", "-c", tex.name], cwd=out_dir, capture_output=True)
    return tex.with_suffix(".pdf")
