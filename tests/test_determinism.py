"""Builders give the same circuit whatever Python's hash seed, so sinter's
strong_id (and resuming a run from its CSV) does not depend on it."""
import os
import subprocess
import sys

SCRIPT = """
import hashlib
from dizx.builders import BUILDERS
kw = dict(distance=3, n_cycles=3, merge_type="ZZ", observable="ZZ")
for name in ("surgery_circuit", "surgery_bell_prep_fusion_zigzag", "surgery_hybrid_linear_5chain",
             "surgery_hybrid_linear_looped"):
    print(name, hashlib.sha256(str(BUILDERS[name](**kw).circuit).encode()).hexdigest())
for name in ("memory_circuit", "memory_cluster", "memory_fusion"):
    print(name, hashlib.sha256(str(BUILDERS[name](3, 3, "X").circuit).encode()).hexdigest())
"""


def _hashes(seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    return subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True,
                          text=True, check=True).stdout


def test_circuits_do_not_depend_on_hash_seed():
    assert _hashes("1") == _hashes("2")
