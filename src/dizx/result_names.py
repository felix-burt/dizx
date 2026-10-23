"""Result file naming for configs on the current (non-legacy) convention.

Per-run sinter CSVs (and their threshold plots) are named

    <family>_<run_type>[_<noise_param>]_mt<merge>_obs<observable>_<decoder>

and the aggregates built across local_probs (correctable-regime PNG/CSV/refine
state) are named

    <run_type>_<family>[_<noise_param>]_mt<merges>_obs<observables>_<decoder>

with the subthreshold alpha plots named the same but `subthresh` in place of
`subthreshold` (e.g. subthresh_loss_bell_...png).

- family: the config's `resource_state` (e.g. linear_k1, bell, bell_zigzag).
  The builder is implied by it, so it is not repeated.
- run_type: the config `name` prefix (crg_local, crg_local_loss, crg_prep,
  erasure, circuit, subthreshold, subthreshold_loss, subthreshold_lambda). It
  replaces the old `noise_model_name` key. subthreshold_loss_target_* configs
  resolve to subthreshold_loss, so they keep sharing those configs' data.
  subthreshold_lambda is its own run type (not subthreshold_loss_lambda) so
  its d=3,5 rows stay out of the CSVs the subthreshold_loss figures read,
  which draw every distance a CSV holds.
- noise_param: the fixed noise levels of the noise channels the file was
  collected under -- pl<p> (circuit noise), pr<p> (resource-state noise),
  n<k> (boost level). Swept channels contribute nothing, so an aggregate
  (built from the unresolved `p_l` config) carries only what is fixed across
  all of its local_probs.
- decoder: pm (pymatching), pm_h (pymatching_heralded), + _corr for
  correlated matching (+ _dec for decomposed-only).

Configs that still set `noise_model_name` are legacy (the mwpf trees under
results/ and results/scale) and keep the old names; see `is_legacy`.
"""
from __future__ import annotations

import copy
from typing import Any

# Longest first, so crg_local_loss wins over crg_local and subthreshold_loss /
# subthreshold_lambda over subthreshold.
RUN_TYPES = ('subthreshold_loss', 'subthreshold_lambda', 'subthreshold', 'crg_local_loss', 'crg_local',
             'crg_prep', 'erasure', 'circuit')

DECODER_TAGS = {'pymatching': 'pm', 'pymatching_heralded': 'pm_h', 'mwpf': 'mwpf'}


def is_legacy(config: dict[str, Any]) -> bool:
    """Legacy configs name their results from `noise_model_name`."""
    return 'noise_model_name' in config


def run_type(config: dict[str, Any]) -> str:
    name = config['name']
    for rt in RUN_TYPES:
        if name == rt or name.startswith(rt + '_'):
            return rt
    raise ValueError(f"config name {name!r} starts with no known run type {RUN_TYPES}")


def family(config: dict[str, Any]) -> str:
    """The config's resource_state (the family in result file names)."""
    return config['resource_state']


def _fixed_probability(channel: dict[str, Any]) -> float | None:
    """A channel's probability if it doesn't move with the swept p."""
    if channel.get('scale') is not None:
        return None
    prob = channel.get('probability')
    return None if prob is None or isinstance(prob, str) else prob


def noise_param(noise_channels: list[dict[str, Any]]) -> str:
    parts = []
    circuit = [p for ch in noise_channels if ch['tag'][0] in ('circuit', 'circuit-graph')
               if (p := _fixed_probability(ch)) is not None]
    if circuit:
        parts.append(f'pl{circuit[0]}')
    graph = [p for ch in noise_channels
             if ch['tag'][0] == 'graph' and ch['kind'] != 'HERALDED_ERASE'
             if (p := _fixed_probability(ch)) is not None]
    if graph:
        parts.append(f'pr{graph[0]}')
    boosts = sorted({ch['n_boost'] for ch in noise_channels if 'n_boost' in ch})
    parts += [f'n{n}' for n in boosts]
    return '_'.join(parts)


def decoder_tag(sinter_args: dict[str, Any]) -> str:
    decoder = sinter_args['decoder']
    tag = DECODER_TAGS.get(decoder, decoder)
    if sinter_args.get('pm_enable_correlations'):
        tag += '_corr'
    elif sinter_args.get('pm_decompose'):
        tag += '_dec'
    return tag


def _axis(values) -> str:
    return '-'.join(str(v) for v in values)


def _tail(noise_channels, merge_types, observables, sinter_args) -> str:
    param = noise_param(noise_channels)
    return (f"{'_' + param if param else ''}_mt{_axis(merge_types)}"
            f"_obs{_axis(observables)}_{decoder_tag(sinter_args)}")


def sinter_stem(config: dict[str, Any], noise_channels: list[dict[str, Any]],
                merge_types, observables) -> str:
    """Stem of a per-run sinter CSV / threshold plot.

    `noise_channels` are the channels actually run (local_prob substituted).
    """
    return (f"{family(config)}_{run_type(config)}"
            f"{_tail(noise_channels, merge_types, observables, config['sinter'])}")


def aggregate_stem(config: dict[str, Any]) -> str:
    """Stem shared by a config's aggregate outputs (crg PNG/CSV/refine state, alpha plots)."""
    sweep = config.get('sweep', {})
    return (f"{run_type(config)}_{family(config)}"
            f"{_tail(config['noise_model'], sweep.get('merge_types', [None]),
                     sweep.get('observables', [None]), config['sinter'])}")


def alpha_plot_stem(config: dict[str, Any]) -> str:
    """Stem of a subthreshold config's diagnostic alpha plot (LER vs distance)."""
    return 'subthresh' + aggregate_stem(config).removeprefix('subthreshold')


def resolve_local_prob(noise_channels: list[dict[str, Any]], local_prob) -> list[dict[str, Any]]:
    """Channels as run at one local_prob: `p_l` takes the value, `p_nl` becomes
    the swept channel (scale 1.0); anything else (fixed values, `p_e`) passes
    through."""
    out = []
    for ch in noise_channels:
        prob = ch.get('probability')
        if prob == 'p_l':
            out.append({**ch, 'probability': local_prob})
        elif prob == 'p_nl':
            out.append({**{k: v for k, v in ch.items() if k != 'probability'}, 'scale': 1.0})
        else:
            out.append(copy.deepcopy(ch))
    return out
