"""Heuristics to locate Transformer layers inside Hugging Face models.

We intentionally keep this small and pragmatic rather than trying to fully
support every architecture on earth.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple


_LAYER_CONTAINER_CANDIDATES: Sequence[Tuple[str, ...]] = (
    # Llama/Mistral/Qwen2 (often model.model.layers)
    ("model", "layers"),
    ("model", "decoder", "layers"),
    ("model", "model", "layers"),
    # OPT/BART-style (model.model.decoder.layers)
    ("model", "decoder", "layers"),
    # GPT-2 style (model.transformer.h)
    ("transformer", "h"),
    ("model", "transformer", "h"),
    # GPT-NeoX (model.gpt_neox.layers)
    ("gpt_neox", "layers"),
    ("model", "gpt_neox", "layers"),
)


def _getattr_chain(obj: Any, chain: Tuple[str, ...]) -> Optional[Any]:
    cur = obj
    for name in chain:
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def find_transformer_layers(model: Any) -> List[Any]:
    """Return a list of Transformer blocks (decoder layers).

    Raises:
        ValueError: if no supported layer container is found.
    """

    for chain in _LAYER_CONTAINER_CANDIDATES:
        layers = _getattr_chain(model, chain)
        if layers is None:
            continue
        try:
            return list(layers)
        except TypeError:
            # Not iterable.
            continue

    raise ValueError(
        "Could not locate transformer layers on this model. "
        "If you are using an uncommon architecture, you may need to extend "
        "openact_collect.tracing.module_resolver.find_transformer_layers()."
    )


def resolve_attention_module(layer: Any) -> Optional[Any]:
    for name in ("self_attn", "attn", "attention"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def resolve_mlp_module(layer: Any) -> Optional[Any]:
    for name in ("mlp", "feed_forward", "ffn", "mixer", "dense"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def find_final_norm(model: Any) -> Any:
    """Resolve the final norm, not a decoder block's internal RMSNorm."""
    for chain in (
        ('model', 'norm'),  # Llama, Mistral, Qwen2/3, Gemma
        ('transformer', 'ln_f'),  # GPT-2, GPT-J, Falcon
        ('gpt_neox', 'final_layer_norm'),
        ('model', 'final_layernorm'),  # Phi
        ('model', 'decoder', 'final_layer_norm'),  # OPT
        ('transformer', 'norm_f'),  # MPT
    ):
        module = _getattr_chain(model, chain)
        if module is not None and hasattr(module, 'register_forward_hook'):
            return module
    raise ValueError(
        'Could not locate the final normalization module. Extend find_final_norm() '
        'for this architecture or disable final_norm capture explicitly.'
    )
