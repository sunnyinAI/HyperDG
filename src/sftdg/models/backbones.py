"""Backbone factory for HyperDG algorithms."""

from __future__ import annotations

from typing import Any

from domainbed import networks

from .vision_transformer import build_vit_backbone


def build_featurizer(input_shape, hparams: dict[str, Any]):
    """Return a feature extractor based on hparams."""
    backbone = hparams.get("backbone", "resnet50").lower()
    if backbone.startswith("vit"):
        return build_vit_backbone(input_shape, hparams)
    return networks.Featurizer(input_shape, hparams)
