"""Opt-in attention algorithms, independent of model/checkpoint architecture."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionOptions:
    method: str = "dense"
    backend: str = "auto"
    query_chunk_size: int = 64
    key_chunk_size: int = 512
    moba_block_size: int = 4096
    moba_top_k: int = 12
    dense_prefix_layers: int = 0
    dense_suffix_layers: int = 0
    dense_after_step: int | None = None

    def __post_init__(self) -> None:
        if self.method not in {"dense", "moba"}:
            raise ValueError(f"unknown attention method: {self.method!r}")
        if self.backend not in {"auto", "pallas", "reference"}:
            raise ValueError(f"unknown sparse attention backend: {self.backend!r}")
        for name in (
            "query_chunk_size", "key_chunk_size", "moba_block_size", "moba_top_k",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dense_prefix_layers", "dense_suffix_layers"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.dense_after_step is not None and (
            type(self.dense_after_step) is not int or self.dense_after_step < 0
        ):
            raise ValueError("dense_after_step must be a nonnegative integer or None")
        if self.method == "dense" and (
            self.dense_prefix_layers or self.dense_suffix_layers or self.dense_after_step is not None
        ):
            raise ValueError("dense layer/recovery options require moba")

    def validate_layers(self, layers: int) -> None:
        if self.dense_prefix_layers + self.dense_suffix_layers > layers:
            raise ValueError("dense prefix/suffix layer counts cannot exceed num_hidden_layers")

    @property
    def needs_dense_backend(self) -> bool:
        return (
            self.method == "dense" or self.dense_prefix_layers > 0 or self.dense_suffix_layers > 0
            or self.dense_after_step is not None
        )
